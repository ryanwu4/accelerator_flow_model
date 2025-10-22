import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import normflows as nf
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
from tqdm import tqdm
import gc
import argparse

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # File paths
    DATA_DIR = "particle_data"
    MODEL_SAVE_PATH = "flows_from_init_model.pt"
    SCALER_SAVE_PATH = "flows_from_init_scalers.pkl"
    
    # Preprocessing
    PREPROCESSED_DIR = "preprocessed_particles"
    USE_PREPROCESSING = True
    
    # Data loading
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    N_PARTICLES_PER_SAMPLE = 1000
    MAX_FILES = 100  # Set to None to use all files
    
    # Training
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    N_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    
    # Loss function
    N_SW_PROJECTIONS_TRAIN = 50
    N_SW_PROJECTIONS_EVAL = 100
    
    # Flow architecture
    N_FLOW_LAYERS = 8
    HIDDEN_UNITS = 128
    
    # Validation
    N_SAMPLES_VISUALIZE = 5


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    print(f"Using device: {device_name}")
    return device


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_trajectory_robust(file_path):
    """Robustly load trajectory data from .pt file."""
    trajectory = torch.load(file_path, weights_only=False)
    
    if isinstance(trajectory, dict):
        if 'trajectory' in trajectory:
            trajectory = trajectory['trajectory']
        elif 'particles' in trajectory:
            trajectory = trajectory['particles']
        elif 'data' in trajectory:
            trajectory = trajectory['data']
        else:
            keys = list(trajectory.keys())
            if all(isinstance(k, int) for k in keys):
                sorted_keys = sorted(keys)
                trajectory = torch.stack([trajectory[k] for k in sorted_keys])
            else:
                raise ValueError(f"Unknown dictionary format with keys: {keys[:5]}...")
    
    if not isinstance(trajectory, torch.Tensor):
        raise ValueError(f"Could not convert to tensor. Type: {type(trajectory)}")
    
    return trajectory


def subsample_particles(particles, n_target):
    """Randomly subsample particles to target number."""
    n_particles = particles.shape[0]
    if n_particles <= n_target:
        return particles
    
    indices = np.random.choice(n_particles, n_target, replace=False)
    return particles[indices]


# ============================================================================
# DATASET CLASSES
# ============================================================================

class ParticleTransformDataset(Dataset):
    """Dataset for direct particle transformation (initial → final)."""
    
    def __init__(self, file_paths, n_particles):
        self.file_paths = file_paths
        self.n_particles = n_particles
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        trajectory = load_trajectory_robust(self.file_paths[idx])
        
        if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
            raise ValueError(f"NaN/Inf in file: {self.file_paths[idx].name}")
        
        particles_init = trajectory[0].numpy()
        particles_final = trajectory[-1].numpy()
        
        particles_init = subsample_particles(particles_init, self.n_particles)
        particles_final = subsample_particles(particles_final, self.n_particles)
        
        return (
            torch.FloatTensor(particles_init),
            torch.FloatTensor(particles_final)
        )


class PreprocessedTransformDataset(Dataset):
    """Fast dataset using preprocessed particle distributions."""
    
    def __init__(self, init_particles, final_particles):
        self.init_particles = torch.FloatTensor(init_particles)
        self.final_particles = torch.FloatTensor(final_particles)
    
    def __len__(self):
        return len(self.init_particles)
    
    def __getitem__(self, idx):
        return self.init_particles[idx], self.final_particles[idx]


class NormalizedTransformDataset(Dataset):
    """Wraps a dataset to normalize inputs on-the-fly."""
    
    def __init__(self, base_dataset, scaler_initial_list, scaler_final_list):
        self.base_dataset = base_dataset
        self.scaler_initial_list = scaler_initial_list
        self.scaler_final_list = scaler_final_list
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        init, final = self.base_dataset[idx]
        
        init_norm = init.clone()
        final_norm = final.clone()
        
        for dim in range(6):
            init_norm[:, dim] = torch.FloatTensor(
                self.scaler_initial_list[dim].transform(init[:, dim:dim+1].numpy())
            ).squeeze()
            final_norm[:, dim] = torch.FloatTensor(
                self.scaler_final_list[dim].transform(final[:, dim:dim+1].numpy())
            ).squeeze()
        
        return init_norm, final_norm


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class CouplingLayer(nn.Module):
    """Simplified affine coupling layer without conditioning."""
    
    def __init__(self, dim, hidden_dim, reverse_mask=False):
        super().__init__()
        self.dim = dim
        self.d = dim // 2
        self.reverse_mask = reverse_mask
        
        self.net = nn.Sequential(
            nn.Linear(self.d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.d) * 2)
        )
    
    def forward(self, x):
        if self.reverse_mask:
            x1, x2 = x[:, self.d:], x[:, :self.d]
        else:
            x1, x2 = x[:, :self.d], x[:, self.d:]
        
        params = self.net(x1)
        scale = params[:, :(self.dim - self.d)]
        shift = params[:, (self.dim - self.d):]
        
        scale = torch.tanh(scale)
        
        x2_new = x2 * torch.exp(scale) + shift
        log_det = scale.sum(dim=1)
        
        if self.reverse_mask:
            return torch.cat([x2_new, x1], dim=1), log_det
        else:
            return torch.cat([x1, x2_new], dim=1), log_det
    
    def inverse(self, y):
        if self.reverse_mask:
            y1, y2 = y[:, self.d:], y[:, :self.d]
        else:
            y1, y2 = y[:, :self.d], y[:, self.d:]
        
        params = self.net(y1)
        scale = params[:, :(self.dim - self.d)]
        shift = params[:, (self.dim - self.d):]
        
        scale = torch.tanh(scale)
        
        y2_new = (y2 - shift) * torch.exp(-scale)
        log_det = -scale.sum(dim=1)
        
        if self.reverse_mask:
            return torch.cat([y2_new, y1], dim=1), log_det
        else:
            return torch.cat([y1, y2_new], dim=1), log_det


class DirectTransformFlow(nn.Module):
    """Direct flow: transforms initial particles to final particles."""
    
    def __init__(self, latent_dim=6, hidden_dim=128, n_layers=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.flows = nn.ModuleList()
        for i in range(n_layers):
            reverse_mask = (i % 2 == 1)
            self.flows.append(CouplingLayer(latent_dim, hidden_dim, reverse_mask=reverse_mask))
    
    def forward(self, x):
        log_det_sum = 0
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum
    
    def inverse(self, y):
        log_det_sum = 0
        for flow in reversed(self.flows):
            y, log_det = flow.inverse(y)
            log_det_sum = log_det_sum + log_det
        return y, log_det_sum
    
    def transform(self, x_init):
        x_final, _ = self.forward(x_init)
        return x_final


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def generate_projection_vectors(n_projections, dim, seed=42, device='cpu'):
    """Generate deterministic projection vectors for Sliced Wasserstein distance."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    projections = torch.randn(n_projections, dim, device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    return projections


def sliced_wasserstein_loss(x_pred, x_target, projection_vectors):
    """Compute differentiable Sliced Wasserstein distance loss."""
    batch_size, n_particles, dim = x_pred.shape
    n_projections = projection_vectors.shape[0]
    
    proj_pred = x_pred @ projection_vectors.T
    proj_target = x_target @ projection_vectors.T
    
    proj_pred_sorted, _ = torch.sort(proj_pred, dim=1)
    proj_target_sorted, _ = torch.sort(proj_target, dim=1)
    
    w_distances_per_sample = torch.mean(
        torch.abs(proj_pred_sorted - proj_target_sorted), 
        dim=1
    )
    
    w_distances_per_sample = w_distances_per_sample.mean(dim=1)
    
    return w_distances_per_sample.mean()


def sliced_wasserstein_distance(particles1, particles2, n_projections=100):
    """Compute Sliced Wasserstein distance between two 6D particle distributions."""
    distances = []
    
    for _ in range(n_projections):
        theta = np.random.randn(6)
        theta = theta / np.linalg.norm(theta)
        
        proj1 = particles1 @ theta
        proj2 = particles2 @ theta
        
        proj1_sorted = np.sort(proj1)
        proj2_sorted = np.sort(proj2)
        
        w_dist = np.mean(np.abs(proj1_sorted - proj2_sorted))
        distances.append(w_dist)
    
    return np.mean(distances)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config):
    """Load and preprocess particle data."""
    print("="*80)
    print("DATA LOADING")
    print("="*80)
    
    # Find files
    pt_files = sorted([f for f in Path(config.DATA_DIR).glob("*.pt") 
                      if f.name.endswith("_particle_data.pt")])
    print(f"Found {len(pt_files)} particle trajectory files")
    
    # Limit files if needed
    if config.MAX_FILES is not None and len(pt_files) > config.MAX_FILES:
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.choice(len(pt_files), config.MAX_FILES, replace=False)
        pt_files = [pt_files[i] for i in sorted(indices)]
        print(f"Randomly selected {len(pt_files)} files (MAX_FILES={config.MAX_FILES})")
    
    preprocessed_file = f"{config.PREPROCESSED_DIR}/particle_transforms.pt"
    
    if config.USE_PREPROCESSING and Path(preprocessed_file).exists():
        print(f"\nLoading preprocessed data from: {preprocessed_file}")
        cached_data = torch.load(preprocessed_file, weights_only=False)
        init_all = cached_data['initial_particles']
        final_all = cached_data['final_particles']
        pt_files = cached_data['file_paths']
        
        print(f"✓ Loaded {len(init_all)} preprocessed samples")
        print(f"  Initial shape:  {init_all.shape}")
        print(f"  Final shape:    {final_all.shape}")
        
        return init_all, final_all, pt_files
    
    elif config.USE_PREPROCESSING:
        print(f"\nPreprocessing {len(pt_files)} files...")
        print(f"Subsampling to {config.N_PARTICLES_PER_SAMPLE} particles per file")
        
        init_list = []
        final_list = []
        valid_files = []
        
        for file_path in tqdm(pt_files, desc="Processing files"):
            try:
                trajectory = load_trajectory_robust(file_path)
                
                if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                    raise ValueError("NaN/Inf in trajectory")
                
                particles_init = trajectory[0].numpy()
                particles_final = trajectory[-1].numpy()
                
                particles_init = subsample_particles(particles_init, config.N_PARTICLES_PER_SAMPLE)
                particles_final = subsample_particles(particles_final, config.N_PARTICLES_PER_SAMPLE)
                
                init_list.append(particles_init)
                final_list.append(particles_final)
                valid_files.append(file_path)
                
            except Exception as e:
                if len(valid_files) < 5:
                    print(f"\n⚠ Skipping {file_path.name}: {e}")
        
        print(f"\nValid files: {len(valid_files)}")
        
        init_all = np.array(init_list, dtype=np.float32)
        final_all = np.array(final_list, dtype=np.float32)
        pt_files = valid_files
        
        # Save
        Path(config.PREPROCESSED_DIR).mkdir(exist_ok=True)
        
        cached_data = {
            'initial_particles': init_all,
            'final_particles': final_all,
            'file_paths': pt_files,
            'n_particles': config.N_PARTICLES_PER_SAMPLE
        }
        
        torch.save(cached_data, preprocessed_file)
        print(f"✓ Saved preprocessed data: {Path(preprocessed_file).stat().st_size / 1e6:.1f} MB")
        
        return init_all, final_all, pt_files
    
    else:
        # On-the-fly loading
        print("\nUsing on-the-fly loading (no preprocessing)")
        return None, None, pt_files


# ============================================================================
# NORMALIZATION
# ============================================================================

def compute_normalization(train_dataset, init_all, final_all, config):
    """Compute per-dimension normalization statistics."""
    print("\n" + "="*80)
    print("COMPUTING NORMALIZATION STATISTICS (PER-DIMENSION)")
    print("="*80)
    
    dim_names = ['x', 'y', 'z', 'px', 'py', 'pz']
    
    if config.USE_PREPROCESSING and init_all is not None:
        train_indices = train_dataset.indices
        
        train_init_flat = init_all[train_indices].reshape(-1, 6)
        train_final_flat = final_all[train_indices].reshape(-1, 6)
        
        n_norm = min(100000, len(train_init_flat))
        norm_idx_init = np.random.choice(len(train_init_flat), n_norm, replace=False)
        norm_idx_final = np.random.choice(len(train_final_flat), n_norm, replace=False)
        
        print(f"Computing per-dimension statistics from {n_norm} particles...")
        
        scaler_initial_list = []
        scaler_final_list = []
        
        for dim in range(6):
            scaler_init_dim = StandardScaler()
            scaler_init_dim.fit(train_init_flat[norm_idx_init, dim:dim+1])
            scaler_init_dim.scale_ = np.maximum(scaler_init_dim.scale_, 1e-10)
            scaler_initial_list.append(scaler_init_dim)
            
            scaler_final_dim = StandardScaler()
            scaler_final_dim.fit(train_final_flat[norm_idx_final, dim:dim+1])
            scaler_final_dim.scale_ = np.maximum(scaler_final_dim.scale_, 1e-10)
            scaler_final_list.append(scaler_final_dim)
        
        print(f"✓ Computed per-dimension normalization")
        print(f"\nInitial distribution statistics (per dimension):")
        for dim, name in enumerate(dim_names):
            mean = scaler_initial_list[dim].mean_[0]
            std = scaler_initial_list[dim].scale_[0]
            print(f"  {name:3s}: mean={mean:12.6e}, std={std:12.6e}")
        
        print(f"\nFinal distribution statistics (per dimension):")
        for dim, name in enumerate(dim_names):
            mean = scaler_final_list[dim].mean_[0]
            std = scaler_final_list[dim].scale_[0]
            print(f"  {name:3s}: mean={mean:12.6e}, std={std:12.6e}")
        
    else:
        # On-the-fly mode
        init_list = []
        final_list = []
        
        for i, (init, final) in enumerate(train_dataset):
            if i >= 100:
                break
            init_list.append(init.numpy())
            final_list.append(final.numpy())
        
        init_array = np.concatenate(init_list, axis=0)
        final_array = np.concatenate(final_list, axis=0)
        
        scaler_initial_list = []
        scaler_final_list = []
        
        for dim in range(6):
            scaler_init_dim = StandardScaler()
            scaler_init_dim.fit(init_array[:, dim:dim+1])
            scaler_init_dim.scale_ = np.maximum(scaler_init_dim.scale_, 1e-10)
            scaler_initial_list.append(scaler_init_dim)
            
            scaler_final_dim = StandardScaler()
            scaler_final_dim.fit(final_array[:, dim:dim+1])
            scaler_final_dim.scale_ = np.maximum(scaler_final_dim.scale_, 1e-10)
            scaler_final_list.append(scaler_final_dim)
        
        print(f"✓ Computed per-dimension normalization from {len(init_array)} particles")
    
    return scaler_initial_list, scaler_final_list


# ============================================================================
# TRAINING
# ============================================================================

def train_model(config, device):
    """Main training function."""
    
    # Load data
    init_all, final_all, pt_files = load_data(config)
    
    # Create dataset
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    
    if config.USE_PREPROCESSING and init_all is not None:
        dataset = PreprocessedTransformDataset(init_all, final_all)
    else:
        dataset = ParticleTransformDataset(pt_files, config.N_PARTICLES_PER_SAMPLE)
    
    n_files = len(dataset)
    n_test = int(n_files * config.TRAIN_TEST_SPLIT)
    n_train = n_files - n_test
    
    train_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"Training samples:   {len(train_dataset):5d}  ({(1-config.TRAIN_TEST_SPLIT)*100:.0f}%)")
    print(f"Test samples:       {len(test_dataset):5d}  ({config.TRAIN_TEST_SPLIT*100:.0f}%)")
    
    # Compute normalization
    scaler_initial_list, scaler_final_list = compute_normalization(
        train_dataset, init_all, final_all, config
    )
    
    # Create normalized datasets
    train_dataset_norm = NormalizedTransformDataset(train_dataset, scaler_initial_list, scaler_final_list)
    test_dataset_norm = NormalizedTransformDataset(test_dataset, scaler_initial_list, scaler_final_list)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset_norm,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0 if device.type == 'mps' else config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=config.PREFETCH_FACTOR if (device.type != 'mps' and config.NUM_WORKERS > 0) else None
    )
    
    test_loader = DataLoader(
        test_dataset_norm,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0 if device.type == 'mps' else config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=config.PREFETCH_FACTOR if (device.type != 'mps' and config.NUM_WORKERS > 0) else None
    )
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # Initialize model
    print("\n" + "="*80)
    print("BUILDING DIRECT TRANSFORM FLOW")
    print("="*80)
    
    flow_model = DirectTransformFlow(
        latent_dim=6,
        hidden_dim=config.HIDDEN_UNITS,
        n_layers=config.N_FLOW_LAYERS
    )
    
    flow_model = flow_model.to(device)
    
    n_params = sum(p.numel() for p in flow_model.parameters())
    print(f"Parameters: {n_params:,} | Layers: {config.N_FLOW_LAYERS} | Device: {device}")
    print(f"Architecture: Direct transformation (initial → final)")
    
    # Generate projection vectors
    print("\n" + "="*80)
    print("GENERATING DETERMINISTIC PROJECTION VECTORS")
    print("="*80)
    
    projection_vectors_train = generate_projection_vectors(
        n_projections=config.N_SW_PROJECTIONS_TRAIN,
        dim=6,
        seed=config.RANDOM_SEED,
        device=device
    )
    
    projection_vectors_eval = generate_projection_vectors(
        n_projections=config.N_SW_PROJECTIONS_EVAL,
        dim=6,
        seed=config.RANDOM_SEED + 1,
        device=device
    )
    
    print(f"✓ Training projections: {projection_vectors_train.shape}")
    print(f"✓ Evaluation projections: {projection_vectors_eval.shape}")
    
    # Training setup
    optimizer = optim.Adam(flow_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs: {config.N_EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print(f"Loss: Sliced Wasserstein ({config.N_SW_PROJECTIONS_TRAIN} projections)")
    print(f"Train: {len(train_loader)} batches | Test: {len(test_loader)} batches")
    print("="*80)
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(config.N_EPOCHS):
        # Training phase
        flow_model.train()
        train_loss_epoch = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS} [Train]", leave=False)
        for init_batch, final_batch in train_pbar:
            init_batch = init_batch.to(device)
            final_batch = final_batch.to(device)
            
            final_pred, _ = flow_model.forward(init_batch.reshape(-1, 6))
            final_pred = final_pred.reshape(init_batch.shape)
            
            loss = sliced_wasserstein_loss(final_pred, final_batch, projection_vectors_train)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = init_batch.size(0)
            train_loss_epoch += loss.item() * batch_size
            train_pbar.set_postfix({'SW': f'{loss.item():.4e}'})
        
        train_loss_epoch /= len(train_dataset)
        
        # Evaluation phase
        flow_model.eval()
        test_loss_epoch = 0.0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS} [Test]", leave=False)
            for init_batch, final_batch in test_pbar:
                init_batch = init_batch.to(device)
                final_batch = final_batch.to(device)
                
                final_pred, _ = flow_model.forward(init_batch.reshape(-1, 6))
                final_pred = final_pred.reshape(init_batch.shape)
                
                loss = sliced_wasserstein_loss(final_pred, final_batch, projection_vectors_eval)
                
                batch_size = init_batch.size(0)
                test_loss_epoch += loss.item() * batch_size
                test_pbar.set_postfix({'SW': f'{loss.item():.4e}'})
        
        test_loss_epoch /= len(test_dataset)
        
        scheduler.step(test_loss_epoch)
        
        train_losses.append(train_loss_epoch)
        test_losses.append(test_loss_epoch)
        
        # Save best model
        if test_loss_epoch < best_test_loss:
            best_test_loss = test_loss_epoch
            best_epoch = epoch + 1
            best_model_state = flow_model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0 or epoch < 5:
                checkpoint_path = config.MODEL_SAVE_PATH.replace('.pt', f'_checkpoint_epoch{epoch+1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_epoch,
                    'test_loss': test_loss_epoch,
                    'best_test_loss': best_test_loss
                }, checkpoint_path)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:4d}/{config.N_EPOCHS} | "
              f"Train: {train_loss_epoch:8.4f} | "
              f"Test: {test_loss_epoch:8.4f} | "
              f"Best: {best_test_loss:8.4f} (epoch {best_epoch}) | "
              f"LR: {current_lr:.2e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best test loss:     {best_test_loss:.4f}  (epoch {best_epoch})")
    print(f"Final train loss:   {train_losses[-1]:.4f}")
    print(f"Final test loss:    {test_losses[-1]:.4f}")
    
    # Load best model
    flow_model.load_state_dict(best_model_state)
    print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Plot training history
    plot_training_history(train_losses, test_losses, best_test_loss, best_epoch)
    
    # Evaluate model
    evaluate_model(flow_model, test_dataset, scaler_initial_list, scaler_final_list, 
                   config, device)
    
    # Save model
    save_model(flow_model, scaler_initial_list, scaler_final_list, train_losses, 
              test_losses, best_test_loss, best_epoch, pt_files, train_dataset, 
              test_dataset, config)
    
    return flow_model, scaler_initial_list, scaler_final_list


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(train_losses, test_losses, best_test_loss, best_epoch):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full training history
    axes[0].plot(train_losses, alpha=0.7, label='Train Loss', linewidth=2)
    axes[0].plot(test_losses, alpha=0.7, label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Sliced Wasserstein Distance', fontsize=12)
    axes[0].set_title('Training History (Full)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Last 50 epochs for detail
    n_detail = min(50, len(train_losses))
    axes[1].plot(train_losses[-n_detail:], alpha=0.7, label='Train Loss', linewidth=2)
    axes[1].plot(test_losses[-n_detail:], alpha=0.7, label='Test Loss', linewidth=2)
    axes[1].set_xlabel(f'Epoch (last {n_detail})', fontsize=12)
    axes[1].set_ylabel('Sliced Wasserstein Distance', fontsize=12)
    axes[1].set_title('Training History (Detail)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training history plot to: training_history.png")
    plt.close()


def plot_evaluation_results(result):
    """Plot and save evaluation results with histograms on every axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    true_final = result['true']
    pred_final = result['pred']
    
    # Create 3x3 grid for phase space plots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Helper function to add marginal histograms
    def add_marginal_hists(ax, x_true, y_true, x_pred, y_pred, xlabel, ylabel):
        """Add marginal histograms to a 2D scatter plot."""
        # Create divider for adding axes
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 0.8, pad=0.1, sharex=ax)
        ax_histy = divider.append_axes("right", 0.8, pad=0.1, sharey=ax)
        
        # Hide tick labels on marginal axes
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        # Scatter plot in main axes
        ax.scatter(x_true, y_true, alpha=0.5, s=2, label='True', color='blue')
        ax.scatter(x_pred, y_pred, alpha=0.5, s=2, label='Flow Predicted', color='red')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Marginal histograms
        bins_x = 30
        bins_y = 30
        
        ax_histx.hist(x_true, bins=bins_x, alpha=0.5, color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax_histx.hist(x_pred, bins=bins_x, alpha=0.5, color='red', density=True, edgecolor='black', linewidth=0.5)
        ax_histx.set_ylabel('Density', fontsize=9)
        ax_histx.grid(True, alpha=0.3)
        
        ax_histy.hist(y_true, bins=bins_y, alpha=0.5, color='blue', density=True, 
                     orientation='horizontal', edgecolor='black', linewidth=0.5)
        ax_histy.hist(y_pred, bins=bins_y, alpha=0.5, color='red', density=True, 
                     orientation='horizontal', edgecolor='black', linewidth=0.5)
        ax_histy.set_xlabel('Density', fontsize=9)
        ax_histy.grid(True, alpha=0.3)
        
        return ax, ax_histx, ax_histy
    
    # Row 0, Col 0: X-Y position
    ax00 = fig.add_subplot(gs[0, 0])
    add_marginal_hists(ax00, 
                      true_final[:, 0]*1e3, true_final[:, 1]*1e3,
                      pred_final[:, 0]*1e3, pred_final[:, 1]*1e3,
                      'x [mm]', 'y [mm]')
    ax00.set_title('Transverse Position (x-y)', fontsize=13, fontweight='bold')
    ax00.set_aspect('equal')
    
    # Row 0, Col 1: X-Px phase space
    ax01 = fig.add_subplot(gs[0, 1])
    add_marginal_hists(ax01,
                      true_final[:, 0]*1e3, true_final[:, 3],
                      pred_final[:, 0]*1e3, pred_final[:, 3],
                      'x [mm]', 'px [eV/c]')
    ax01.set_title('X Phase Space', fontsize=13, fontweight='bold')
    
    # Row 0, Col 2: Y-Py phase space
    ax02 = fig.add_subplot(gs[0, 2])
    add_marginal_hists(ax02,
                      true_final[:, 1]*1e3, true_final[:, 4],
                      pred_final[:, 1]*1e3, pred_final[:, 4],
                      'y [mm]', 'py [eV/c]')
    ax02.set_title('Y Phase Space', fontsize=13, fontweight='bold')
    
    # Row 1, Col 0: Z-Pz phase space (LONGITUDINAL)
    ax10 = fig.add_subplot(gs[1, 0])
    add_marginal_hists(ax10,
                      true_final[:, 2]*1e3, true_final[:, 5]/1e6,
                      pred_final[:, 2]*1e3, pred_final[:, 5]/1e6,
                      'z [mm]', 'pz [MeV/c]')
    ax10.set_title('Longitudinal Phase Space (z-pz)', fontsize=13, fontweight='bold')
    
    # Row 1, Col 1: X distribution (histogram only)
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.hist(true_final[:, 0]*1e3, bins=40, alpha=0.5, label='True', 
             edgecolor='black', color='blue', density=True)
    ax11.hist(pred_final[:, 0]*1e3, bins=40, alpha=0.5, label='Flow Predicted', 
             edgecolor='black', color='red', density=True)
    ax11.set_xlabel('x [mm]', fontsize=11)
    ax11.set_ylabel('Density', fontsize=11)
    ax11.set_title('X Position Distribution', fontsize=13, fontweight='bold')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # Row 1, Col 2: Y distribution (histogram only)
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.hist(true_final[:, 1]*1e3, bins=40, alpha=0.5, label='True', 
             edgecolor='black', color='blue', density=True)
    ax12.hist(pred_final[:, 1]*1e3, bins=40, alpha=0.5, label='Flow Predicted', 
             edgecolor='black', color='red', density=True)
    ax12.set_xlabel('y [mm]', fontsize=11)
    ax12.set_ylabel('Density', fontsize=11)
    ax12.set_title('Y Position Distribution', fontsize=13, fontweight='bold')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    # Row 2, Col 0: Z distribution (histogram only)
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.hist(true_final[:, 2]*1e3, bins=40, alpha=0.5, label='True', 
             edgecolor='black', color='blue', density=True)
    ax20.hist(pred_final[:, 2]*1e3, bins=40, alpha=0.5, label='Flow Predicted', 
             edgecolor='black', color='red', density=True)
    ax20.set_xlabel('z [mm]', fontsize=11)
    ax20.set_ylabel('Density', fontsize=11)
    ax20.set_title('Z Position Distribution', fontsize=13, fontweight='bold')
    ax20.legend()
    ax20.grid(True, alpha=0.3)
    
    # Row 2, Col 1: Px distribution (histogram only)
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.hist(true_final[:, 3], bins=40, alpha=0.5, label='True', 
             edgecolor='black', color='blue', density=True)
    ax21.hist(pred_final[:, 3], bins=40, alpha=0.5, label='Flow Predicted', 
             edgecolor='black', color='red', density=True)
    ax21.set_xlabel('px [eV/c]', fontsize=11)
    ax21.set_ylabel('Density', fontsize=11)
    ax21.set_title('Px Momentum Distribution', fontsize=13, fontweight='bold')
    ax21.legend()
    ax21.grid(True, alpha=0.3)
    
    # Row 2, Col 2: Py distribution (histogram only)
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.hist(true_final[:, 4], bins=40, alpha=0.5, label='True', 
             edgecolor='black', color='blue', density=True)
    ax22.hist(pred_final[:, 4], bins=40, alpha=0.5, label='Flow Predicted', 
             edgecolor='black', color='red', density=True)
    ax22.set_xlabel('py [eV/c]', fontsize=11)
    ax22.set_ylabel('Density', fontsize=11)
    ax22.set_title('Py Momentum Distribution', fontsize=13, fontweight='bold')
    ax22.legend()
    ax22.grid(True, alpha=0.3)
    
    plt.suptitle(f'Direct Transform Flow: Predicted vs True Distribution\nSW Distance: {result["sw_pred"]:.4e}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved evaluation plot to: evaluation_results.png")
    plt.close()
    
    # Also create a separate plot for Pz distribution (important one!)
    fig_pz, ax_pz = plt.subplots(1, 1, figsize=(10, 6))
    ax_pz.hist(true_final[:, 5]/1e6, bins=50, alpha=0.5, label='True', 
              edgecolor='black', color='blue', density=True, linewidth=1.5)
    ax_pz.hist(pred_final[:, 5]/1e6, bins=50, alpha=0.5, label='Flow Predicted', 
              edgecolor='black', color='red', density=True, linewidth=1.5)
    ax_pz.set_xlabel('pz [MeV/c]', fontsize=14)
    ax_pz.set_ylabel('Density', fontsize=14)
    ax_pz.set_title('Longitudinal Momentum (Pz) Distribution', fontsize=16, fontweight='bold')
    ax_pz.legend(fontsize=12)
    ax_pz.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_pz_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved Pz distribution plot to: evaluation_pz_distribution.png")
    plt.close()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(flow_model, test_dataset, scaler_initial_list, scaler_final_list, 
                   config, device):
    """Evaluate model on test set."""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS ON TEST SET")
    print("="*80)
    
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    flow_model.eval()
    
    n_eval = min(config.N_SAMPLES_VISUALIZE, len(test_dataset))
    print(f"Evaluating {n_eval} test samples")
    test_results = []
    
    with torch.no_grad():
        for i in range(n_eval):
            print(f"\nProcessing sample {i+1}/{n_eval}...")
            
            try:
                init, final_true = test_dataset[i]
                
                init_denorm = init.numpy()
                final_true_denorm = final_true.numpy()
                
                # Normalize initial particles per-dimension
                init_norm = np.zeros_like(init_denorm)
                for dim in range(6):
                    init_norm[:, dim] = scaler_initial_list[dim].transform(
                        init_denorm[:, dim:dim+1]
                    ).squeeze()
                
                init_tensor = torch.FloatTensor(init_norm).to(device)
                
                # Transform through flow
                if device.type == 'mps':
                    flow_model_cpu = flow_model.cpu()
                    final_pred_norm = flow_model_cpu.transform(init_tensor.cpu())
                    flow_model.to(device)
                    gc.collect()
                else:
                    final_pred_norm = flow_model.transform(init_tensor)
                
                # Handle NaN/Inf
                if torch.isnan(final_pred_norm).any() or torch.isinf(final_pred_norm).any():
                    print(f"  Warning: NaN/Inf detected in sample {i+1}")
                    final_pred_norm = torch.where(
                        torch.isnan(final_pred_norm) | torch.isinf(final_pred_norm),
                        torch.zeros_like(final_pred_norm),
                        final_pred_norm
                    )
                
                # Denormalize predictions per-dimension
                final_pred_norm_np = final_pred_norm.cpu().numpy()
                final_pred = np.zeros_like(final_pred_norm_np)
                for dim in range(6):
                    final_pred[:, dim] = scaler_final_list[dim].inverse_transform(
                        final_pred_norm_np[:, dim:dim+1]
                    ).squeeze()
                
                # Compute Sliced Wasserstein distances
                sw_pred_vs_true = sliced_wasserstein_distance(
                    final_pred, final_true_denorm, config.N_SW_PROJECTIONS_EVAL
                )
                sw_init_vs_true = sliced_wasserstein_distance(
                    init_denorm, final_true_denorm, config.N_SW_PROJECTIONS_EVAL
                )
                
                test_results.append({
                    'init': init_denorm,
                    'true': final_true_denorm,
                    'pred': final_pred,
                    'sw_pred': sw_pred_vs_true,
                    'sw_baseline': sw_init_vs_true,
                    'ratio': sw_pred_vs_true / sw_init_vs_true
                })
                
                print(f"  SW(pred, true): {sw_pred_vs_true:.4e}")
                print(f"  SW(init, true): {sw_init_vs_true:.4e}")
                print(f"  Ratio: {sw_pred_vs_true / sw_init_vs_true:.4f}")
                
            except Exception as e:
                print(f"  ERROR in sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary statistics
    if len(test_results) > 0:
        sw_preds = [r['sw_pred'] for r in test_results]
        sw_baselines = [r['sw_baseline'] for r in test_results]
        ratios = [r['ratio'] for r in test_results]
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Successfully evaluated: {len(test_results)}/{n_eval} samples")
        print(f"Average SW(predicted, true):  {np.mean(sw_preds):.4e}")
        print(f"Average SW(initial, true):    {np.mean(sw_baselines):.4e}")
        print(f"Average improvement ratio:    {np.mean(ratios):.4f}")
        print(f"\n✓ Lower ratio = better prediction (ratio < 1 means improvement)")
        
        # Plot first result
        plot_evaluation_results(test_results[0])
    else:
        print("\n⚠ WARNING: No samples were successfully evaluated!")


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(flow_model, scaler_initial_list, scaler_final_list, train_losses, 
               test_losses, best_test_loss, best_epoch, pt_files, train_dataset, 
               test_dataset, config):
    """Save model and scalers."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    checkpoint = {
        'flow_state_dict': flow_model.state_dict(),
        'model_config': {
            'latent_dim': 6,
            'n_layers': config.N_FLOW_LAYERS,
            'hidden_units': config.HIDDEN_UNITS,
        },
        'training_config': {
            'n_epochs': config.N_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'batch_size': config.BATCH_SIZE,
            'n_particles': config.N_PARTICLES_PER_SAMPLE,
            'train_test_split': config.TRAIN_TEST_SPLIT,
            'random_seed': config.RANDOM_SEED
        },
        'training_history': {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss,
            'best_epoch': best_epoch
        },
        'data_info': {
            'n_files': len(pt_files),
            'n_train': len(train_dataset),
            'n_test': len(test_dataset)
        }
    }
    
    torch.save(checkpoint, config.MODEL_SAVE_PATH)
    print(f"✓ Model saved to: {config.MODEL_SAVE_PATH}")
    
    # Save scalers
    scalers = {
        'scaler_initial_list': scaler_initial_list,
        'scaler_final_list': scaler_final_list
    }
    
    with open(config.SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ Scalers saved to: {config.SCALER_SAVE_PATH}")
    
    print(f"\n" + "="*80)
    print("LOADING INSTRUCTIONS")
    print("="*80)
    print("""
# Load checkpoint
checkpoint = torch.load('flows_from_init_model.pt')

# Rebuild model
flow_model = DirectTransformFlow(**checkpoint['model_config'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])

# Load scalers (per-dimension lists)
with open('flows_from_init_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
    scaler_initial_list = scalers['scaler_initial_list']  # List of 6 scalers
    scaler_final_list = scalers['scaler_final_list']      # List of 6 scalers

# Set device and eval mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flow_model = flow_model.to(device).eval()

# To use for prediction:
# 1. Normalize initial particles per-dimension with scaler_initial_list
# 2. Transform through flow: final_norm = flow_model.transform(init_norm)
# 3. Denormalize per-dimension with scaler_final_list
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train direct transform flow model')
    parser.add_argument('--max-files', type=int, default=None, 
                       help='Maximum number of files to use (default: all)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.max_files is not None:
        Config.MAX_FILES = args.max_files
    Config.N_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    # Get device
    device = get_device()
    
    # Train model
    train_model(Config, device)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - ALL OUTPUTS SAVED")
    print("="*80)


if __name__ == "__main__":
    main()
