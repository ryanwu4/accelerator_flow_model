import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
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
    MODEL_SAVE_PATH = "conditional_flow_model.pt"
    SCALER_SAVE_PATH = "conditional_flow_scalers.pkl"
    
    # Preprocessing
    PREPROCESSED_DIR = "preprocessed_particles"
    USE_PREPROCESSING = True
    PREPROCESSING_BATCH_SIZE = 50  # Process this many files at a time to avoid OOM
    
    # Data loading
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    N_PARTICLES_PER_SAMPLE = 1000
    MAX_FILES = None  # Set to None to use all files
    
    # Training
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    N_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    
    # Flow architecture
    N_FLOW_LAYERS = 8
    HIDDEN_UNITS = 128
    CONDITION_DIM = 45  # 6 mean + 6 std + 21 cov + 6 skew + 6 kurt
    
    # Loss weights
    WEIGHT_EMITTANCE_2D = 0.0
    WEIGHT_EMITTANCE_4D = 0.0
    WEIGHT_EMITTANCE_6D = 0.0
    
    # Validation
    N_SW_PROJECTIONS = 100
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


def compute_distribution_statistics(phase_space):
    """
    Compute distributional statistics from 6D phase space data.
    Returns 45 features: mean(6) + std(6) + cov(21) + skew(6) + kurt(6)
    """
    mean = np.mean(phase_space, axis=0)
    std = np.std(phase_space, axis=0)
    cov = np.cov(phase_space.T)
    cov_triu = cov[np.triu_indices(6)]
    skew = stats.skew(phase_space, axis=0)
    kurt = stats.kurtosis(phase_space, axis=0)
    
    statistics = np.concatenate([mean, std, cov_triu, skew, kurt])
    return statistics


def subsample_particles(particles, n_target):
    """Randomly subsample particles to target number."""
    n_particles = particles.shape[0]
    if n_particles <= n_target:
        return particles
    
    indices = np.random.choice(n_particles, n_target, replace=False)
    return particles[indices]


def compute_emittance_torch(particles):
    """
    Compute emittance in PyTorch for a batch of particles.
    particles: (Batch, N_particles, 6)
    Assumes particles are in (x, y, z, px, py, pz) order.
    Returns dictionary with 'x_xp', 'y_yp', 'z_delta', 'fourd', 'sixd' emittances.
    """
    # Indices for canonical pairs in (x, y, z, px, py, pz)
    # x-px: 0, 3
    # y-py: 1, 4
    # z-pz: 2, 5
    
    # Centering
    means = particles.mean(dim=1, keepdim=True)
    centered = particles - means
    
    # Covariance matrix: (Batch, 6, 6)
    # (B, 6, N) @ (B, N, 6) -> (B, 6, 6)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (particles.shape[1] - 1)
    
    emittances = {}
    
    # 2D Emittances
    # eps = sqrt(det(cov_2x2))
    
    # x-px (indices 0, 3)
    cov_x_px = cov[:, [0, 3]][:, :, [0, 3]]
    det_x_px = torch.linalg.det(cov_x_px)
    emittances['x_xp'] = torch.sqrt(torch.abs(det_x_px) + 1e-16)
    
    # y-py (indices 1, 4)
    cov_y_py = cov[:, [1, 4]][:, :, [1, 4]]
    det_y_py = torch.linalg.det(cov_y_py)
    emittances['y_yp'] = torch.sqrt(torch.abs(det_y_py) + 1e-16)
    
    # z-pz (indices 2, 5)
    cov_z_pz = cov[:, [2, 5]][:, :, [2, 5]]
    det_z_pz = torch.linalg.det(cov_z_pz)
    emittances['z_delta'] = torch.sqrt(torch.abs(det_z_pz) + 1e-16)
    
    # 4D Emittance (x, px, y, py -> 0, 3, 1, 4)
    cov_4d = cov[:, [0, 3, 1, 4]][:, :, [0, 3, 1, 4]]
    det_4d = torch.linalg.det(cov_4d)
    emittances['fourd'] = torch.sqrt(torch.abs(det_4d) + 1e-16)
    
    # 6D Emittance (all)
    # Canonical order: 0, 3, 1, 4, 2, 5
    cov_6d = cov[:, [0, 3, 1, 4, 2, 5]][:, :, [0, 3, 1, 4, 2, 5]]
    det_6d = torch.linalg.det(cov_6d)
    emittances['sixd'] = torch.sqrt(torch.abs(det_6d) + 1e-16)
    
    return emittances


# ============================================================================
# DATASET CLASSES
# ============================================================================

class ParticleDistributionDataset(Dataset):
    """
    Dataset for conditional flow training.
    Returns:
        - final_particles: (n_particles, 6) final particles to predict
        - condition: (45,) statistics of initial distribution for conditioning
    """
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
        
        # Compute conditioning statistics from initial distribution
        condition = compute_distribution_statistics(particles_init)
        
        if np.isnan(condition).any() or np.isinf(condition).any():
            raise ValueError(f"NaN/Inf in statistics: {self.file_paths[idx].name}")
        
        return (
            torch.FloatTensor(particles_final),
            torch.FloatTensor(condition)
        )


class PreprocessedFlowDataset(Dataset):
    """Fast dataset using preprocessed particle distributions."""
    def __init__(self, final_particles, conditions):
        self.final_particles = torch.FloatTensor(final_particles)
        self.conditions = torch.FloatTensor(conditions)
    
    def __len__(self):
        return len(self.final_particles)
    
    def __getitem__(self, idx):
        return self.final_particles[idx], self.conditions[idx]


class NormalizedFlowDataset(Dataset):
    """Wraps a dataset to normalize inputs on-the-fly."""
    def __init__(self, base_dataset, scaler_final, scaler_condition):
        self.base_dataset = base_dataset
        self.scaler_final = scaler_final
        self.scaler_condition = scaler_condition
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        final, cond = self.base_dataset[idx]
        
        # Normalize final particles
        final_norm = torch.FloatTensor(
            self.scaler_final.transform(final.numpy().reshape(-1, 6)).reshape(-1, 6)
        )
        
        # Normalize condition
        cond_norm = torch.FloatTensor(
            self.scaler_condition.transform(cond.numpy().reshape(1, -1))[0]
        )
        
        return final_norm, cond_norm


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConditionalCouplingLayer(nn.Module):
    """Single coupling layer with conditioning and alternating masking."""
    def __init__(self, dim, cond_dim, reverse_mask=False):
        super().__init__()
        self.dim = dim
        self.d = dim // 2
        self.reverse_mask = reverse_mask
        
        # Network that takes [z1, condition] and outputs [scale, shift] for z2
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, (dim - self.d) * 2)
        )
    
    def forward(self, z, condition):
        # Apply alternating masking
        if self.reverse_mask:
            z1, z2 = z[:, self.d:], z[:, :self.d]
        else:
            z1, z2 = z[:, :self.d], z[:, self.d:]
        
        params = self.net(torch.cat([z1, condition], dim=1))
        scale = params[:, :(self.dim - self.d)]
        shift = params[:, (self.dim - self.d):]
        
        # Prevent extreme scales - use small range for numerical stability
        scale = torch.tanh(scale) * 0.5
        
        z2_new = z2 * torch.exp(scale) + shift
        log_det = scale.sum(dim=1)
        
        # Reassemble with correct order
        if self.reverse_mask:
            return torch.cat([z2_new, z1], dim=1), log_det
        else:
            return torch.cat([z1, z2_new], dim=1), log_det
    
    def inverse(self, z, condition):
        # Apply alternating masking
        if self.reverse_mask:
            z1, z2 = z[:, self.d:], z[:, :self.d]
        else:
            z1, z2 = z[:, :self.d], z[:, self.d:]
        
        params = self.net(torch.cat([z1, condition], dim=1))
        scale = params[:, :(self.dim - self.d)]
        shift = params[:, (self.dim - self.d):]
        
        scale = torch.tanh(scale) * 0.5
        
        z2_new = (z2 - shift) * torch.exp(-scale)
        log_det = -scale.sum(dim=1)
        
        # Reassemble with correct order
        if self.reverse_mask:
            return torch.cat([z2_new, z1], dim=1), log_det
        else:
            return torch.cat([z1, z2_new], dim=1), log_det


class ConditionalFlowModel(nn.Module):
    """
    Conditional flow: transforms N(0,I) → final distribution.
    The conditioning network outputs parameters that modulate the flow transformations.
    """
    def __init__(self, latent_dim=6, condition_dim=45, hidden_dim=128, n_layers=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Condition encoder
        self.condition_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Flow layers with conditioning and alternating masking
        self.flows = nn.ModuleList()
        for i in range(n_layers):
            reverse_mask = (i % 2 == 1)
            self.flows.append(ConditionalCouplingLayer(latent_dim, hidden_dim, reverse_mask=reverse_mask))
        
        # Base distribution
        self.register_buffer('base_mean', torch.zeros(latent_dim))
        self.register_buffer('base_std', torch.ones(latent_dim))
    
    def forward(self, z, condition):
        """Transform z through flow conditioned on condition."""
        # Encode condition
        cond_encoding = self.condition_net(condition)
        
        log_det_sum = 0
        for flow in self.flows:
            z, log_det = flow(z, cond_encoding)
            log_det_sum = log_det_sum + log_det
        
        return z, log_det_sum
    
    def inverse(self, x, condition):
        """Inverse transform: x → z."""
        cond_encoding = self.condition_net(condition)
        
        log_det_sum = 0
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x, cond_encoding)
            log_det_sum = log_det_sum + log_det
        
        return x, log_det_sum
    
    def log_prob(self, x, condition):
        """Compute log probability."""
        z, log_det = self.inverse(x, condition)
        
        # Base log prob (standard Gaussian) - per dimension
        log_prob_per_dim = -0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)
        log_prob_base = log_prob_per_dim.sum(dim=1)
        
        log_prob = log_prob_base + log_det
        
        return log_prob
    
    def sample(self, n_samples, condition):
        """Sample from conditional distribution."""
        # Sample from base
        z = torch.randn(n_samples, self.latent_dim, device=condition.device)
        
        # Transform
        x, _ = self.forward(z, condition)
        
        return x


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
    
    preprocessed_file = f"{config.PREPROCESSED_DIR}/particle_distributions.pt"
    
    if config.USE_PREPROCESSING and Path(preprocessed_file).exists():
        print(f"\nLoading preprocessed data from: {preprocessed_file}")
        
        # Check file size to warn about memory usage
        file_size_mb = Path(preprocessed_file).stat().st_size / 1e6
        print(f"  File size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 5000:  # Warn if over 5GB
            print(f"  ⚠ Large file detected. This may take a while to load...")
        
        cached_data = torch.load(preprocessed_file, weights_only=False)
        final_all = cached_data['final_particles']
        cond_all = cached_data['conditions']
        pt_files = cached_data['file_paths']
        
        print(f"✓ Loaded {len(final_all)} preprocessed samples")
        print(f"  Final shape:     {final_all.shape}")
        print(f"  Condition shape: {cond_all.shape}")
        print(f"  Memory usage:    ~{(final_all.nbytes + cond_all.nbytes) / 1e6:.1f} MB")
        
        return final_all, cond_all, pt_files
    
    elif config.USE_PREPROCESSING:
        print(f"\nPreprocessing {len(pt_files)} files...")
        print(f"Subsampling to {config.N_PARTICLES_PER_SAMPLE} particles per file")
        print(f"Processing in batches of {config.PREPROCESSING_BATCH_SIZE} to avoid OOM")
        
        # Estimate memory usage
        bytes_per_sample = config.N_PARTICLES_PER_SAMPLE * 6 * 4  # float32
        batch_memory_mb = (config.PREPROCESSING_BATCH_SIZE * bytes_per_sample) / 1e6
        total_memory_mb = (len(pt_files) * bytes_per_sample) / 1e6
        print(f"  Estimated memory per batch: ~{batch_memory_mb:.1f} MB")
        print(f"  Estimated total data size:  ~{total_memory_mb:.1f} MB")
        print(f"  (Using batching saves {total_memory_mb/batch_memory_mb:.1f}x memory)")
        
        # Create batches of files
        n_batches = (len(pt_files) + config.PREPROCESSING_BATCH_SIZE - 1) // config.PREPROCESSING_BATCH_SIZE
        print(f"Total batches: {n_batches}")
        
        valid_files = []
        batch_dir = Path(config.PREPROCESSED_DIR) / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each batch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config.PREPROCESSING_BATCH_SIZE
            end_idx = min(start_idx + config.PREPROCESSING_BATCH_SIZE, len(pt_files))
            batch_files = pt_files[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches} ({len(batch_files)} files)...")
            
            final_list = []
            cond_list = []
            batch_valid_files = []
            
            for file_path in tqdm(batch_files, desc=f"Batch {batch_idx + 1}/{n_batches}", leave=False):
                try:
                    trajectory = load_trajectory_robust(file_path)
                    
                    if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                        raise ValueError("NaN/Inf in trajectory")
                    
                    particles_init = trajectory[0].numpy()
                    particles_final = trajectory[-1].numpy()
                    
                    particles_init = subsample_particles(particles_init, config.N_PARTICLES_PER_SAMPLE)
                    particles_final = subsample_particles(particles_final, config.N_PARTICLES_PER_SAMPLE)
                    
                    # Compute condition
                    condition = compute_distribution_statistics(particles_init)
                    
                    if np.isnan(condition).any() or np.isinf(condition).any():
                        raise ValueError("NaN/Inf in statistics")
                    
                    final_list.append(particles_final)
                    cond_list.append(condition)
                    batch_valid_files.append(file_path)
                    
                except Exception as e:
                    if len(batch_valid_files) < 5:
                        print(f"\n⚠ Skipping {file_path.name}: {e}")
            
            # Save batch
            if len(batch_valid_files) > 0:
                batch_final = np.array(final_list, dtype=np.float32)
                batch_cond = np.array(cond_list, dtype=np.float32)
                
                batch_data = {
                    'final_particles': batch_final,
                    'conditions': batch_cond,
                    'file_paths': batch_valid_files
                }
                
                batch_file = batch_dir / f"batch_{batch_idx:04d}.pt"
                torch.save(batch_data, batch_file)
                print(f"  ✓ Saved batch {batch_idx + 1}: {len(batch_valid_files)} files, {batch_file.stat().st_size / 1e6:.1f} MB")
                
                valid_files.extend(batch_valid_files)
                
                # Clear memory
                del final_list, cond_list, batch_final, batch_cond, batch_data
                gc.collect()
            else:
                print(f"  ⚠ No valid files in batch {batch_idx + 1}")
        
        print(f"\n{'='*80}")
        print(f"Valid files across all batches: {len(valid_files)}")
        
        # Now load all batches and concatenate (memory efficient: load one at a time)
        print(f"\nCombining batches into final dataset...")
        final_all_list = []
        cond_all_list = []
        
        batch_files = sorted(batch_dir.glob("batch_*.pt"))
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            batch_data = torch.load(batch_file, weights_only=False)
            final_all_list.append(batch_data['final_particles'])
            cond_all_list.append(batch_data['conditions'])
            
            # Free memory immediately
            del batch_data
        
        # Concatenate all batches
        print("Concatenating batches...")
        final_all = np.concatenate(final_all_list, axis=0)
        cond_all = np.concatenate(cond_all_list, axis=0)
        pt_files = valid_files
        
        # Clear batch memory
        del final_all_list, cond_all_list
        gc.collect()
        
        # Save final combined file
        Path(config.PREPROCESSED_DIR).mkdir(exist_ok=True)
        
        cached_data = {
            'final_particles': final_all,
            'conditions': cond_all,
            'file_paths': pt_files,
            'n_particles': config.N_PARTICLES_PER_SAMPLE
        }
        
        print(f"Saving combined dataset...")
        torch.save(cached_data, preprocessed_file)
        print(f"✓ Saved preprocessed data: {Path(preprocessed_file).stat().st_size / 1e6:.1f} MB")
        
        # Clean up batch files to save disk space
        print(f"Cleaning up batch files...")
        for batch_file in batch_files:
            batch_file.unlink()
        batch_dir.rmdir()
        print(f"✓ Removed {len(batch_files)} temporary batch files")
        
        return final_all, cond_all, pt_files
    
    else:
        # On-the-fly loading
        print("\nUsing on-the-fly loading (no preprocessing)")
        return None, None, pt_files


# ============================================================================
# NORMALIZATION
# ============================================================================

def compute_normalization(train_dataset, final_all, cond_all, config):
    """Compute normalization statistics."""
    print("\n" + "="*80)
    print("COMPUTING NORMALIZATION STATISTICS")
    print("="*80)
    
    if config.USE_PREPROCESSING and final_all is not None:
        train_indices = train_dataset.indices
        
        # Particle normalization (flatten to compute per-dimension stats)
        train_final_flat = final_all[train_indices].reshape(-1, 6)
        train_cond = cond_all[train_indices]
        
        # Use subset for efficiency
        n_norm = min(100000, len(train_final_flat))
        norm_idx_final = np.random.choice(len(train_final_flat), n_norm, replace=False)
        
        scaler_final = StandardScaler()
        scaler_final.fit(train_final_flat[norm_idx_final])
        
        scaler_condition = StandardScaler()
        scaler_condition.fit(train_cond)
        
        print(f"✓ Computed normalization from {n_norm} particles")
        print(f"\nFinal distribution: Mean: {scaler_final.mean_}, Std: {scaler_final.scale_}")
        
    else:
        # On-the-fly mode: compute from small batch
        final_list = []
        cond_list = []
        
        for i, (final, cond) in enumerate(train_dataset):
            if i >= 100:  # Use subset
                break
            final_list.append(final.numpy())
            cond_list.append(cond.numpy())
        
        final_array = np.concatenate(final_list, axis=0)
        cond_array = np.array(cond_list)
        
        scaler_final = StandardScaler()
        scaler_final.fit(final_array)
        scaler_final.scale_ = np.maximum(scaler_final.scale_, 1e-3)
        
        scaler_condition = StandardScaler()
        scaler_condition.fit(cond_array)
        
        print(f"✓ Computed normalization from {len(final_array)} particles")
        print(f"\nFinal distribution: Mean: {scaler_final.mean_}, Std: {scaler_final.scale_}")
    
    return scaler_final, scaler_condition


# ============================================================================
# SLICED WASSERSTEIN DISTANCE
# ============================================================================

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
# TRAINING
# ============================================================================

def train_model(config, device):
    """Main training function."""
    
    # Load data
    final_all, cond_all, pt_files = load_data(config)
    
    # Create dataset
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    
    if config.USE_PREPROCESSING and final_all is not None:
        dataset = PreprocessedFlowDataset(final_all, cond_all)
    else:
        dataset = ParticleDistributionDataset(pt_files, config.N_PARTICLES_PER_SAMPLE)
    
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
    scaler_final, scaler_condition = compute_normalization(
        train_dataset, final_all, cond_all, config
    )
    
    # Create normalized datasets
    train_dataset_norm = NormalizedFlowDataset(train_dataset, scaler_final, scaler_condition)
    test_dataset_norm = NormalizedFlowDataset(test_dataset, scaler_final, scaler_condition)
    
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
    print("BUILDING CONDITIONAL FLOW MODEL")
    print("="*80)
    
    flow_model = ConditionalFlowModel(
        latent_dim=6,
        condition_dim=config.CONDITION_DIM,
        hidden_dim=config.HIDDEN_UNITS,
        n_layers=config.N_FLOW_LAYERS
    )
    
    flow_model = flow_model.to(device)
    
    n_params = sum(p.numel() for p in flow_model.parameters())
    print(f"Parameters: {n_params:,} | Layers: {config.N_FLOW_LAYERS} | Device: {device}")
    print(f"Architecture: Conditional flow (N(0,I) → final, conditioned on initial moments)")
    
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
    print(f"Loss: Negative Log-Likelihood")
    print(f"Train: {len(train_loader)} batches | Test: {len(test_loader)} batches")
    print("="*80)
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(config.N_EPOCHS):
        # Training phase
        flow_model.train()
        train_loss_epoch = 0.0
        
        for final_batch, cond_batch in train_loader:
            final_batch = final_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            # Flatten particles: (batch * n_particles, 6)
            batch_size = final_batch.size(0)
            n_particles = final_batch.size(1)
            final_flat = final_batch.reshape(batch_size * n_particles, 6)
            
            # Repeat condition for each particle: (batch * n_particles, 45)
            cond_expanded = cond_batch.unsqueeze(1).expand(batch_size, n_particles, -1)
            cond_flat = cond_expanded.reshape(batch_size * n_particles, -1)
            
            # Compute negative log-likelihood
            log_prob = flow_model.log_prob(final_flat, cond_flat)
            # Normalize by number of dimensions for interpretable loss values
            nll_loss = -log_prob.mean() / 6.0
            
            # ----------------------------------------------------------------
            # Emittance Loss
            # ----------------------------------------------------------------
            loss_emittance = 0.0
            if config.WEIGHT_EMITTANCE_2D > 0 or config.WEIGHT_EMITTANCE_4D > 0 or config.WEIGHT_EMITTANCE_6D > 0:
                # Generate samples for emittance calculation
                # We need to sample from the model using the same conditions
                # cond_flat is (Batch * N, 45)
                # We want to generate x_pred matching final_flat structure
                
                # Note: flow_model.sample uses torch.randn which is stochastic.
                # We want gradients to flow through the generated samples to the model parameters.
                # The reparameterization trick is implicit in the flow transformation (z -> x).
                
                x_pred_flat = flow_model.sample(final_flat.size(0), cond_flat)
                
                # Reshape to (Batch, N, 6) for emittance calculation
                x_pred = x_pred_flat.reshape(batch_size, n_particles, 6)
                x_true = final_batch  # Already (Batch, N, 6) and normalized
                
                # Compute emittances (on normalized data)
                # This keeps the loss magnitude reasonable (O(1))
                em_pred = compute_emittance_torch(x_pred)
                em_true = compute_emittance_torch(x_true)
                
                if config.WEIGHT_EMITTANCE_2D > 0:
                    # Percent error: |pred - true| / true
                    # Add epsilon to denominator for stability
                    eps = 1e-20
                    l2d = (torch.abs((em_pred['x_xp'] - em_true['x_xp']) / (em_true['x_xp'] + eps)).mean() +
                           torch.abs((em_pred['y_yp'] - em_true['y_yp']) / (em_true['y_yp'] + eps)).mean() +
                           torch.abs((em_pred['z_delta'] - em_true['z_delta']) / (em_true['z_delta'] + eps)).mean()) / 3.0
                    loss_emittance += config.WEIGHT_EMITTANCE_2D * l2d
                    
                if config.WEIGHT_EMITTANCE_4D > 0:
                    eps = 1e-20
                    l4d = torch.abs((em_pred['fourd'] - em_true['fourd']) / (em_true['fourd'] + eps)).mean()
                    loss_emittance += config.WEIGHT_EMITTANCE_4D * l4d
                    
                if config.WEIGHT_EMITTANCE_6D > 0:
                    eps = 1e-20
                    l6d = torch.abs((em_pred['sixd'] - em_true['sixd']) / (em_true['sixd'] + eps)).mean()
                    loss_emittance += config.WEIGHT_EMITTANCE_6D * l6d
            
            loss = nll_loss + loss_emittance
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item() * batch_size
        
        train_loss_epoch /= len(train_dataset)
        
        # Evaluation phase
        flow_model.eval()
        test_loss_epoch = 0.0
        
        with torch.no_grad():
            for final_batch, cond_batch in test_loader:
                final_batch = final_batch.to(device)
                cond_batch = cond_batch.to(device)
                
                batch_size = final_batch.size(0)
                n_particles = final_batch.size(1)
                final_flat = final_batch.reshape(batch_size * n_particles, 6)
                
                cond_expanded = cond_batch.unsqueeze(1).expand(batch_size, n_particles, -1)
                cond_flat = cond_expanded.reshape(batch_size * n_particles, -1)
                
                log_prob = flow_model.log_prob(final_flat, cond_flat)
                loss = -log_prob.mean() / 6.0
                test_loss_epoch += loss.item() * batch_size
        
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
    evaluate_model(flow_model, test_dataset, scaler_final, scaler_condition, config, device)
    
    # Save model
    save_model(flow_model, scaler_final, scaler_condition, train_losses, test_losses, 
              best_test_loss, best_epoch, pt_files, train_dataset, test_dataset, config)
    
    return flow_model, scaler_final, scaler_condition


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
    axes[0].set_ylabel('Negative Log-Likelihood', fontsize=12)
    axes[0].set_title('Training History (Full)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Last 50 epochs for detail
    n_detail = min(50, len(train_losses))
    axes[1].plot(train_losses[-n_detail:], alpha=0.7, label='Train Loss', linewidth=2)
    axes[1].plot(test_losses[-n_detail:], alpha=0.7, label='Test Loss', linewidth=2)
    axes[1].set_xlabel(f'Epoch (last {n_detail})', fontsize=12)
    axes[1].set_ylabel('Negative Log-Likelihood', fontsize=12)
    axes[1].set_title('Training History (Detail)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conditional_flow_training_history.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved training history plot to: conditional_flow_training_history.png")
    plt.close()


def plot_evaluation_results(result, idx):
    """Plot and save evaluation results with marginal histograms on scatter plots."""
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
    
    plt.suptitle(f'Conditional Flow: Predicted vs True Distribution (Sample {idx})\nSW Distance: {result["sw_pred"]:.4e}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(f'conditional_flow_evaluation_{idx}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a separate plot for Pz distribution (important one!)
    fig_pz, ax_pz = plt.subplots(1, 1, figsize=(10, 6))
    ax_pz.hist(true_final[:, 5]/1e6, bins=50, alpha=0.5, label='True', 
               edgecolor='black', color='blue', density=True, linewidth=1.5)
    ax_pz.hist(pred_final[:, 5]/1e6, bins=50, alpha=0.5, label='Flow Predicted', 
               edgecolor='black', color='red', density=True, linewidth=1.5)
    ax_pz.set_xlabel('pz [MeV/c]', fontsize=14)
    ax_pz.set_ylabel('Density', fontsize=14)
    ax_pz.set_title(f'Longitudinal Momentum (Pz) Distribution - Sample {idx}', fontsize=16, fontweight='bold')
    ax_pz.legend(fontsize=12)
    ax_pz.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'conditional_flow_evaluation_{idx}_pz.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved Pz distribution plot to: conditional_flow_evaluation_{idx}_pz.png")
    plt.close()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(flow_model, test_dataset, scaler_final, scaler_condition, config, device):
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
                # Load test sample (UNNORMALIZED)
                final_true, cond = test_dataset[i]
                
                # Ground truth is already in raw form
                final_true_denorm = final_true.numpy()
                
                # Prepare conditioning (normalize)
                cond_norm = scaler_condition.transform(cond.numpy().reshape(1, -1))
                cond_tensor = torch.FloatTensor(cond_norm).to(device)
                n_particles = final_true.size(0)
                
                # Sample from flow
                if device.type == 'mps':
                    # MPS workaround: sample on CPU
                    flow_model_cpu = flow_model.cpu()
                    cond_expanded = cond_tensor.cpu().expand(n_particles, -1)
                    final_pred_norm = flow_model_cpu.sample(n_particles, cond_expanded)
                    flow_model.to(device)
                    gc.collect()
                else:
                    # CUDA/CPU: sample normally
                    cond_expanded = cond_tensor.expand(n_particles, -1)
                    final_pred_norm = flow_model.sample(n_particles, cond_expanded)
                
                # Handle NaN/Inf
                if torch.isnan(final_pred_norm).any() or torch.isinf(final_pred_norm).any():
                    print(f"  Warning: NaN/Inf detected in sample {i+1}, replacing with zeros")
                    final_pred_norm = torch.where(
                        torch.isnan(final_pred_norm) | torch.isinf(final_pred_norm),
                        torch.zeros_like(final_pred_norm),
                        final_pred_norm
                    )
                
                # Denormalize predictions
                final_pred = scaler_final.inverse_transform(final_pred_norm.cpu().numpy())
                
                # Compute Sliced Wasserstein distance
                sw_pred_vs_true = sliced_wasserstein_distance(
                    final_pred, final_true_denorm, config.N_SW_PROJECTIONS
                )
                
                test_results.append({
                    'true': final_true_denorm,
                    'pred': final_pred,
                    'sw_pred': sw_pred_vs_true
                })
                
                print(f"  SW(pred, true): {sw_pred_vs_true:.4e}")
                
                # Plot results
                plot_evaluation_results(test_results[-1], i)
                
            except Exception as e:
                print(f"  ERROR in sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Summary statistics
    if len(test_results) > 0:
        sw_preds = [r['sw_pred'] for r in test_results]
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Successfully evaluated: {len(test_results)}/{n_eval} samples")
        print(f"Average SW(predicted, true):  {np.mean(sw_preds):.4e}")
        print(f"Std SW(predicted, true):      {np.std(sw_preds):.4e}")
        print(f"Min SW(predicted, true):      {np.min(sw_preds):.4e}")
        print(f"Max SW(predicted, true):      {np.max(sw_preds):.4e}")
    else:
        print("\n⚠ WARNING: No samples were successfully evaluated!")


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(flow_model, scaler_final, scaler_condition, train_losses, test_losses, 
               best_test_loss, best_epoch, pt_files, train_dataset, test_dataset, config):
    """Save model and scalers."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    checkpoint = {
        'flow_state_dict': flow_model.state_dict(),
        'model_config': {
            'latent_dim': 6,
            'condition_dim': config.CONDITION_DIM,
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
        'scaler_final': scaler_final,
        'scaler_condition': scaler_condition
    }
    
    with open(config.SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ Scalers saved to: {config.SCALER_SAVE_PATH}")
    
    print(f"\n" + "="*80)
    print("LOADING INSTRUCTIONS")
    print("="*80)
    print(f"""
# Load checkpoint
checkpoint = torch.load('{config.MODEL_SAVE_PATH}')

# Rebuild model
from train_norm_flow_conditional import ConditionalFlowModel
flow_model = ConditionalFlowModel(**checkpoint['model_config'])
flow_model.load_state_dict(checkpoint['flow_state_dict'])

# Load scalers
with open('{config.SCALER_SAVE_PATH}', 'rb') as f:
    scalers = pickle.load(f)
    scaler_final = scalers['scaler_final']
    scaler_condition = scalers['scaler_condition']

# Set device and eval mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flow_model = flow_model.to(device).eval()
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train conditional normalizing flow model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of flow layers')
    parser.add_argument('--hidden-units', type=int, default=128, help='Hidden units')
    parser.add_argument('--max-files', type=int, default=None, help='Max files to use')
    parser.add_argument('--preprocessing-batch-size', type=int, default=50, 
                        help='Number of files to process at once during preprocessing (lower = less memory)')
    parser.add_argument('--no-preprocess', action='store_true', help='Disable preprocessing')
    
    # Emittance loss weights
    parser.add_argument('--w-2d', type=float, default=0.0, help='Weight for 2D emittance loss')
    parser.add_argument('--w-4d', type=float, default=0.0, help='Weight for 4D emittance loss')
    parser.add_argument('--w-6d', type=float, default=0.0, help='Weight for 6D emittance loss')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.N_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.N_FLOW_LAYERS = args.n_layers
    config.HIDDEN_UNITS = args.hidden_units
    config.MAX_FILES = args.max_files
    config.PREPROCESSING_BATCH_SIZE = args.preprocessing_batch_size
    config.USE_PREPROCESSING = not args.no_preprocess
    config.WEIGHT_EMITTANCE_2D = args.w_2d
    config.WEIGHT_EMITTANCE_4D = args.w_4d
    config.WEIGHT_EMITTANCE_6D = args.w_6d
    
    # Get device
    device = get_device()
    
    # Train model
    train_model(config, device)


if __name__ == '__main__':
    main()
