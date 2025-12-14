import numpy as np
import os
#enable mps fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
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

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # File paths
    DATA_DIR = "../graph_surrogate_shared/particles_data/Archive_4"
    # DATA_DIR = "../particle_data"
    MODEL_SAVE_PATH = "conditional_flow_model.pt"
    SCALER_SAVE_PATH = "conditional_flow_scalers.pkl"
    
    # Preprocessing
    PREPROCESSED_DIR = "preprocessed_stats"
    USE_PREPROCESSING = True
    PREPROCESSING_BATCH_SIZE = 5000  # Process this many files at a time to avoid OOM

    # Data loading
    BATCH_SIZE = 32
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 2
    N_PARTICLES_PER_SAMPLE = 2000
    MAX_FILES = None  # Set to None to use all files, or a number for testing (e.g., 100)

    # Training
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    LEARNING_RATE = 1e-4
    N_EPOCHS = 250
    WEIGHT_DECAY = 1e-5
    
    # Flow architecture
    N_FLOW_LAYERS = 8
    HIDDEN_UNITS = 128
    CONDITION_DIM = 45  # 6 mean + 6 std + 21 cov + 6 skew + 6 kurt
    RQS_N_BINS = 8
    RQS_TAIL_BOUND = 3.0
    
    # Loss weights
    WEIGHT_EMITTANCE_2D = 1.0
    WEIGHT_EMITTANCE_4D = 0.5
    WEIGHT_EMITTANCE_6D = 0.1
    WEIGHT_BEAM_MATRIX = 0.005  # Weight for beam matrix percent error loss
    BEAM_MATRIX_REGULARIZATION = 1e-2 # Regularization term for beam matrix SMAPE denominator
    
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


def compute_beam_matrix_torch(particles):
    """
    Compute beam matrix (covariance matrix) in PyTorch for a batch of particles.
    particles: (Batch, N_particles, 6)
    Assumes particles are in (x, y, z, px, py, pz) order.
    Returns: (Batch, 6, 6) covariance matrix in (x, px, y, py, z, pz) order.
    """
    # Reorder to (x, px, y, py, z, pz)
    # Input indices: 0:x, 1:y, 2:z, 3:px, 4:py, 5:pz
    # Output indices: 0:x, 3:px, 1:y, 4:py, 2:z, 5:pz
    perm = [0, 3, 1, 4, 2, 5]
    particles_reordered = particles[:, :, perm]
    
    # Use torch.cov for each batch element
    batch_size = particles.shape[0]
    cov_list = []
    
    for i in range(batch_size):
        # Transpose to (6, N) as expected by torch.cov
        cov_list.append(torch.cov(particles_reordered[i].T))
        
    return torch.stack(cov_list)


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


def rational_quadratic_spline(inputs, widths, heights, derivatives, inverse=False,
                               tail_bound=3.0, min_bin_width=1e-3,
                               min_bin_height=1e-3, min_derivative=1e-3,
                               max_derivative=10.0, eps=1e-6):
    """Monotonic rational quadratic spline for elementwise transforms with small eps for stability."""
    num_bins = widths.shape[-1]
    left, right = -tail_bound, tail_bound
    bottom, top = -tail_bound, tail_bound

    widths = F.softmax(widths, dim=-1)
    widths = widths * (right - left - min_bin_width * num_bins) + min_bin_width
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = cumwidths + left
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(heights, dim=-1)
    heights = heights * (top - bottom - min_bin_height * num_bins) + min_bin_height
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = cumheights + bottom
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = F.softplus(derivatives) + min_derivative
    derivatives = torch.clamp(derivatives, max=max_derivative)

    inside_interval = (inputs >= left) & (inputs <= right)
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)

    if inverse:
        inputs = torch.clamp(inputs, bottom, top)
        bin_idx = torch.sum(inputs[..., None] >= cumheights[..., 1:], dim=-1)
        bin_idx = torch.clamp(bin_idx, max=num_bins - 1)

        input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        output_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        output_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

        input_bin_widths = torch.clamp(input_bin_widths, min=eps)
        delta = output_bin_heights / input_bin_widths

        derivative_left = derivatives[..., :-1].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        derivative_right = derivatives[..., 1:].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

        # Quadratic coefficients for solving spline inverse
        a = (inputs - output_cumheights) * (derivative_left + derivative_right - 2 * delta) + output_bin_heights * (delta - derivative_left)
        b = output_bin_heights * delta
        c = - (inputs - output_cumheights) * delta

        discriminant = torch.clamp(b.pow(2) - 4 * a * c, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)
        denom = -b - torch.sign(b) * sqrt_disc
        denom = torch.where(denom == 0, denom + eps, denom)
        root = (2 * c) / denom
        root = torch.clamp(root, 0.0, 1.0)

        outputs_inside = root * input_bin_widths + input_cumwidths

        denominator = delta + (derivative_left + derivative_right - 2 * delta) * root * (1 - root)
        denominator = torch.clamp(denominator, min=eps)
        derivative_numerator = delta.pow(2) * (
            derivative_right * root.pow(2)
            + 2 * delta * root * (1 - root)
            + derivative_left * (1 - root).pow(2)
        )
        derivative_denominator = denominator.pow(2)
        derivative_numerator = torch.clamp(derivative_numerator, min=eps)
        derivative_denominator = torch.clamp(derivative_denominator, min=eps)
        logabsdet_forward = torch.log(derivative_numerator) - torch.log(derivative_denominator) - torch.log(input_bin_widths)

        outputs = torch.where(inside_interval, outputs_inside, outputs)
        logabsdet = torch.where(inside_interval, -logabsdet_forward, logabsdet)

    else:
        inputs = torch.clamp(inputs, left, right)
        bin_idx = torch.sum(inputs[..., None] >= cumwidths[..., 1:], dim=-1)
        bin_idx = torch.clamp(bin_idx, max=num_bins - 1)

        input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        output_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        output_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

        input_bin_widths = torch.clamp(input_bin_widths, min=eps)
        delta = output_bin_heights / input_bin_widths

        derivative_left = derivatives[..., :-1].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        derivative_right = derivatives[..., 1:].gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = output_bin_heights * (delta * theta.pow(2) + derivative_left * theta_one_minus_theta)
        denominator = delta + (derivative_left + derivative_right - 2 * delta) * theta_one_minus_theta
        denominator = torch.clamp(denominator, min=eps)

        outputs_inside = output_cumheights + numerator / denominator

        derivative_numerator = delta.pow(2) * (
            derivative_right * theta.pow(2) + 2 * delta * theta_one_minus_theta + derivative_left * (1 - theta).pow(2)
        )
        derivative_denominator = denominator.pow(2)
        derivative_numerator = torch.clamp(derivative_numerator, min=eps)
        derivative_denominator = torch.clamp(derivative_denominator, min=eps)
        input_bin_widths = torch.clamp(input_bin_widths, min=eps)
        logabsdet_inside = torch.log(derivative_numerator) - torch.log(derivative_denominator) - torch.log(input_bin_widths)

        outputs = torch.where(inside_interval, outputs_inside, outputs)
        logabsdet = torch.where(inside_interval, logabsdet_inside, logabsdet)

    # Final safety: remove any lingering non-finite values to prevent cascading NaNs
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=tail_bound, neginf=-tail_bound)
    logabsdet = torch.nan_to_num(logabsdet, nan=0.0, posinf=0.0, neginf=0.0)

    return outputs, logabsdet


class ConditionalRQSplineCouplingLayer(nn.Module):
    """Spline coupling layer with conditioning and alternating masking."""
    def __init__(self, dim, cond_dim, reverse_mask=False, hidden_dim=128, n_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.d = dim // 2
        self.reverse_mask = reverse_mask
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        self.output_size = (dim - self.d) * (3 * n_bins + 1)
        
        self.net = nn.Sequential(
            nn.Linear(self.d + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_size)
        )
    
    def _chunk_params(self, params):
        params = params.view(params.size(0), self.dim - self.d, 3 * self.n_bins + 1)
        widths = params[..., :self.n_bins]
        heights = params[..., self.n_bins:2 * self.n_bins]
        derivatives = params[..., 2 * self.n_bins:]
        return widths, heights, derivatives
    
    def forward(self, z, condition):
        if self.reverse_mask:
            z1, z2 = z[:, self.d:], z[:, :self.d]
        else:
            z1, z2 = z[:, :self.d], z[:, self.d:]
        
        params = self.net(torch.cat([z1, condition], dim=1))
        widths, heights, derivatives = self._chunk_params(params)
        z2_new, logabsdet = rational_quadratic_spline(
            z2, widths, heights, derivatives, inverse=False, tail_bound=self.tail_bound
        )
        log_det = logabsdet.sum(dim=1)
        
        if self.reverse_mask:
            return torch.cat([z2_new, z1], dim=1), log_det
        else:
            return torch.cat([z1, z2_new], dim=1), log_det
    
    def inverse(self, z, condition):
        if self.reverse_mask:
            z1, z2 = z[:, self.d:], z[:, :self.d]
        else:
            z1, z2 = z[:, :self.d], z[:, self.d:]
        
        params = self.net(torch.cat([z1, condition], dim=1))
        widths, heights, derivatives = self._chunk_params(params)
        z2_new, logabsdet = rational_quadratic_spline(
            z2, widths, heights, derivatives, inverse=True, tail_bound=self.tail_bound
        )
        log_det = logabsdet.sum(dim=1)
        
        if self.reverse_mask:
            return torch.cat([z2_new, z1], dim=1), log_det
        else:
            return torch.cat([z1, z2_new], dim=1), log_det


class ConditionalFlowModel(nn.Module):
    """
    Conditional flow with RQ-spline coupling: transforms N(0,I) → final distribution.
    """
    def __init__(self, latent_dim=6, condition_dim=45, hidden_dim=128, n_layers=8, n_bins=8, tail_bound=3.0):
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
        
        self.flows = nn.ModuleList()
        for i in range(n_layers):
            reverse_mask = (i % 2 == 1)
            self.flows.append(
                ConditionalRQSplineCouplingLayer(
                    latent_dim,
                    hidden_dim,
                    reverse_mask=reverse_mask,
                    hidden_dim=hidden_dim,
                    n_bins=n_bins,
                    tail_bound=tail_bound,
                )
            )
        
        self.register_buffer('base_mean', torch.zeros(latent_dim))
        self.register_buffer('base_std', torch.ones(latent_dim))
    
    def forward(self, z, condition):
        cond_encoding = self.condition_net(condition)
        log_det_sum = 0
        for flow in self.flows:
            z, log_det = flow(z, cond_encoding)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum
    
    def inverse(self, x, condition):
        cond_encoding = self.condition_net(condition)
        log_det_sum = 0
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x, cond_encoding)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum
    
    def log_prob(self, x, condition):
        z, log_det = self.inverse(x, condition)
        log_prob_per_dim = -0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)
        log_prob_base = log_prob_per_dim.sum(dim=1)
        return log_prob_base + log_det
    
    def sample(self, n_samples, condition):
        z = torch.randn(n_samples, self.latent_dim, device=condition.device)
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
        n_norm = min(1000000, len(train_final_flat))
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
        n_layers=config.N_FLOW_LAYERS,
        n_bins=config.RQS_N_BINS,
        tail_bound=config.RQS_TAIL_BOUND
    )
    
    flow_model = flow_model.to(device)
    
    n_params = sum(p.numel() for p in flow_model.parameters())
    print(f"Parameters: {n_params:,} | Layers: {config.N_FLOW_LAYERS} | Device: {device}")
    print(f"Architecture: Conditional RQ-spline flow (N(0,I) → final, conditioned on initial moments)")
    
    # Training setup
    optimizer = optim.Adam(flow_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # History storage
    history = {
        'train': {'total': [], 'nll': [], 'emit_2d': [], 'emit_4d': [], 'emit_6d': [], 'beam_matrix': []},
        'test': {'total': [], 'nll': [], 'emit_2d': [], 'emit_4d': [], 'emit_6d': [], 'beam_matrix': []}
    }
    
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Epochs: {config.N_EPOCHS} | Batch: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print(f"Loss: Negative Log-Likelihood + Emittance Terms")
    print(f"Train: {len(train_loader)} batches | Test: {len(test_loader)} batches")
    print("="*80)
    print("\nStarting training...\n")
    
    # Training loop
    for epoch in range(config.N_EPOCHS):
        # Training phase
        flow_model.train()
        
        epoch_losses = {'total': 0.0, 'nll': 0.0, 'emit_2d': 0.0, 'emit_4d': 0.0, 'emit_6d': 0.0, 'beam_matrix': 0.0}
        
        for step, (final_batch, cond_batch) in enumerate(train_loader):
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
            l2d_val = 0.0
            l4d_val = 0.0
            l6d_val = 0.0
            
            l_beam_val = 0.0
            
            if config.WEIGHT_EMITTANCE_2D > 0 or config.WEIGHT_EMITTANCE_4D > 0 or config.WEIGHT_EMITTANCE_6D > 0 or config.WEIGHT_BEAM_MATRIX > 0:
                # Generate samples for emittance calculation
                x_pred_flat = flow_model.sample(final_flat.size(0), cond_flat)
                
                # Reshape to (Batch, N, 6) for emittance calculation
                x_pred = x_pred_flat.reshape(batch_size, n_particles, 6)
                x_true = final_batch  # Already (Batch, N, 6) and normalized
                
                # Compute emittances (on normalized data)
                em_pred = compute_emittance_torch(x_pred)
                em_true = compute_emittance_torch(x_true)
                
                if config.WEIGHT_EMITTANCE_2D > 0:
                    eps = 1e-20
                    l2d = (torch.abs((em_pred['x_xp'] - em_true['x_xp']) / (em_true['x_xp'] + eps)).mean() +
                           torch.abs((em_pred['y_yp'] - em_true['y_yp']) / (em_true['y_yp'] + eps)).mean() +
                           torch.abs((em_pred['z_delta'] - em_true['z_delta']) / (em_true['z_delta'] + eps)).mean()) / 3.0
                    loss_emittance += config.WEIGHT_EMITTANCE_2D * l2d
                    l2d_val = l2d.item()
                    
                if config.WEIGHT_EMITTANCE_4D > 0:
                    eps = 1e-20
                    l4d = torch.abs((em_pred['fourd'] - em_true['fourd']) / (em_true['fourd'] + eps)).mean()
                    loss_emittance += config.WEIGHT_EMITTANCE_4D * l4d
                    l4d_val = l4d.item()
                    
                if config.WEIGHT_EMITTANCE_6D > 0:
                    eps = 1e-20
                    l6d = torch.abs((em_pred['sixd'] - em_true['sixd']) / (em_true['sixd'] + eps)).mean()
                    loss_emittance += config.WEIGHT_EMITTANCE_6D * l6d
                    l6d_val = l6d.item()

                if config.WEIGHT_BEAM_MATRIX > 0:
                    cov_pred = compute_beam_matrix_torch(x_pred)
                    cov_true = compute_beam_matrix_torch(x_true)
                    
                    # Symmetric MAPE with regularization
                    beam_loss = (2 * torch.abs(cov_pred - cov_true) /
                                 (torch.abs(cov_pred) + torch.abs(cov_true) + config.BEAM_MATRIX_REGULARIZATION)).mean()
                    
                    loss_emittance += config.WEIGHT_BEAM_MATRIX * beam_loss
                    l_beam_val = beam_loss.item()
            
            loss = nll_loss + loss_emittance

            optimizer.zero_grad()
            loss.backward()

            # Sanitize gradients in-place to remove NaN/Inf and clamp magnitude
            grad_was_nonfinite = False
            for name, param in flow_model.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grad_was_nonfinite = True
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                param.grad.data.clamp_(-100.0, 100.0)
            if grad_was_nonfinite:
                cond_min = cond_flat.min().item()
                cond_max = cond_flat.max().item()
                final_min = final_flat.min().item()
                final_max = final_flat.max().item()
                print(f"Grad had non-finite values; sanitized and clamped at epoch {epoch} step {step} | cond_min={cond_min:.3e}, cond_max={cond_max:.3e}, final_min={final_min:.3e}, final_max={final_max:.3e}")

            # Detect any remaining non-finite gradients before clipping/step
            grad_is_finite = True
            for name, param in flow_model.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grad_is_finite = False
                    g = param.grad
                    g_nonfinite = g[~torch.isfinite(g)]
                    g_min = g_nonfinite.min().item() if g_nonfinite.numel() > 0 else float('nan')
                    g_max = g_nonfinite.max().item() if g_nonfinite.numel() > 0 else float('nan')
                    cond_min = cond_flat.min().item()
                    cond_max = cond_flat.max().item()
                    final_min = final_flat.min().item()
                    final_max = final_flat.max().item()
                    print(f"Non-finite grad at epoch {epoch} step {step} in {name}: min={g_min:.3e}, max={g_max:.3e} | cond_min={cond_min:.3e}, cond_max={cond_max:.3e}, final_min={final_min:.3e}, final_max={final_max:.3e}")
                    break
            if not grad_is_finite:
                optimizer.zero_grad()
                continue

            total_norm = torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Log gradient norm occasionally
            if step % 100 == 0:  # or epoch-level
                print(f"grad_norm={total_norm.item():.2f} (clipped at 1.0)")
            
            # Accumulate losses
            epoch_losses['total'] += loss.item() * batch_size
            epoch_losses['nll'] += nll_loss.item() * batch_size
            epoch_losses['emit_2d'] += l2d_val * batch_size
            epoch_losses['emit_4d'] += l4d_val * batch_size
            epoch_losses['emit_6d'] += l6d_val * batch_size
            epoch_losses['beam_matrix'] += l_beam_val * batch_size
        
        # Average over dataset
        for k in epoch_losses:
            epoch_losses[k] /= len(train_dataset)
            history['train'][k].append(epoch_losses[k])
        
        train_loss_epoch = epoch_losses['total']
        
        # Evaluation phase
        flow_model.eval()
        test_epoch_losses = {'total': 0.0, 'nll': 0.0, 'emit_2d': 0.0, 'emit_4d': 0.0, 'emit_6d': 0.0, 'beam_matrix': 0.0}
        
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
                nll_loss = -log_prob.mean() / 6.0
                
                # Emittance loss for validation
                loss_emittance = 0.0
                l2d_val = 0.0
                l4d_val = 0.0
                l6d_val = 0.0
                l_beam_val = 0.0
                
                if config.WEIGHT_EMITTANCE_2D > 0 or config.WEIGHT_EMITTANCE_4D > 0 or config.WEIGHT_EMITTANCE_6D > 0 or config.WEIGHT_BEAM_MATRIX > 0:
                    x_pred_flat = flow_model.sample(final_flat.size(0), cond_flat)
                    x_pred = x_pred_flat.reshape(batch_size, n_particles, 6)
                    x_true = final_batch
                    
                    em_pred = compute_emittance_torch(x_pred)
                    em_true = compute_emittance_torch(x_true)
                    
                    if config.WEIGHT_EMITTANCE_2D > 0:
                        eps = 1e-20
                        l2d = (torch.abs((em_pred['x_xp'] - em_true['x_xp']) / (em_true['x_xp'] + eps)).mean() +
                               torch.abs((em_pred['y_yp'] - em_true['y_yp']) / (em_true['y_yp'] + eps)).mean() +
                               torch.abs((em_pred['z_delta'] - em_true['z_delta']) / (em_true['z_delta'] + eps)).mean()) / 3.0
                        loss_emittance += config.WEIGHT_EMITTANCE_2D * l2d
                        l2d_val = l2d.item()
                        
                    if config.WEIGHT_EMITTANCE_4D > 0:
                        eps = 1e-20
                        l4d = torch.abs((em_pred['fourd'] - em_true['fourd']) / (em_true['fourd'] + eps)).mean()
                        loss_emittance += config.WEIGHT_EMITTANCE_4D * l4d
                        l4d_val = l4d.item()
                        
                    if config.WEIGHT_EMITTANCE_6D > 0:
                        eps = 1e-20
                        l6d = torch.abs((em_pred['sixd'] - em_true['sixd']) / (em_true['sixd'] + eps)).mean()
                        loss_emittance += config.WEIGHT_EMITTANCE_6D * l6d
                        l6d_val = l6d.item()

                    if config.WEIGHT_BEAM_MATRIX > 0:
                        cov_pred = compute_beam_matrix_torch(x_pred)
                        cov_true = compute_beam_matrix_torch(x_true)
                        
                        # Symmetric MAPE with regularization
                        beam_loss = (2 * torch.abs(cov_pred - cov_true) /
                                     (torch.abs(cov_pred) + torch.abs(cov_true) + config.BEAM_MATRIX_REGULARIZATION)).mean()
                        
                        loss_emittance += config.WEIGHT_BEAM_MATRIX * beam_loss
                        l_beam_val = beam_loss.item()
                
                loss = nll_loss + loss_emittance
                
                test_epoch_losses['total'] += loss.item() * batch_size
                test_epoch_losses['nll'] += nll_loss.item() * batch_size
                test_epoch_losses['emit_2d'] += l2d_val * batch_size
                test_epoch_losses['emit_4d'] += l4d_val * batch_size
                test_epoch_losses['emit_6d'] += l6d_val * batch_size
                test_epoch_losses['beam_matrix'] += l_beam_val * batch_size
        
        for k in test_epoch_losses:
            test_epoch_losses[k] /= len(test_dataset)
            history['test'][k].append(test_epoch_losses[k])
        
        test_loss_epoch = test_epoch_losses['total']
        
        scheduler.step(test_loss_epoch)
        
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
                    'best_test_loss': best_test_loss,
                    'history': history
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
    print(f"Final train loss:   {history['train']['total'][-1]:.4f}")
    print(f"Final test loss:    {history['test']['total'][-1]:.4f}")
    
    # Load best model
    flow_model.load_state_dict(best_model_state)
    print(f"\n✓ Loaded best model from epoch {best_epoch}")
    
    # Plot training history
    plot_training_history(history, best_test_loss, best_epoch)
    
    # Evaluate model
    evaluate_model(flow_model, test_dataset, scaler_final, scaler_condition, config, device)
    
    # Save model
    save_model(flow_model, scaler_final, scaler_condition, history, 
              best_test_loss, best_epoch, pt_files, train_dataset, test_dataset, config)
    
    return flow_model, scaler_final, scaler_condition


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, best_test_loss, best_epoch):
    """Plot and save training history with breakdown of loss terms."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract total losses
    train_total = history['train']['total']
    test_total = history['test']['total']
    
    # Plot 1: Total Loss
    axes[0].plot(train_total, alpha=0.7, label='Train Total', linewidth=2, color='blue')
    axes[0].plot(test_total, alpha=0.7, label='Test Total', linewidth=2, color='orange')
    axes[0].axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss History', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss Breakdown (Train)
    # Plot NLL and weighted emittance terms
    epochs = range(len(train_total))
    axes[1].plot(epochs, history['train']['nll'], alpha=0.7, label='NLL', linewidth=2)
    
    if any(v > 0 for v in history['train']['emit_2d']):
        axes[1].plot(epochs, history['train']['emit_2d'], alpha=0.6, label='Emit 2D (weighted)', linestyle='--')
    if any(v > 0 for v in history['train']['emit_4d']):
        axes[1].plot(epochs, history['train']['emit_4d'], alpha=0.6, label='Emit 4D (weighted)', linestyle='--')
    if any(v > 0 for v in history['train']['emit_6d']):
        axes[1].plot(epochs, history['train']['emit_6d'], alpha=0.6, label='Emit 6D (weighted)', linestyle='--')
    if any(v > 0 for v in history['train']['beam_matrix']):
        axes[1].plot(epochs, history['train']['beam_matrix'], alpha=0.6, label='Beam Matrix (weighted)', linestyle='--')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss Component', fontsize=12)
    axes[1].set_title('Training Loss Breakdown', fontsize=14, fontweight='bold')
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
    
    plt.suptitle(f'Conditional Flow: Predicted vs True Distribution (Sample {idx})\nSW Distance: {result["sw_pred"]:.4e} | 6D Emit Err: {result["err_6d"]:.1f}%', 
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

    # Plot Beam Matrix Comparison
    fig_cov, axes_cov = plt.subplots(1, 3, figsize=(18, 5))
    
    # Reorder to canonical (x, px, y, py, z, pz)
    perm = [0, 3, 1, 4, 2, 5]
    true_reordered = true_final[:, perm]
    pred_reordered = pred_final[:, perm]
    
    # Compute covariance matrices
    cov_true = np.cov(true_reordered.T)
    cov_pred = np.cov(pred_reordered.T)
    
    # Calculate percent error matrix
    eps = 1e-20
    cov_error = 100 * np.abs((cov_pred - cov_true) / (np.abs(cov_true) + eps))
    
    # Labels
    labels = ['x', 'px', 'y', 'py', 'z', 'pz']
    
    # Plot True
    im0 = axes_cov[0].imshow(cov_true, cmap='viridis')
    axes_cov[0].set_title('True Beam Matrix', fontsize=14)
    axes_cov[0].set_xticks(range(6))
    axes_cov[0].set_yticks(range(6))
    axes_cov[0].set_xticklabels(labels)
    axes_cov[0].set_yticklabels(labels)
    plt.colorbar(im0, ax=axes_cov[0])
    
    # Annotate True
    for i in range(6):
        for j in range(6):
            text = axes_cov[0].text(j, i, f"{cov_true[i, j]:.1e}",
                                   ha="center", va="center", color="w", fontsize=8)
    
    # Plot Predicted
    im1 = axes_cov[1].imshow(cov_pred, cmap='viridis')
    axes_cov[1].set_title('Predicted Beam Matrix', fontsize=14)
    axes_cov[1].set_xticks(range(6))
    axes_cov[1].set_yticks(range(6))
    axes_cov[1].set_xticklabels(labels)
    axes_cov[1].set_yticklabels(labels)
    plt.colorbar(im1, ax=axes_cov[1])
    
    # Annotate Predicted
    for i in range(6):
        for j in range(6):
            text = axes_cov[1].text(j, i, f"{cov_pred[i, j]:.1e}",
                                   ha="center", va="center", color="w", fontsize=8)
    
    # Plot Error
    im2 = axes_cov[2].imshow(cov_error, cmap='Reds', vmin=0, vmax=100)
    axes_cov[2].set_title('Percent Error (%)', fontsize=14)
    axes_cov[2].set_xticks(range(6))
    axes_cov[2].set_yticks(range(6))
    axes_cov[2].set_xticklabels(labels)
    axes_cov[2].set_yticklabels(labels)
    plt.colorbar(im2, ax=axes_cov[2])
    
    # Annotate Error
    for i in range(6):
        for j in range(6):
            val = cov_error[i, j]
            color = "white" if val > 50 else "black"
            text = axes_cov[2].text(j, i, f"{val:.1f}",
                                   ha="center", va="center", color=color, fontsize=8)
    
    plt.suptitle(f'Beam Matrix Comparison - Sample {idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'conditional_flow_evaluation_{idx}_beam_matrix.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved Beam Matrix plot to: conditional_flow_evaluation_{idx}_beam_matrix.png")
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
                
                # Compute Emittance Errors (Percent)
                # Need to reshape for compute_emittance_torch: (1, N, 6)
                x_pred_tensor = torch.FloatTensor(final_pred).unsqueeze(0)
                x_true_tensor = torch.FloatTensor(final_true_denorm).unsqueeze(0)
                
                em_pred = compute_emittance_torch(x_pred_tensor)
                em_true = compute_emittance_torch(x_true_tensor)
                
                # Calculate percent errors
                eps = 1e-20
                err_2d_x = torch.abs((em_pred['x_xp'] - em_true['x_xp']) / (em_true['x_xp'] + eps)).item() * 100
                err_2d_y = torch.abs((em_pred['y_yp'] - em_true['y_yp']) / (em_true['y_yp'] + eps)).item() * 100
                err_2d_z = torch.abs((em_pred['z_delta'] - em_true['z_delta']) / (em_true['z_delta'] + eps)).item() * 100
                err_4d = torch.abs((em_pred['fourd'] - em_true['fourd']) / (em_true['fourd'] + eps)).item() * 100
                err_6d = torch.abs((em_pred['sixd'] - em_true['sixd']) / (em_true['sixd'] + eps)).item() * 100
                
                test_results.append({
                    'true': final_true_denorm,
                    'pred': final_pred,
                    'sw_pred': sw_pred_vs_true,
                    'err_2d_x': err_2d_x,
                    'err_2d_y': err_2d_y,
                    'err_2d_z': err_2d_z,
                    'err_4d': err_4d,
                    'err_6d': err_6d
                })
                
                print(f"  SW: {sw_pred_vs_true:.4e} | Err 6D: {err_6d:.2f}% | Err 4D: {err_4d:.2f}%")
                
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
        err_6d = [r['err_6d'] for r in test_results]
        err_4d = [r['err_4d'] for r in test_results]
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Successfully evaluated: {len(test_results)}/{n_eval} samples")
        print(f"Average SW Distance:      {np.mean(sw_preds):.4e}")
        print(f"Average 6D Emittance Err: {np.mean(err_6d):.2f}%")
        print(f"Average 4D Emittance Err: {np.mean(err_4d):.2f}%")
        
        # Detailed breakdown
        print("\nDetailed Emittance Errors (Average %):")
        print(f"  x-px:    {np.mean([r['err_2d_x'] for r in test_results]):.2f}%")
        print(f"  y-py:    {np.mean([r['err_2d_y'] for r in test_results]):.2f}%")
        print(f"  z-pz:    {np.mean([r['err_2d_z'] for r in test_results]):.2f}%")
        print(f"  4D:      {np.mean(err_4d):.2f}%")
        print(f"  6D:      {np.mean(err_6d):.2f}%")
    else:
        print("\n⚠ WARNING: No samples were successfully evaluated!")


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(flow_model, scaler_final, scaler_condition, history, 
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
            'n_bins': flow_model.flows[0].n_bins,
            'tail_bound': flow_model.flows[0].tail_bound,
            'coupling': 'conditional_rq_spline'
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
            'history': history,
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    # Get device
    device = get_device()
    
    # Train model
    train_model(config, device)


if __name__ == '__main__':
    main()
