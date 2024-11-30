import sys
from pathlib import Path
import math

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import time
from typing import Dict, Any
import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from models.transformer import SimpleTransformer
import torch.nn.functional as F
from math import exp
import math

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        """Create a Gaussian window for SSIM calculation."""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                for x in range(window_size)])
            return gauss/gauss.sum()
            
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        # Ensure input tensors are contiguous
        pred = pred.contiguous()
        target = target.contiguous()
        
        # Get dimensions
        b, t, c, h, w = pred.shape
        
        # Reshape tensors
        pred = pred.reshape(b * t, c, h, w)
        target = target.reshape(b * t, c, h, w)
        
        # Move window to the same device as input
        window = self.window.to(pred.device)
        
        return 1 - self.ssim(pred, target, window, self.window_size, self.channel, self.size_average)

class VideoDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"Data file not found: {h5_path}")
            
        # Get total number of sequences across all categories and videos
        self.sequences = []  # List to store (category, video, index) tuples
        
        with h5py.File(h5_path, 'r') as f:
            # Iterate through categories
            for category in f.keys():
                # Iterate through videos in category
                for video in f[category].keys():
                    num_sequences = len(f[category][video]['inputs'])
                    for i in range(num_sequences):
                        self.sequences.append((category, video, i))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        category, video, seq_idx = self.sequences[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            # Access the specific sequence
            inputs = torch.FloatTensor(f[category][video]['inputs'][seq_idx])
            targets = torch.FloatTensor(f[category][video]['outputs'][seq_idx])
            
            # Reshape if needed (B, T, C, H, W)
            if inputs.dim() == 4:  # (T, H, W, C)
                inputs = inputs.permute(0, 3, 1, 2)  # -> (T, C, H, W)
            if targets.dim() == 4:
                targets = targets.permute(0, 3, 1, 2)
                
        return inputs, targets

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.6, ssim_weight=0.4):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_criterion = nn.MSELoss()
        self.ssim_criterion = SSIMLoss()
        
    def forward(self, pred, target):
        # Ensure input tensors are contiguous
        pred = pred.contiguous()
        target = target.contiguous()
        
        # MSE Loss
        mse_loss = self.mse_criterion(pred, target)
        
        # SSIM Loss
        ssim_loss = self.ssim_criterion(pred, target)
        
        # Combine losses
        total_loss = (self.mse_weight * mse_loss + 
                     self.ssim_weight * ssim_loss)
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'ssim_loss': ssim_loss
        }

class TransformerTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 5)
        
        self.setup_directories()
        self.setup_logging()
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Setup loss and optimizer
        self.criterion = CombinedLoss(
            mse_weight=config['mse_weight'],
            ssim_weight=config['ssim_weight']
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        # Initialize metrics tracking
        self.metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'mse': [], 'ssim': [], 'epoch_time': []
        }
        
        # Setup tensorboard
        self.writer = SummaryWriter(
            project_root / f"logs/transformer/train/{model.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Log hyperparameters
        self.writer.add_hparams(
            {
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'mse_weight': config['mse_weight'],
                'ssim_weight': config['ssim_weight'],
                'weight_decay': config.get('weight_decay', 0),
                'dropout': config.get('dropout', 0),
                'gradient_clip': config.get('gradient_clip', 0)
            },
            {'dummy_metric': 0}
        )

    def setup_directories(self):
        """Create organized directory structure"""
        self.metrics_dir = project_root / "metrics/transformer/train"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{self.model.name}_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.log_dir = project_root / "logs/transformer/train"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging with organized directory structure"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.log_dir / f"{self.model.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _log_predictions(self, input_seq, pred_seq, target_seq, global_step):
        input_seq = input_seq.detach().cpu().numpy()
        pred_seq = pred_seq.detach().cpu().numpy()
        target_seq = target_seq.detach().cpu().numpy()
        
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        for i in range(input_seq.shape[0]):
            img = normalize(input_seq[i, 0])
            self.writer.add_image(f'Input/frame_{i}', img, global_step, dataformats='HW')
        
        for i in range(pred_seq.shape[0]):
            pred_img = normalize(pred_seq[i, 0])
            target_img = normalize(target_seq[i, 0])
            
            self.writer.add_image(f'Prediction/frame_{i}', pred_img, global_step, dataformats='HW')
            self.writer.add_image(f'Target/frame_{i}', target_img, global_step, dataformats='HW')

    def save_metrics(self):
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.metrics_file, index=False)
        self.logger.info(f'Saved metrics to {self.metrics_file}')

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics history
            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['mse'].append(val_metrics['mse'])
            self.metrics_history['ssim'].append(val_metrics['ssim'])
            self.metrics_history['epoch_time'].append(epoch_time)
            
            self.save_metrics()
            
            # Log epoch summary
            self.logger.info(f'Epoch {epoch}:')
            self.logger.info(f'Train Loss: {train_loss:.6f}')
            self.logger.info(f'Val Loss: {val_metrics["val_loss"]:.6f}')
            self.logger.info(f'MSE: {val_metrics["mse"]:.6f}')
            self.logger.info(f'SSIM: {val_metrics["ssim"]:.6f}')
            self.logger.info(f'Epoch time: {epoch_time:.2f}s')
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                model_save_path = project_root / f"models/{self.model.name}_best.pth"
                model_save_path.parent.mkdir(exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.best_val_loss,
                    'metrics_history': self.metrics_history,
                    'config': self.config
                }, model_save_path)
                
                self.logger.info(f'Saved best model to {model_save_path}')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f'Early stopping triggered after {epoch} epochs')
                    break

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Don't split the target sequence - use full 5 frames
            predictions, _ = self.model(inputs, targets, training=True)
            
            # Calculate loss using full predictions
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Log progress
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            if batch_idx % self.config['log_interval'] == 0:
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
                
                # Log all 5 frames
                if batch_idx % (self.config['log_interval'] * 10) == 0:
                    self._log_predictions(inputs[0], predictions[0], targets[0], global_step)
        
        avg_loss = total_loss / batch_count
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_ssim = 0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Use full sequence prediction
                predictions, _ = self.model(inputs, targets, training=False)
                
                # Calculate losses with full sequences
                loss_dict = self.criterion(predictions, targets)
                total_loss += loss_dict['loss'].item()
                
                # Calculate metrics for all 5 frames
                predictions_np = predictions.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                # Calculate MSE
                mse = mean_squared_error(targets_np.flatten(), predictions_np.flatten())
                total_mse += mse
                
                # Calculate SSIM
                ssim_score = 0
                for i in range(predictions_np.shape[1]):
                    ssim_score += ssim(
                        targets_np[0, i, 0],
                        predictions_np[0, i, 0],
                        data_range=1.0
                    )
                total_ssim += ssim_score / predictions_np.shape[1]
                
                batch_count += 1
        
        # Calculate averages
        metrics = {
            'val_loss': total_loss / batch_count,
            'mse': total_mse / batch_count,
            'ssim': total_ssim / batch_count
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        return metrics

def main():
    config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
        'warmup_epochs': 2,
        'log_interval': 50,
        'input_frames': 10,
        'output_frames': 5,
        'weight_decay': 0.001,
        'd_model': 16,        # Decrease from 32
        'num_layers': 1,      # Decrease from 2
        'gradient_clip': 1.0,
        'mse_weight': 0.6,
        'ssim_weight': 0.4,
        'gradient_weight': 0.0,  # Removed gradient loss for simplicity
        'dropout': 0.1,         # Add dropout
    }
    
    # Setup datasets
    processed_data_path = project_root / 'processed_data'
    
    print("Loading datasets...")
    train_dataset = VideoDataset(processed_data_path / 'train_sequences.h5')
    val_dataset = VideoDataset(processed_data_path / 'val_sequences.h5')
    print(f"Found {len(train_dataset)} training sequences")
    print(f"Found {len(val_dataset)} validation sequences")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = SimpleTransformer(
        input_channels=1,
        d_model=config['d_model'],
        num_layers=config['num_layers']
    )
    
    # Initialize trainer and start training
    trainer = TransformerTrainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()