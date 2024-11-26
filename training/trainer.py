import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import time
from typing import Dict, Any
import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import torchmetrics
import pandas as pd


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.length = 0
            for category in f.keys():
                for video in f[category].keys():
                    self.length += len(f[category][video]['inputs'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            current_idx = idx
            for category in f.keys():
                for video in f[category].keys():
                    video_data = f[category][video]
                    num_sequences = len(video_data['inputs'])
                    
                    if current_idx < num_sequences:
                        # Load data
                        inputs = torch.FloatTensor(video_data['inputs'][current_idx])
                        outputs = torch.FloatTensor(video_data['outputs'][current_idx])
                        
                        # Ensure correct dimension order (batch, channels, height, width)
                        if inputs.size(1) != 1:  # If channels is not the second dimension
                            inputs = inputs.permute(0, 3, 1, 2)
                            outputs = outputs.permute(0, 3, 1, 2)
                            
                        return inputs, outputs
                    
                    current_idx -= num_sequences


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, gdl_weight=1.0, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.gdl_weight = gdl_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_criterion = nn.MSELoss()
        
    def gradient_difference_loss(self, pred, target):
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Compute GDL
        dx_loss = torch.mean(torch.abs(pred_dx - target_dx))
        dy_loss = torch.mean(torch.abs(pred_dy - target_dy))
        
        return dx_loss + dy_loss
    
    def forward(self, pred, target):
        # Reshape if needed
        if pred.dim() == 5:  # [batch, seq_len, channel, height, width]
            b, t, c, h, w = pred.size()
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)
        
        # MSE Loss
        mse_loss = self.mse_criterion(pred, target)
        
        # Gradient Difference Loss
        gdl_loss = self.gradient_difference_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.mse_weight * mse_loss + 
            self.gdl_weight * gdl_loss
        )
        
        return total_loss


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = None
    ):
        self.model = model
        self.config = config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize the combined loss function
        self.criterion = CombinedLoss(
            mse_weight=1.0,
            gdl_weight=1.0,
            perceptual_weight=0.1
        ).to(self.device)
        
        # Tensorboard writer
        self.writer = SummaryWriter(f"logs/{model.name}_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Log hyperparameters
        self.writer.add_hparams(
            {
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'hidden_channels': model.hidden_channels,
            },
            {'dummy_metric': 0}  # Required by tensorboard
        )
        
        # Initialize metrics tracking
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'mse': [],
            'ssim': [],
            'epoch_time': []
        }
        
        # Create metrics directory
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{model.name}_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    def setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/{self.model.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs, self.config['output_frames'])
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # Log batch loss
            total_loss += loss.item()
            batch_count += 1
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            if batch_idx % 50 == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                               f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                # Log sample predictions periodically
                if batch_idx % (self.config['log_interval'] * 10) == 0:
                    self._log_predictions(inputs[0], outputs[0], targets[0], global_step)
        
        avg_loss = total_loss / batch_count
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss

    def _log_predictions(self, input_seq, pred_seq, target_seq, global_step):
        """Log sample predictions to TensorBoard"""
        # Convert tensors to numpy arrays
        input_seq = input_seq.detach().cpu().numpy()
        pred_seq = pred_seq.detach().cpu().numpy()
        target_seq = target_seq.detach().cpu().numpy()
        
        # Ensure values are in [0, 1] range
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        # Log input sequence
        for i in range(input_seq.shape[0]):
            img = normalize(input_seq[i, 0])  # Take first channel
            self.writer.add_image(f'Input/frame_{i}', img, global_step, dataformats='HW')
        
        # Log predictions and targets
        for i in range(pred_seq.shape[0]):
            pred_img = normalize(pred_seq[i, 0])
            target_img = normalize(target_seq[i, 0])
            
            self.writer.add_image(f'Prediction/frame_{i}', pred_img, global_step, dataformats='HW')
            self.writer.add_image(f'Target/frame_{i}', target_img, global_step, dataformats='HW')

    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_ssim = 0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, self.config['output_frames'])
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate MSE and SSIM
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                mse = mean_squared_error(targets_np.flatten(), outputs_np.flatten())
                total_mse += mse
                
                # Calculate SSIM for each frame
                ssim_score = 0
                for i in range(outputs_np.shape[1]):
                    ssim_score += ssim(
                        targets_np[0, i, 0],
                        outputs_np[0, i, 0],
                        data_range=1.0
                    )
                total_ssim += ssim_score / outputs_np.shape[1]
                
                batch_count += 1
        
        metrics = {
            'val_loss': total_loss / batch_count,
            'mse': total_mse / batch_count,
            'ssim': total_ssim / batch_count
        }
        
        # Log metrics to TensorBoard
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        return metrics

    def save_metrics(self):
        """Save metrics to CSV file"""
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
            
            # Store metrics
            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['mse'].append(val_metrics['mse'])
            self.metrics_history['ssim'].append(val_metrics['ssim'])
            self.metrics_history['epoch_time'].append(epoch_time)
            
            # Save metrics after each epoch
            self.save_metrics()
            
            # Log metrics
            self.logger.info(f'Epoch {epoch}:')
            self.logger.info(f'Train Loss: {train_loss:.6f}')
            self.logger.info(f'Val Loss: {val_metrics["val_loss"]:.6f}')
            self.logger.info(f'MSE: {val_metrics["mse"]:.6f}')
            self.logger.info(f'SSIM: {val_metrics["ssim"]:.6f}')
            self.logger.info(f'Epoch time: {epoch_time:.2f}s')
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                model_save_path = Path(f"models/{self.model.name}_best.pth")
                model_save_path.parent.mkdir(exist_ok=True)
                
                # Save model with metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'metrics_history': self.metrics_history,  # Save metrics history with model
                    'config': self.config  # Save configuration with model
                }, model_save_path)
                
                self.logger.info(f'Saved best model to {model_save_path}')