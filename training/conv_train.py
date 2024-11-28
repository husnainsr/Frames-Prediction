import sys
from pathlib import Path

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
from models.conv_lstm import ConvLSTMPredictor

class VideoDataset(Dataset):
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
                        inputs = torch.FloatTensor(video_data['inputs'][current_idx])
                        outputs = torch.FloatTensor(video_data['outputs'][current_idx])
                        
                        if inputs.size(1) != 1:
                            inputs = inputs.permute(0, 3, 1, 2)
                            outputs = outputs.permute(0, 3, 1, 2)
                            
                        return inputs, outputs
                    
                    current_idx -= num_sequences

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=1.0, gdl_weight=1.0, perceptual_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.ssim_weight = ssim_weight
        self.gdl_weight = gdl_weight
        self.perceptual_weight = perceptual_weight
        
    def ssim_loss(self, pred, target, window_size=11):
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.view(1, 1, size, 1) * g.view(1, 1, 1, size)
        
        window = gaussian_window(window_size).to(pred.device)
        
        def _ssim(img1, img2, window):
            mu1 = nn.functional.conv2d(img1, window, padding=window_size//2, groups=1)
            mu2 = nn.functional.conv2d(img2, window, padding=window_size//2, groups=1)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
            sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
            sigma12 = nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
        
        return 1 - _ssim(pred, target, window)
        
    def gradient_difference_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        dx_loss = torch.mean(torch.abs(pred_dx - target_dx))
        dy_loss = torch.mean(torch.abs(pred_dy - target_dy))
        
        return dx_loss + dy_loss
    
    def forward(self, pred, target):
        if pred.dim() == 5:
            b, t, c, h, w = pred.size()
            pred = pred.view(b * t, c, h, w)
            target = target.view(b * t, c, h, w)
        
        ssim_loss = self.ssim_loss(pred, target)
        gdl_loss = self.gradient_difference_loss(pred, target)
        
        return self.ssim_weight * ssim_loss + self.gdl_weight * gdl_loss

class ModelTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = None):
        self.model = model
        self.config = config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.setup_directories()
        self.setup_logging()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = CombinedLoss(ssim_weight=1.0, gdl_weight=1.0).to(self.device)
        
        self.writer = SummaryWriter(
            project_root / f"logs/conv/train/{model.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        self.writer.add_hparams(
            {
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'hidden_channels': model.hidden_channels,
            },
            {'dummy_metric': 0}
        )
        
        self.metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'mse': [], 'ssim': [], 'epoch_time': []
        }

    def setup_directories(self):
        """Create organized directory structure"""
        self.metrics_dir = project_root / "metrics/conv/train"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{self.model.name}_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.log_dir = project_root / "logs/conv/train"
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
            
            total_loss += loss.item()
            batch_count += 1
            
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            if batch_idx % 50 == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                               f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
                if batch_idx % (self.config['log_interval'] * 10) == 0:
                    self._log_predictions(inputs[0], outputs[0], targets[0], global_step)
        
        avg_loss = total_loss / batch_count
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        return avg_loss

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
                
                outputs_np = outputs.cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                mse = mean_squared_error(targets_np.flatten(), outputs_np.flatten())
                total_mse += mse
                
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
        
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        return metrics

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
            
            self.metrics_history['epoch'].append(epoch)
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_metrics['val_loss'])
            self.metrics_history['mse'].append(val_metrics['mse'])
            self.metrics_history['ssim'].append(val_metrics['ssim'])
            self.metrics_history['epoch_time'].append(epoch_time)
            
            self.save_metrics()
            
            self.logger.info(f'Epoch {epoch}:')
            self.logger.info(f'Train Loss: {train_loss:.6f}')
            self.logger.info(f'Val Loss: {val_metrics["val_loss"]:.6f}')
            self.logger.info(f'MSE: {val_metrics["mse"]:.6f}')
            self.logger.info(f'SSIM: {val_metrics["ssim"]:.6f}')
            self.logger.info(f'Epoch time: {epoch_time:.2f}s')
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                model_save_path = project_root / f"models/{self.model.name}_best.pth"
                model_save_path.parent.mkdir(exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'metrics_history': self.metrics_history,
                    'config': self.config
                }, model_save_path)
                
                self.logger.info(f'Saved best model to {model_save_path}')

def main():
    config = {
        'batch_size': 16,
        'learning_rate': 0.0001,
        'epochs': 30,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'log_interval': 50,
        'input_frames': 10,
        'output_frames': 5,
        'checkpoint_interval': 5,
        'augmentation': False
    }
    
    train_dataset = VideoDataset(project_root / 'processed_data/train_sequences.h5')
    val_dataset = VideoDataset(project_root / 'processed_data/val_sequences.h5')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    model = ConvLSTMPredictor(
        input_channels=1,
        hidden_channels=128,
        kernel_size=3
    )
    
    trainer = ModelTrainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()