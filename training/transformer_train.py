import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Define important paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "processed_data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create necessary directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from typing import Dict, Any
import h5py
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models.transformer import TransformerModel
from training.predrnn_train import VideoDataset, CombinedLoss
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import pandas as pd

class TransformerTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = None):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        self.setup_directories()
        self.setup_logging()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.criterion = CombinedLoss().to(self.device)

        self.writer = SummaryWriter(
            LOG_DIR / "transformer/train" / f"{model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        self.metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'mse': [], 'ssim': [], 'epoch_time': []
        }

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )

    def setup_directories(self):
        self.metrics_dir = BASE_DIR / "metrics/transformer/train"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / f"{self.model.__class__.__name__}_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"

        self.log_dir = LOG_DIR / "transformer/train"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        log_file = self.log_dir / f"{self.model.__class__.__name__}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
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

        for i in range(input_seq.shape[1]):
            img = normalize(input_seq[0, i, 0])
            self.writer.add_image(f'Input/frame_{i}', img, global_step, dataformats='HW')

        for i in range(pred_seq.shape[1]):
            pred_img = normalize(pred_seq[0, i, 0])
            target_img = normalize(target_seq[0, i, 0])

            self.writer.add_image(f'Prediction/frame_{i}', pred_img, global_step, dataformats='HW')
            self.writer.add_image(f'Target/frame_{i}', target_img, global_step, dataformats='HW')

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
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
                    self._log_predictions(inputs, outputs, targets, global_step)

        self.scheduler.step()
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
                outputs = self.model(inputs)

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
                model_save_path = MODEL_DIR / f"transformer/{self.model.__class__.__name__}_best.pth"
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
        'epochs': 150,
        'weight_decay': 1e-4,
        'log_interval': 10,
        'input_frames': 10,
        'output_frames': 5,
        'hidden_dim': 128,
        'num_layers': 3,
        'num_heads': 8
    }

    train_dataset = VideoDataset(DATA_DIR / 'train_sequences.h5')
    val_dataset = VideoDataset(DATA_DIR / 'val_sequences.h5')

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

    model = TransformerModel(
        input_dim=64*64,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        output_dim=64*64,
        seq_length=config['input_frames']
    )

    trainer = TransformerTrainer(model, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main() 