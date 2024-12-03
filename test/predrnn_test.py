import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import h5py
import numpy as np
from models.predrnn import PredRNN
import logging
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2
from typing import Dict, Union
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm

class PredRNNTester:
    def __init__(self, model_path: str, device: str = None):
        """Initialize the model tester"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories first (creates necessary paths)
        self.test_dir = Path(__file__).parent / "results" / "predrnn"
        self.comparison_dir = self.test_dir / "comparisons"
        self.metrics_dir = self.test_dir / "metrics"
        
        # Create all directories
        for directory in [self.test_dir, self.comparison_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.load_model(model_path)
        self.model.eval()

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.test_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str):
        """Load the trained model and its configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = PredRNN(
            input_channels=1,
            hidden_channels=checkpoint['config']['hidden_channels'],
            num_layers=checkpoint['config']['num_layers'],
            num_frames_output=checkpoint['config']['output_frames']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
        with open(self.test_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def create_comparison_gif(self, pred_frames: np.ndarray, target_frames: np.ndarray, 
                        path: str, fps: int = 10):
        """Create a side-by-side comparison GIF"""
        try:
            # Ensure the directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize frames to 0-255 range
            pred_frames = (pred_frames * 255).astype(np.uint8)
            target_frames = (target_frames * 255).astype(np.uint8)
            
            # Get dimensions and scale up the images (2x)
            h, w = pred_frames[0].shape
            scale = 2
            h, w = h * scale, w * scale
            combined_w = w * 2 + 50  # Reduced space between images
            
            # Prepare frames for GIF
            gif_frames = []
            
            for pred, target in zip(pred_frames, target_frames):
                # Resize frames
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                target = cv2.resize(target, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Create a white background
                combined = np.ones((h, combined_w), dtype=np.uint8) * 255
                
                # Place the frames
                combined[:, :w] = pred
                combined[:, w+50:] = target
                
                # Convert to RGB for adding colored text
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
                
                # Add labels with smaller text
                font_scale = 0.8
                thickness = 1
                cv2.putText(combined_rgb, 'Pred', (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), thickness)
                cv2.putText(combined_rgb, 'GT', (w+60, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
                
                # Convert from BGR to RGB
                combined_rgb = cv2.cvtColor(combined_rgb, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(combined_rgb)
                gif_frames.append(pil_image)
            
            # Save as GIF
            duration = 1000 // fps  # Convert fps to milliseconds per frame
            gif_frames[0].save(
                path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=0
            )
            
            self.logger.info(f"Saved comparison GIF to {path}")
            
        except Exception as e:
            self.logger.error(f"Error creating comparison GIF: {str(e)}")
            raise

    @torch.no_grad()
    def test(self, test_data_path: str, num_sequences: int = None):
        """Run testing on multiple sequences and save results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create metrics files
        sequence_metrics_path = self.metrics_dir / f'sequence_metrics_{self.model.name}_{timestamp}.csv'
        overall_metrics_path = self.metrics_dir / f'overall_metrics_{self.model.name}_{timestamp}.csv'
        
        # Initialize lists to store all metrics for overall calculation
        all_mse = []
        all_ssim = []
        category_metrics = {}  # To store metrics per category
        
        with open(sequence_metrics_path, 'w') as seq_file:
            seq_file.write('category,video_name,sequence_id,mse,ssim\n')
        
        try:
            with h5py.File(test_data_path, 'r') as f:
                # Analysis of data structure
                print("\nData Structure Analysis:")
                total_sequences = 0
                for category in f.keys():
                    category_sequences = 0
                    print(f"\nCategory: {category}")
                    for video in f[category].keys():
                        num_seqs = len(f[category][video]['inputs'])
                        print(f"  Video: {video} - Sequences: {num_seqs}")
                        category_sequences += num_seqs
                    print(f"Total sequences in {category}: {category_sequences}")
                    total_sequences += category_sequences
                print(f"\nTotal sequences across all categories: {total_sequences}")
                
                sequence_count = 0
                
                for category in f.keys():
                    category_group = f[category]
                    category_metrics[category] = {'mse': [], 'ssim': []}
                    
                    for video_name in category_group.keys():
                        video_data = category_group[video_name]
                        
                        inputs = torch.FloatTensor(video_data['inputs'][:]).to(self.device)
                        targets = torch.FloatTensor(video_data['outputs'][:]).to(self.device)
                        
                        for seq_idx in range(inputs.size(0)):
                            try:
                                input_seq = inputs[seq_idx:seq_idx+1]
                                target_seq = targets[seq_idx:seq_idx+1]
                                
                                if input_seq.size(2) != 1:
                                    input_seq = input_seq.permute(0, 1, 4, 2, 3)
                                    target_seq = target_seq.permute(0, 1, 4, 2, 3)
                                
                                predictions = self.model(input_seq, self.config['output_frames'])
                                
                                preds_np = predictions.squeeze().cpu().numpy()
                                targets_np = target_seq.squeeze().cpu().numpy()
                                
                                metrics = self.calculate_metrics(preds_np, targets_np)
                                #print("metrics")
                                
                                # Save sequence metrics
                                with open(sequence_metrics_path, 'a') as seq_file:
                                    seq_file.write(f'{category},{video_name},seq{seq_idx},{metrics["mse"]},{metrics["ssim"]}\n')
                                
                                # Store metrics for overall calculation
                                all_mse.append(metrics["mse"])
                                all_ssim.append(metrics["ssim"])
                                category_metrics[category]['mse'].append(metrics["mse"])
                                category_metrics[category]['ssim'].append(metrics["ssim"])
                                
                                # Create comparison GIF
                                gif_path = self.comparison_dir / f"{self.model.name}_{category}_{video_name}_seq{seq_idx}.gif"
                                self.create_comparison_gif(preds_np, targets_np, str(gif_path))
                                
                                sequence_count += 1
                                if num_sequences and sequence_count >= num_sequences:
                                    break
                                    
                            except Exception as e:
                                self.logger.error(f"Error processing sequence {category}_{video_name}_seq{seq_idx}: {str(e)}")
                                continue
                        
                        if num_sequences and sequence_count >= num_sequences:
                            break
                    
                    if num_sequences and sequence_count >= num_sequences:
                        break
                
                # Calculate and save overall metrics
                with open(overall_metrics_path, 'w') as overall_file:
                    overall_file.write('metric_type,category,mse,ssim\n')
                    
                    # Overall model performance
                    overall_mse = np.mean(all_mse)
                    overall_ssim = np.mean(all_ssim)
                    overall_file.write(f'overall,all,{overall_mse},{overall_ssim}\n')
                    
                    # Per-category performance
                    for category, metrics in category_metrics.items():
                        cat_mse = np.mean(metrics['mse'])
                        cat_ssim = np.mean(metrics['ssim'])
                        overall_file.write(f'category,{category},{cat_mse},{cat_ssim}\n')
                
                self.logger.info(f"Testing completed. Processed {sequence_count} sequences.")
                self.logger.info(f"Overall MSE: {overall_mse:.6f}")
                self.logger.info(f"Overall SSIM: {overall_ssim:.6f}")
                self.logger.info(f"Results saved to:\n{sequence_metrics_path}\n{overall_metrics_path}")
        
        except Exception as e:
            self.logger.error(f"Error during testing: {str(e)}")
            raise

    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calculate MSE and SSIM metrics between predictions and targets.
        
        Args:
            predictions: Predicted frames
            targets: Ground truth frames
            
        Returns:
            Dictionary containing MSE and SSIM metrics
        """
        # Ensure predictions and targets are in the correct range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        targets = np.clip(targets, 0, 1)
        
        # Calculate MSE
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        
        # Calculate SSIM for each frame pair and take the mean
        ssim_scores = []
        for pred, target in zip(predictions, targets):
            score = ssim(target, pred, data_range=1.0)
            ssim_scores.append(score)
        
        avg_ssim = np.mean(ssim_scores)
        
        return {
            'mse': float(mse),
            'ssim': float(avg_ssim)
        }

def main():
    # Define paths relative to project root
    model_path = project_root / "models/models/PredRNN_best.pth"
    test_data_path = project_root / "processed_data/test_sequences.h5"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    tester = PredRNNTester(str(model_path))
    tester.test(str(test_data_path))

if __name__ == "__main__":
    main() 