import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import h5py
import numpy as np
from models.conv_lstm import ConvLSTMPredictor
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

class ModelTester:
    def __init__(self, model_path: str, device: str = None):
        """Initialize the model tester"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories first (creates necessary paths)
        self.test_dir = Path(__file__).parent / "results" / "conv"
        self.comparison_dir = self.test_dir / "comparisons"
        self.metrics_dir = self.test_dir / "metrics"
        
        # Create all directories
        for directory in [self.test_dir, self.comparison_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model(model_path)
        self.model.eval()

    def load_model(self, model_path: str):
        """Load the trained model and its configuration"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = ConvLSTMPredictor(
            input_channels=1,
            hidden_channels=128,
            kernel_size=3
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
        # Create metrics file
        metrics_path = self.metrics_dir / 'all_metrics.csv'
        with open(metrics_path, 'w') as metrics_file:
            metrics_file.write('sequence_name,mse,ssim\n')
        
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
                
                # Create progress bar
                pbar = tqdm(total=total_sequences if not num_sequences else min(num_sequences, total_sequences),
                          desc="Processing sequences")
                
                sequence_count = 0
                
                # Iterate through categories
                for category in f.keys():
                    category_group = f[category]
                    
                    # Iterate through videos in the category
                    for video_name in category_group.keys():
                        video_data = category_group[video_name]
                        
                        # Get input and target sequences
                        inputs = torch.FloatTensor(video_data['inputs'][:]).to(self.device)
                        targets = torch.FloatTensor(video_data['outputs'][:]).to(self.device)
                        
                        # Process each sequence
                        for seq_idx in range(inputs.size(0)):
                            try:
                                # Generate predictions
                                input_seq = inputs[seq_idx:seq_idx+1]
                                target_seq = targets[seq_idx:seq_idx+1]
                                input_seq = input_seq.permute(0, 1, 4, 2, 3)
                                predictions = self.model(input_seq, self.config['output_frames'])
                                
                                # Convert predictions back to original format
                                preds_np = predictions.squeeze(0).squeeze(1).cpu().numpy()
                                targets_np = target_seq.squeeze(0).squeeze(-1).cpu().numpy()
                                
                                # Calculate metrics
                                metrics = self.calculate_metrics(preds_np, targets_np)
                                
                                # Save metrics to CSV
                                sequence_name = f"{category}_{video_name}_seq{seq_idx}"
                                with open(metrics_path, 'a') as metrics_file:
                                    metrics_file.write(f'{sequence_name},{metrics["mse"]},{metrics["ssim"]}\n')
                                
                                # Save comparison GIF
                                gif_path = self.comparison_dir / f"{sequence_name}.gif"
                                self.create_comparison_gif(preds_np, targets_np, str(gif_path))
                                
                                sequence_count += 1
                                pbar.update(1)
                                
                                if num_sequences and sequence_count >= num_sequences:
                                    break
                                    
                            except Exception as e:
                                continue
                        
                        if num_sequences and sequence_count >= num_sequences:
                            break
                    
                    if num_sequences and sequence_count >= num_sequences:
                        break
                
                pbar.close()
                
        except Exception as e:
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
    model_path = project_root / "models/ConvLSTMPredictor_best.pth"
    test_data_path = project_root / "processed_data/test_sequences.h5"
    
    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    # Initialize tester
    tester = ModelTester(str(model_path))
    
    # Run testing on all sequences
    tester.test(str(test_data_path))

if __name__ == "__main__":
    main()