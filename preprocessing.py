import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import h5py
from typing import Tuple, List, Dict
import pandas as pd

class VideoPreprocessor:
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        input_frames: int = 10,
        output_frames: int = 5,
        use_rgb: bool = True
    ):
        """
        Initialize the video preprocessor.
        
        Args:
            input_size: Target frame size (height, width)
            input_frames: Number of input frames for prediction
            output_frames: Number of frames to predict
            use_rgb: If True, keep RGB channels; if False, convert to grayscale
        """
        self.input_size = input_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.use_rgb = use_rgb
        self.channels = 3 if use_rgb else 1

    def process_video(self, video_path: str) -> np.ndarray:
        """
        Extract and preprocess frames from a video file.
        
        Returns:
            Preprocessed frames as numpy array
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                frame = cv2.resize(frame, self.input_size)
                
                # Convert to grayscale if specified
                if not self.use_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
                
            cap.release()
            return np.array(frames)
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None

    def create_sequences(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences from frames.
        
        Returns:
            Tuple of (input_sequences, output_sequences)
        """
        total_frames = len(frames)
        sequence_length = self.input_frames + self.output_frames
        
        if total_frames < sequence_length:
            return None, None
            
        input_sequences = []
        output_sequences = []
        
        for i in range(total_frames - sequence_length + 1):
            input_seq = frames[i:i + self.input_frames]
            output_seq = frames[i + self.input_frames:i + sequence_length]
            
            input_sequences.append(input_seq)
            output_sequences.append(output_seq)
            
        return np.array(input_sequences), np.array(output_sequences)

    def process_dataset(
        self,
        data_dir: str,
        split: str,
        save_path: str
    ) -> None:
        """
        Process all videos in the dataset and save sequences to H5 file.
        """
        base_path = Path(data_dir) / split
        if not base_path.exists():
            raise ValueError(f"Directory not found: {base_path}")

        # Get category directories
        categories = [d for d in base_path.iterdir() if d.is_dir()]
        
        with h5py.File(save_path, 'w') as h5f:
            for category in categories:
                print(f"\nProcessing {category.name} videos...")
                video_files = list(category.glob('*.avi'))
                
                # Create category group in H5 file
                category_group = h5f.create_group(category.name)
                
                for video_file in tqdm(video_files):
                    frames = self.process_video(str(video_file))
                    if frames is None or len(frames) < (self.input_frames + self.output_frames):
                        continue
                        
                    input_seq, output_seq = self.create_sequences(frames)
                    if input_seq is None:
                        continue
                        
                    # Store sequences in H5 file
                    video_group = category_group.create_group(video_file.stem)
                    video_group.create_dataset('inputs', data=input_seq)
                    video_group.create_dataset('outputs', data=output_seq)

def main():
    # Initialize preprocessor with new frame counts
    preprocessor = VideoPreprocessor(
        input_size=(64, 64),
        input_frames=10,
        output_frames=5,
        use_rgb=False
    )
    
    # Process each split
    for split in ['train', 'test', 'val']:
        print(f"\nProcessing {split} split...")
        save_path = f'./processed_data/{split}_sequences.h5'
        os.makedirs('./processed_data', exist_ok=True)
        
        preprocessor.process_dataset(
            data_dir='./Sample',
            split=split,
            save_path=save_path
        )

if __name__ == "__main__":
    main()