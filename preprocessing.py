import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'selected_categories': [
        'Basketball', 
        'CricketShot', 
        'TennisSwing', 
        'PlayingDhol', 
        'PlayingCello'
    ],
    'input_frame_size': (64, 64),
    'input_sequence_length': 10,
    'prediction_sequence_length': 5,
    'color_mode': 'rgb',  # 'rgb' or 'grayscale'
    'data_dir': './Sample',
    'processed_data_dir': './processed_data'
}

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        CONFIG['processed_data_dir'],
        os.path.join(CONFIG['processed_data_dir'], 'train'),
        os.path.join(CONFIG['processed_data_dir'], 'val'),
        os.path.join(CONFIG['processed_data_dir'], 'test'),
        './models',
        './results',
        './logs'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def preprocess_video(video_path, target_size):
    """
    Preprocess a single video file
    Returns: List of preprocessed frames
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Convert to grayscale if specified
        if CONFIG['color_mode'] == 'grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    
    cap.release()
    return frames

def create_sequences(frames, input_length, pred_length):
    """Create input-output sequences from frames"""
    sequences = []
    for i in range(len(frames) - input_length - pred_length + 1):
        input_seq = frames[i:i + input_length]
        output_seq = frames[i + input_length:i + input_length + pred_length]
        sequences.append((np.array(input_seq), np.array(output_seq)))
    return sequences

def process_dataset():
    """Process all videos in the dataset"""
    batch_size = 100  # Adjust this based on your available memory
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(CONFIG['data_dir'], split)
        sequence_count = 0
        batch_count = 0
        current_batch_inputs = []
        current_batch_outputs = []
        current_batch_videos = set()  # Track videos in current batch
        
        for category in CONFIG['selected_categories']:
            category_dir = os.path.join(split_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            print(f"Processing {split} - {category}")
            
            for video_file in os.listdir(category_dir):
                if video_file.endswith('.avi'):
                    video_path = os.path.join(category_dir, video_file)
                    frames = preprocess_video(video_path, CONFIG['input_frame_size'])
                    sequences = create_sequences(
                        frames, 
                        CONFIG['input_sequence_length'],
                        CONFIG['prediction_sequence_length']
                    )
                    
                    for input_seq, output_seq in sequences:
                        current_batch_inputs.append(input_seq)
                        current_batch_outputs.append(output_seq)
                        sequence_count += 1
                    current_batch_videos.add(video_file)
                    
                    if len(current_batch_inputs) >= batch_size:
                        save_path = os.path.join(CONFIG['processed_data_dir'], split)
                        np.save(
                            f"{save_path}/sequences_batch_{batch_count}_inputs.npy", 
                            np.array(current_batch_inputs)
                        )
                        np.save(
                            f"{save_path}/sequences_batch_{batch_count}_outputs.npy", 
                            np.array(current_batch_outputs)
                        )
                        print(f"Batch {batch_count}: {len(current_batch_inputs)} sequences from {len(current_batch_videos)} videos")
                        
                        current_batch_inputs = []
                        current_batch_outputs = []
                        current_batch_videos = set()
                        batch_count += 1
        
        # Save any remaining sequences in the last batch
        if current_batch_inputs:
            save_path = os.path.join(CONFIG['processed_data_dir'], split)
            np.save(
                f"{save_path}/sequences_batch_{batch_count}_inputs.npy", 
                np.array(current_batch_inputs)
            )
            np.save(
                f"{save_path}/sequences_batch_{batch_count}_outputs.npy", 
                np.array(current_batch_outputs)
            )
            print(f"Final batch {batch_count}: {len(current_batch_inputs)} sequences from {len(current_batch_videos)} videos")
        
        print(f"Total sequences for {split}: {sequence_count}")

def analyze_processed_data():
    """Analyze the processed dataset and generate statistics"""
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(CONFIG['processed_data_dir'], split)
        total_sequences = 0
        input_shape = None
        output_shape = None
        
        batch_files = [f for f in os.listdir(split_dir) if f.endswith('_inputs.npy')]
        
        for batch_file in batch_files:
            batch_num = batch_file.split('_')[2]  # Get batch number
            inputs = np.load(os.path.join(split_dir, f"sequences_batch_{batch_num}_inputs.npy"))
            outputs = np.load(os.path.join(split_dir, f"sequences_batch_{batch_num}_outputs.npy"))
            
            total_sequences += len(inputs)
            
            if input_shape is None:
                input_shape = inputs.shape[1:]  # Remove batch dimension
                output_shape = outputs.shape[1:]  # Remove batch dimension
        
        stats[split] = {
            'num_sequences': total_sequences,
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    
    # Print statistics
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        print(f"Number of sequences: {split_stats['num_sequences']}")
        print(f"Input sequence shape: {split_stats['input_shape']}")
        print(f"Output sequence shape: {split_stats['output_shape']}")

if __name__ == "__main__":
    print("Setting up project structure...")
    create_directory_structure()
    
    print("\nProcessing dataset...")
    process_dataset()
    
    print("\nAnalyzing processed data...")
    analyze_processed_data()