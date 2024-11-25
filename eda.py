import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from pathlib import Path
import numpy as np
from scipy import stats
from tqdm import tqdm

# Create eda directory if it doesn't exist
eda_dir = './eda'
os.makedirs(eda_dir, exist_ok=True)

print("=== Video Properties Analysis ===")

# Load existing sample datasets
sample_train = pd.read_csv('./Sample/train.csv')
sample_test = pd.read_csv('./Sample/test.csv')
sample_val = pd.read_csv('./Sample/val.csv')

all_data = pd.concat([
    sample_train.assign(split='train'),
    sample_test.assign(split='test'),
    sample_val.assign(split='val')
])

print("\nDataset Overview:")
print(f"Total number of videos: {len(all_data)}")
print("\nDistribution across splits:")
print(all_data['split'].value_counts())
print("\nDistribution across categories:")
print(all_data['label'].value_counts())

def get_video_properties(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if frame_count <= 0 or fps <= 0:
            print(f"Invalid frame count or FPS for {video_path}")
            return None
            
        properties = {
            'frame_count': frame_count,
            'fps': fps,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': frame_count / fps if fps > 0 else 0
        }
        return properties
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None
    finally:
        if 'cap' in locals():
            cap.release()

# Analyze video properties with progress bar
print("\nAnalyzing video properties...")
video_properties = []
for split in ['train', 'test', 'val']:
    base_path = Path(f'./Sample/{split}')
    for category in all_data['label'].unique():
        category_path = base_path / category
        if category_path.exists():
            video_files = list(category_path.glob('*.avi'))
            for video_file in tqdm(video_files, desc=f"{split}/{category}"):
                props = get_video_properties(str(video_file))
                if props:
                    props['category'] = category
                    props['split'] = split
                    props['filename'] = video_file.name
                    video_properties.append(props)

if not video_properties:
    print("No video properties could be extracted!")
    exit(1)

video_props_df = pd.DataFrame(video_properties)

# Generate visualizations and statistics
print("\nVideo Properties Summary:")
if 'duration' in video_props_df.columns:
    print("\nDuration (seconds):")
    print(video_props_df['duration'].describe())
if 'frame_count' in video_props_df.columns:
    print("\nFrame Count:")
    print(video_props_df['frame_count'].describe())
if 'width' in video_props_df.columns and 'height' in video_props_df.columns:
    print("\nResolutions:")
    resolutions = video_props_df.apply(lambda x: f"{int(x['width'])}x{int(x['height'])}", axis=1).value_counts()
    print(resolutions)

# Save detailed analysis to file
with open(os.path.join(eda_dir, 'video_analysis.txt'), 'w') as f:
    f.write("=== Video Analysis Report ===\n\n")
    
    f.write("Category Statistics:\n")
    for category in video_props_df['category'].unique():
        cat_data = video_props_df[video_props_df['category'] == category]
        f.write(f"\n{category}:\n")
        f.write(f"Number of videos: {len(cat_data)}\n")
        if 'duration' in cat_data.columns:
            f.write(f"Average duration: {cat_data['duration'].mean():.2f} seconds\n")
        if 'frame_count' in cat_data.columns:
            f.write(f"Average frame count: {cat_data['frame_count'].mean():.2f}\n")
        if 'width' in cat_data.columns and 'height' in cat_data.columns:
            f.write(f"Common resolutions: {cat_data.apply(lambda x: f'{int(x.width)}x{int(x.height)}', axis=1).value_counts().to_dict()}\n")

# Generate visualizations
plt.figure(figsize=(15, 10))

if 'duration' in video_props_df.columns:
    plt.subplot(2, 2, 1)
    sns.boxplot(data=video_props_df, x='category', y='duration')
    plt.xticks(rotation=45)
    plt.title('Video Durations by Category')

if 'frame_count' in video_props_df.columns:
    plt.subplot(2, 2, 2)
    sns.boxplot(data=video_props_df, x='category', y='frame_count')
    plt.xticks(rotation=45)
    plt.title('Frame Counts by Category')

plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'video_properties.png'))
plt.close()

print("\nAnalysis complete! Check the 'eda' directory for detailed results.")