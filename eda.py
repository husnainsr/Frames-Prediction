import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from pathlib import Path
import numpy as np
import shutil
from scipy import stats

# Create eda directory if it doesn't exist
eda_dir = './eda'
os.makedirs(eda_dir, exist_ok=True)

print("=== Part 1: Data Extraction ===")

train_df = pd.read_csv('./dataset/train.csv')
test_df = pd.read_csv('./dataset/test.csv')
val_df = pd.read_csv('./dataset/val.csv')

all_labels_counts = pd.concat([train_df['label'], test_df['label'], val_df['label']]).value_counts()
print("\nLabel distribution (in descending order):")
pd.set_option('display.max_rows', None)
print(all_labels_counts)

sample_categories = ['Basketball', 'CricketShot', 'TennisSwing', 'PlayingDhol', 'PlayingCello']

sample_dir = './Sample'
sample_train_dir = os.path.join(sample_dir, 'train')
sample_test_dir = os.path.join(sample_dir, 'test')
sample_val_dir = os.path.join(sample_dir, 'val')

for dir_path in [sample_dir, sample_train_dir, sample_test_dir, sample_val_dir]:
    os.makedirs(dir_path, exist_ok=True)

def copy_category_videos(df, category, source_dir, dest_dir):
    category_videos = df[df['label'] == category]['clip_name'].tolist()
    category_dir = os.path.join(dest_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    print(f"\nProcessing {category} videos:")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {category_dir}")
    print(f"Number of videos to copy: {len(category_videos)}")
    
    for video in category_videos:
        source_path = os.path.join(source_dir, category, video + '.avi')
        dest_path = os.path.join(category_dir, video + '.avi')
        print(f"Attempting to copy: {source_path}")
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Successfully copied: {video}.avi")
        else:
            print(f"File not found: {source_path}")

for category in sample_categories:
    print(f"\nProcessing category: {category}")
    copy_category_videos(train_df, category, './dataset/train', sample_train_dir)
    copy_category_videos(test_df, category, './dataset/test', sample_test_dir)
    copy_category_videos(val_df, category, './dataset/val', sample_val_dir)

sample_train = train_df[train_df['label'].isin(sample_categories)]
sample_test = test_df[test_df['label'].isin(sample_categories)]
sample_val = val_df[val_df['label'].isin(sample_categories)]

sample_train.to_csv(os.path.join(sample_dir, 'train.csv'), index=False)
sample_test.to_csv(os.path.join(sample_dir, 'test.csv'), index=False)
sample_val.to_csv(os.path.join(sample_dir, 'val.csv'), index=False)

print("\nSample dataset created successfully!")

print("\n=== Part 2: Exploratory Data Analysis ===")

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

plt.figure(figsize=(12, 6))
sns.countplot(data=all_data, x='label', hue='split')
plt.xticks(rotation=45)
plt.title('Distribution of Videos Across Categories and Splits')
plt.tight_layout()
plt.savefig(os.path.join('./eda', 'category_distribution.png'))
plt.close()

def get_video_properties(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    properties = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return properties

print("\n=== Video Properties Analysis ===")
video_properties = []
for split in ['train', 'test', 'val']:
    base_path = Path(f'./Sample/{split}')
    for category in all_data['label'].unique():
        category_path = base_path / category
        if category_path.exists():
            for video_file in list(category_path.glob('*.avi'))[:3]:  # Analyze first 3 videos of each category
                props = get_video_properties(video_file)
                if props:
                    props['category'] = category
                    props['split'] = split
                    video_properties.append(props)

video_props_df = pd.DataFrame(video_properties)

print("\nVideo Properties Summary:")
print("\nDuration (seconds):")
print(video_props_df['duration'].describe())
print("\nFrame Count:")
print(video_props_df['frame_count'].describe())
print("\nResolutions:")
resolutions = video_props_df.apply(lambda x: f"{int(x['width'])}x{int(x['height'])}", axis=1).value_counts()
print(resolutions)

plt.figure(figsize=(15, 5))

plt.subplot(131)
sns.boxplot(data=video_props_df, x='category', y='duration')
plt.xticks(rotation=45)
plt.title('Video Duration Distribution by Category')

plt.subplot(132)
sns.boxplot(data=video_props_df, x='category', y='frame_count')
plt.xticks(rotation=45)
plt.title('Frame Count Distribution by Category')

plt.subplot(133)
sns.boxplot(data=video_props_df, x='category', y='fps')
plt.xticks(rotation=45)
plt.title('FPS Distribution by Category')

plt.tight_layout()
plt.savefig(os.path.join('./eda', 'video_properties.png'))
plt.close()

print("\n=== Additional EDA Analysis ===")

# 1. Temporal Analysis
plt.figure(figsize=(15, 5))
temporal_stats = video_props_df.groupby('category')[['duration', 'frame_count', 'fps']].agg(['mean', 'std'])
print("\nTemporal Statistics by Category:")
print(temporal_stats)

# Visualize temporal patterns
plt.subplot(131)
temporal_stats['duration']['mean'].plot(kind='bar')
plt.title('Average Duration by Category')
plt.xticks(rotation=45)
plt.ylabel('Seconds')

plt.subplot(132)
temporal_stats['frame_count']['mean'].plot(kind='bar')
plt.title('Average Frame Count by Category')
plt.xticks(rotation=45)

plt.subplot(133)
temporal_stats['fps']['mean'].plot(kind='bar')
plt.title('Average FPS by Category')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join('./eda', 'temporal_analysis.png'))
plt.close()

# 2. Resolution Analysis
video_props_df['aspect_ratio'] = video_props_df['width'] / video_props_df['height']
video_props_df['resolution'] = video_props_df['width'] * video_props_df['height']

plt.figure(figsize=(15, 5))

plt.subplot(131)
sns.boxplot(data=video_props_df, x='category', y='aspect_ratio')
plt.title('Aspect Ratio Distribution')
plt.xticks(rotation=45)

plt.subplot(132)
sns.boxplot(data=video_props_df, x='category', y='resolution')
plt.title('Resolution Distribution')
plt.xticks(rotation=45)

plt.subplot(133)
resolution_counts = video_props_df.groupby(['width', 'height']).size().reset_index(name='count')
plt.scatter(resolution_counts['width'], resolution_counts['height'], s=resolution_counts['count']*100)
plt.title('Common Resolutions')
plt.xlabel('Width')
plt.ylabel('Height')

plt.tight_layout()
plt.savefig(os.path.join('./eda', 'resolution_analysis.png'))
plt.close()

# 3. Correlation Analysis
correlation_features = ['duration', 'frame_count', 'fps', 'width', 'height', 'aspect_ratio', 'resolution']
correlation_matrix = video_props_df[correlation_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Video Properties')
plt.tight_layout()
plt.savefig(os.path.join('./eda', 'correlation_matrix.png'))
plt.close()

# 4. Statistical Tests

# Test for normal distribution of durations
print("\nNormality Test for Video Durations:")
for category in video_props_df['category'].unique():
    stat, p_value = stats.normaltest(video_props_df[video_props_df['category'] == category]['duration'])
    print(f"{category}: p-value = {p_value:.4f}")

# ANOVA test for duration differences between categories
f_stat, p_value = stats.f_oneway(*[group['duration'].values for name, group in video_props_df.groupby('category')])
print(f"\nANOVA test for duration differences between categories:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Update the summary file with new analyses
with open(os.path.join('./eda', 'eda_summary.txt'), 'a') as f:
    f.write("\n\n=== Additional Analyses ===\n")
    f.write("\nTemporal Statistics by Category:\n")
    f.write(str(temporal_stats))
    f.write("\n\nResolution Statistics:\n")
    f.write("Aspect Ratio Summary:\n")
    f.write(str(video_props_df['aspect_ratio'].describe()))
    f.write("\n\nTotal Resolution Summary:\n")
    f.write(str(video_props_df['resolution'].describe()))
    f.write("\n\nNormality Test Results:\n")
    for category in video_props_df['category'].unique():
        stat, p_value = stats.normaltest(video_props_df[video_props_df['category'] == category]['duration'])
        f.write(f"{category}: p-value = {p_value:.4f}\n")
    f.write(f"\nANOVA test results:\n")
    f.write(f"F-statistic: {f_stat:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n")

print("\n=== EDA Complete ===")
print("Generated files in 'eda' directory:")
print("1. category_distribution.png - Shows distribution of videos across categories")
print("2. video_properties.png - Shows video duration, frame count, and FPS distributions")
print("3. temporal_analysis.png - Shows temporal patterns across categories")
print("4. resolution_analysis.png - Shows resolution patterns and distributions")
print("5. correlation_matrix.png - Shows correlations between video properties")
print("6. eda_summary.txt - Contains detailed statistics and test results")