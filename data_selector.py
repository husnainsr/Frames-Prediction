import pandas as pd
import shutil
import os
import cv2
from tqdm import tqdm

def check_video_consistency(video_path, expected_shape=(320, 240)):
    """
    Check if a video has consistent frame sizes and is readable
    Returns: bool, str (is_consistent, error_message)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video"
            
        # Read first frame to get initial shape
        ret, frame = cap.read()
        if not ret:
            return False, "Could not read first frame"
            
        initial_shape = frame.shape[:2]
        frame_count = 1
        
        # Check subsequent frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame.shape[:2] != initial_shape:
                return False, f"Inconsistent frame shape: {frame.shape[:2]} vs {initial_shape}"
                
            frame_count += 1
            
        cap.release()
        return True, f"Consistent video with {frame_count} frames"
        
    except Exception as e:
        return False, str(e)
    finally:
        cap.release()

def validate_category(category, source_dir):
    """
    Check all videos in a category for consistency
    Returns: bool (is_category_consistent)
    """
    video_dir = os.path.join(source_dir, category)
    if not os.path.exists(video_dir):
        return False
        
    videos = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
    consistent_count = 0
    
    for video in videos:
        is_consistent, _ = check_video_consistency(os.path.join(video_dir, video))
        if is_consistent:
            consistent_count += 1
            
    return (consistent_count / len(videos)) >= 0.95  # 95% videos should be consistent

# Load datasets
train_df = pd.read_csv('./dataset/train.csv')
test_df = pd.read_csv('./dataset/test.csv')
val_df = pd.read_csv('./dataset/val.csv')

# Categories with typically consistent formats
potential_categories = [
    'PushUps',          # Fixed camera, indoor setting
    'JumpingJack',      # Fixed camera, consistent motion
    'Lunges',           # Fixed camera, controlled environment
    'BodyWeightSquats', # Fixed camera, controlled motion
    'PullUps',          # Fixed camera, fixed equipment
    'WallPushups',      # Fixed camera, consistent background
    'HandstandPushups', # Fixed camera, controlled environment
    'BoxingPunchingBag' # Fixed camera, consistent setup
]

print("Validating categories for consistency...")
consistent_categories = []

for category in tqdm(potential_categories):
    if validate_category(category, './dataset/train'):
        consistent_categories.append(category)
        print(f"{category}: Passed validation")
    else:
        print(f"{category}: Failed validation")

# Select top 5 most consistent categories
sample_categories = consistent_categories[:5]
print(f"\nSelected categories: {sample_categories}")

# Create Sample directory structure
sample_dir = './Sample'
sample_train_dir = os.path.join(sample_dir, 'train')
sample_test_dir = os.path.join(sample_dir, 'test')
sample_val_dir = os.path.join(sample_dir, 'val')

for dir_path in [sample_dir, sample_train_dir, sample_test_dir, sample_val_dir]:
    os.makedirs(dir_path, exist_ok=True)

def copy_category_videos(df, category, source_dir, dest_dir, max_videos=None):
    category_videos = df[df['label'] == category]['clip_name'].tolist()
    category_dir = os.path.join(dest_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    copied_count = 0
    for video in category_videos:
        if max_videos and copied_count >= max_videos:
            break
            
        source_path = os.path.join(source_dir, category, video + '.avi')
        if os.path.exists(source_path):
            # Validate video before copying
            is_consistent, _ = check_video_consistency(source_path)
            if is_consistent:
                dest_path = os.path.join(category_dir, video + '.avi')
                shutil.copy2(source_path, dest_path)
                copied_count += 1
    
    print(f"Copied {copied_count} consistent videos for {category}")

# Copy validated videos
for category in sample_categories:
    print(f"\nProcessing category: {category}")
    copy_category_videos(train_df, category, './dataset/train', sample_train_dir)
    copy_category_videos(test_df, category, './dataset/test', sample_test_dir)
    copy_category_videos(val_df, category, './dataset/val', sample_val_dir)

# Save filtered CSV files
sample_train = train_df[train_df['label'].isin(sample_categories)]
sample_test = test_df[test_df['label'].isin(sample_categories)]
sample_val = val_df[val_df['label'].isin(sample_categories)]

sample_train.to_csv(os.path.join(sample_dir, 'train.csv'), index=False)
sample_test.to_csv(os.path.join(sample_dir, 'test.csv'), index=False)
sample_val.to_csv(os.path.join(sample_dir, 'val.csv'), index=False)

print("\nSample dataset created successfully!")
print("Selected categories with consistent video formats:")
for category in sample_categories:
    count = len(os.listdir(os.path.join(sample_train_dir, category)))
    print(f"{category}: {count} videos")
