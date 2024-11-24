from kaggle.api.kaggle_api_extended import KaggleApi
import os

try:
    # Initialize the API
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful!")

    # Create a directory for the dataset
    download_path = './dataset'
    os.makedirs(download_path, exist_ok=True)

    # Using the UCF101 action recognition dataset
    dataset_name = 'matthewjansen/ucf101-action-recognition'
    print(f"Downloading dataset: {dataset_name}")
    
    api.dataset_download_files(
        dataset_name,
        path=download_path,
        unzip=True
    )
    print(f"Dataset downloaded successfully to {download_path}")

except Exception as e:
    print(f"Error occurred: {str(e)}")