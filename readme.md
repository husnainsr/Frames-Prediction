# Predicting and Generating Video Sequences Using Deep Learning


## Run this Project
Make sure you have all the dependencies installed.
or you can use the requirements.txt file to install the dependencies.

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Download the dataset from kaggle and put it in the dataset folder.
or you can download it using the data_downloader.py file.
Note: You need to have kaggle api key to download the dataset. You can get it from [here](https://www.kaggle.com/settings/account) and place it in `~/.kaggle/kaggle.json` file.

```bash
python data_downloader.py
```

Run the data_selector.py file to select the categories that have consistent formats.

```bash
python data_selector.py
```


Run the eda.py file to get the summary of the dataset.

```bash
python eda.py
```

### Categories

The following exercise categories are included in this dataset:

- PushUps
- JumpingJack
- Lunges  
- BodyWeightSquats
- PullUps

Run the preprocessing.py file to preprocess the dataset.
```bash
python preprocessing.py
```

## Collaborators
- Husnain Sattar
- Isma
- Laiba Batool
