import os

MP3_AUDIO_DIRECTORY = os.path.join(os.path.dirname(os.getcwd()), "DataSet/splitter")
print(MP3_AUDIO_DIRECTORY)
META_DATA_FILE_PATH = os.path.join(os.getcwd(), "dataset.csv")
META_DATA_FILE_PATH_NON_TOXIC = os.path.join(os.getcwd(), "dataset_non_toxic.csv")
WAV_FILE_DIRECTORY = os.path.join(os.path.dirname(os.getcwd()), "DataSet/splitted_wav")
AUDIO_SPLIT_DURATION = 12 * 1000
