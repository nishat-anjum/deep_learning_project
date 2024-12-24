import os

MP3_AUDIO_DIRECTORY = os.path.join(os.path.dirname(os.getcwd()), "mp3_audios")
META_DATA_FILE_PATH = os.path.join(os.getcwd(), "dataset.csv")
META_DATA_FILE_PATH_NON_TOXIC = os.path.join(os.getcwd(), "dataset_non_toxic.csv")
WAV_FILE_DIRECTORY = os.path.join(os.path.dirname(os.getcwd()), "wav_files")
AUDIO_SPLIT_DURATION = 15 * 1000
