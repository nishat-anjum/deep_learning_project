# This is a sample Python script.
import argparse
from data_processing import audio_spliter
from data_processing import mp3_to_wav_dataset
from logger_config.logger import configure_log
logger = configure_log(__name__)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        logger.debug("Start Processing toxic data")

        # mp3_to_wav_dataset.create_dataset('toxic', MP3_AUDIO_DIRECTORY, WAV_FILE_DIRECTORY, META_DATA_FILE_PATH)
        # mp3_to_wav_dataset.generate_csv('toxic', WAV_FILE_DIRECTORY, META_DATA_FILE_PATH)
        # mp3_to_wav_dataset.create_dataset_non_toxic("non-toxic", os.path.join(os.path.dirname(os.getcwd()), "non-toxic"),
        #                                             os.path.join(os.path.dirname(os.getcwd()), "wav_files_non"))

        # mp3_to_wav_dataset.generate_csv("non-toxic", os.path.join(os.path.dirname(os.getcwd()), "wav_files_non"), META_DATA_FILE_PATH)

        parser = argparse.ArgumentParser(description="Convert audio files to wav and creat dataset")
        parser.add_argument("toxic_audio_directory", type=str, help="Root directory containing toxic MP3 files.")
        parser.add_argument("non_toxic_audio_directory", type=str, help="Root directory containing toxic MP3 files.")
        parser.add_argument("wav_directory", type=str, help="Root directory to store wav files")
        parser.add_argument("csv_file_path", type=str, help="Csv file path for data set")

        args = parser.parse_args()
        mp3_to_wav_dataset.create_dataset(args.toxic_audio_directory, args.non_toxic_audio_directory, args.wav_directory)
        mp3_to_wav_dataset.generate_csv(args.wav_directory, args.csv_file_path)

    except Exception as e:
        logger.error("Unhandled exception: %s", e, exc_info=True)  # Log stack trace
        print("Something went wrong. Check the log for details.")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
