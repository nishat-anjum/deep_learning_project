# This is a sample Python script.
import argparse
import logging
from data_processing import wav_converter
def configure_log(module_name):
    logger = logging.getLogger(module_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = configure_log(__name__)


if __name__ == '__main__':
    try:
        logger.debug("Start Processing toxic data")
        parser = argparse.ArgumentParser(description="Convert audio files to wav and creat dataset")
        parser.add_argument("toxic_audio_directory", type=str, help="Root directory containing toxic MP3 files.")
        parser.add_argument("non_toxic_audio_directory", type=str, help="Root directory containing toxic MP3 files.")
        parser.add_argument("wav_directory", type=str, help="Root directory to store wav files")
        parser.add_argument("csv_file_path", type=str, help="Csv file path for data set")

        args = parser.parse_args()
        wav_converter.generate_csv(args.wav_directory, args.csv_file_path)

    except Exception as e:
        logger.error("Unhandled exception: %s", e, exc_info=True)  # Log stack trace
        print("Something went wrong. Check the log for details.")
