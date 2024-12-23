# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data_processing import mp3_to_wav_dataset
from logger_config.logger import configure_log
from config import MP3_AUDIO_DIRECTORY, WAV_FILE_DIRECTORY, META_DATA_FILE_PATH

logger = configure_log(__name__)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        logger.debug("Start Processing toxic data")
        mp3_to_wav_dataset.create_dataset('toxic', MP3_AUDIO_DIRECTORY, WAV_FILE_DIRECTORY, META_DATA_FILE_PATH)
    except Exception as e:
        logger.error("Unhandled exception: %s", e, exc_info=True)  # Log stack trace
        print("Something went wrong. Check the log for details.")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
