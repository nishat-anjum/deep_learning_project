import os
from logger import configure_log
from pytube import YouTube
from config import DOWNLOAD_DIRECTORY

logger = configure_log()

def download_audio_from_youtube(url):
    try:
        yt = YouTube(url)
        logger.debug(f"downloading: {yt.title}")
        audio_stream = yt.streams.filter(only_audio=True).first()
        if not os.path.exists(DOWNLOAD_DIRECTORY):
            os.mkdir(DOWNLOAD_DIRECTORY)
        output_file = audio_stream.download(output_path=DOWNLOAD_DIRECTORY)
        base, ext = os.path.splitext(output_file)
        new_file = base + ".mp3"
        os.rename(output_file, new_file)

    except Exception as e:
        logger.error("Error downloading audio")
