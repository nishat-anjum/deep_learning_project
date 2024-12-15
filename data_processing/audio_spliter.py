import os
from pydub import AudioSegment
import uuid
from config import DOWNLOAD_DIRECTORY, WAV_FILE_DIRECTORY, AUDIO_SPLIT_DURATION

def split_audio():

    if not os.path.exists(DOWNLOAD_DIRECTORY):
        raise FileNotFoundError(f"The  '{DOWNLOAD_DIRECTORY}' was not found.")

    if not os.path.exists(WAV_FILE_DIRECTORY):
        os.makedirs(WAV_FILE_DIRECTORY)

    for filename in os.listdir(DOWNLOAD_DIRECTORY):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(DOWNLOAD_DIRECTORY, filename)
            audio = AudioSegment.from_mp3(mp3_path)
            filtered_audio = audio.set_channels(1).set_frame_rate(16000)
            audio_length = len(filtered_audio)

            #TODO Filter noise


            for start_ms in range(0, audio_length, AUDIO_SPLIT_DURATION):
                end_ms = min(start_ms + AUDIO_SPLIT_DURATION, audio_length)
                chunk = filtered_audio[start_ms:end_ms]
                output_filename = f"toxic_{uuid.uuid4().hex}.wav"
                output_path = os.path.join(WAV_FILE_DIRECTORY, output_filename)

                chunk.export(output_path, format="wav")
                print(f"Exported: {output_path}")
    print("Processing complete!")
