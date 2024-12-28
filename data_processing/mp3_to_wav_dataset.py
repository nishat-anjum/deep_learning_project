import os
import uuid
import csv
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from logger_config.logger import configure_log

logger = configure_log(__name__)


def resample_audio_pydub(filename, audio_file_path, data_set_label, wav_file_directory, target_sr=16000,
                         silence_threshold=-40, min_silence_length=800):
    try:
        audio = AudioSegment.from_file(audio_file_path)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_length,
            silence_thresh=silence_threshold,
        )

        if not nonsilent_ranges:
            print(f"Ignoring {filename}: Contains only silence.")
            return

        audio = audio.set_frame_rate(target_sr).set_channels(1)
        file_id = uuid.uuid4().hex
        output_filename = f"{file_id}_{data_set_label}.wav"
        output_path = os.path.join(wav_file_directory, output_filename)
        audio.export(output_path, format="wav")

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")


def create_dataset(toxic_mp3_audio_directory, non_toxic_mp3_audio_directory, wav_file_directory):

    if not os.path.exists(wav_file_directory):
        os.makedirs(wav_file_directory)

    if toxic_mp3_audio_directory and os.path.exists(toxic_mp3_audio_directory):
        for root, _, files in os.walk(toxic_mp3_audio_directory):
            for filename in files:
                if not filename.lower().startswith("k"):
                    logger.debug(f'convert Toxic -{filename} to wav')
                    mp3_path = os.path.join(toxic_mp3_audio_directory, filename)
                    resample_audio_pydub(filename, mp3_path, "toxic", wav_file_directory)

    if non_toxic_mp3_audio_directory and os.path.exists(non_toxic_mp3_audio_directory):
        for root, _, files in os.walk(non_toxic_mp3_audio_directory):
            for filename in files:
                logger.debug(f'convert NoToxic-{filename} to wav')
                mp3_path = os.path.abspath(os.path.join(root, filename))
                logger.debug(mp3_path)
                resample_audio_pydub(filename, mp3_path, "non-toxic", wav_file_directory)

    print("Wav Conversion completed!")


def generate_csv(wav_file_directory, metadata_file_path):
    logger.debug("Start generating CSV for data Set")
    dataset = []

    for filename in os.listdir(wav_file_directory):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(wav_file_directory, filename)
            audio = AudioSegment.from_wav(file_path)
            samples = np.array(audio.get_array_of_samples())
            labelString = filename.split('_')[1].split('.')[0]
            label = 0 if "non-toxic" in filename.lower() else 1
            dataset.append({"file": filename,
                            "label": labelString})

    with open(metadata_file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['file', 'label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print("Csv Generation complete!")
