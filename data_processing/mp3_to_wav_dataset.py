import os
import uuid
import csv
from pydub import AudioSegment
from logger_config.logger import configure_log

logger = configure_log(__name__)


def resample_audio_pydub(mp3_path, data_set_label, wav_file_directory, target_sr=16000):
    try:
        audio = AudioSegment.from_file(mp3_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        file_id = uuid.uuid4().hex
        output_filename = f"{data_set_label}_{file_id}.wav"
        output_path = os.path.join(wav_file_directory, output_filename)
        audio.export(output_path, format="wav")
        return {"file_name": {output_filename}, "file_id": {file_id}}
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")


def create_dataset(data_set_label, mp3_audio_directory, wav_file_directory, metadata_file_path):
    dataset = []
    if not os.path.exists(mp3_audio_directory):
        raise FileNotFoundError(f"The '{mp3_audio_directory}' was not found.")

    if not os.path.exists(wav_file_directory):
        os.makedirs(wav_file_directory)

    for filename in os.listdir(mp3_audio_directory):
        if filename.lower().endswith(".mp3") and not filename.lower().startswith("k"):
            logger.debug(f'convert-{filename} to wav')
            mp3_path = os.path.join(mp3_audio_directory, filename)
            file_metadata = resample_audio_pydub(mp3_path, data_set_label, wav_file_directory)
            dataset.append({"audio_file": file_metadata['file_name'],
                            "file_id": file_metadata['file_id'],
                            "label": data_set_label})

    with open(metadata_file_path, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['audio_file', 'file_id', 'label']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print("Processing complete!")
