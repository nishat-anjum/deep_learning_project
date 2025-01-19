import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
def main():
    # Load the feature extractor and trained model
    processor = AutoFeatureExtractor.from_pretrained("best-model")
    model = AutoModelForAudioClassification.from_pretrained("best-model")

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def predict_single_file(file_path):
        # Load and preprocess the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=64000)

        # Move inputs to the device
        input_values = inputs.input_values.to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_values)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return predicted_class

    # Path to the wav file
    file_path = "E:\\DeepLearning\\wav2vecPy3.8\\src\\brand_new_2_256kbs.wav"  # Replace with the path to your wav file

    # Predict
    predicted_class = predict_single_file(file_path)
    print(f"Predicted class: { 'toxic' if predicted_class == 1 else 'non-toxic' }")

if __name__=="__main__":
    main()