from datasets import Dataset, load_dataset, Audio
from transformers import AutoFeatureExtractor
import evaluate
import numpy as np
from pynvml import *
import torchaudio
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import pandas as pd
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch

processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
def preprocess_data(batch):
    # Load the processor (e.g., for Wav2Vec2)

    waveform, sample_rate = torchaudio.load("E:\\DeepLearning\\wav2vecPy3.8\\bangla-toxic-data-set\\" + batch["file"])  # Load the audio file
    inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt", truncation=True, padding="max_length", max_length=41600)
    batch["input_values"] = inputs.input_values[0]
    batch["label"] = 0 if batch["label"] == 'non-toxic' else 1
    return batch

# Prepare DataLoader
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    return {"input_values": torch.tensor(input_values), "labels": torch.tensor(labels)}

def evaluate(model, dataloader, device):
    model.eval()
    accuracy = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy += (predictions == labels).sum().item()
            total += labels.size(0)
    return accuracy / total

def main():


    # Load the dataset
    file_path = 'E:\DeepLearning\wav2vecPy3.8\src\dataset.csv'  # Path to your CSV
    df = pd.read_csv(file_path)
    # Inspect the dataset
    print(df.head())

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2)
    test_valid_split = dataset["test"].train_test_split(test_size=0.5)
    dataset["validation"] = test_valid_split["train"]
    dataset["test"] = test_valid_split["test"]

    dataset = dataset.map(preprocess_data)

    # Load the model
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(set(df["label"])),  # Number of unique labels
    )

    train_dataloader = DataLoader(dataset["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(dataset["validation"], batch_size=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset["test"], batch_size=8, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir="toxic",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,

    )

    for epoch in range(10):
        model.train()
        for batch in train_dataloader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        val_accuracy, val_loss = evaluate(model, valid_dataloader, device)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Test the model
    test_accuracy, test_loss = evaluate(model, test_dataloader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__=="__main__":
    main()