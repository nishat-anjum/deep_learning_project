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
# Load metrics globally
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
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

def compute_metrics(eval_pred):
    # Extract predictions and references
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids

    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)

    # Compute F1-score
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")

    # Return both metrics
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def main():


    # Load the dataset
    file_path = 'E:\DeepLearning\wav2vecPy3.8\src\dataset.csv'  # Path to your CSV
    df = pd.read_csv(file_path)
    # Inspect the dataset
    print(df.head())

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.2)

    dataset = dataset.map(preprocess_data)

    # Load the model
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(set(df["label"])),  # Number of unique labels
    )

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__=="__main__":
    main()