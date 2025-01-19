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
from datasets import concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support

processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# Load metrics globally
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
dataSetDir = "E:\\DeepLearning\\wav2vecPy3.8\\src\\TRAINING_DATA\\"
file_path = "E:\\DeepLearning\\wav2vecPy3.8\\src\\training_dataset.csv"
#dataSetDir = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\LARGER_NON-TOXIC_AND_LESS_TOXIC\\"
#file_path = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\larger_non-toxic_and_less_toxic.csv"
testDir = "E:\\DeepLearning\\wav2vecPy3.8\\src\\TEST_UNIQ_DATA\\"
test_file = "E:\\DeepLearning\\wav2vecPy3.8\\src\\test_dataset.csv"

def preprocess_data(batch, data_dir):
    # Load the processor (e.g., for Wav2Vec2)

    waveform, sample_rate = torchaudio.load( data_dir + batch["file"])  # Load the audio file
    inputs = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt", truncation=True, padding="max_length", max_length=64000)
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

    precision, recall, _, _ = precision_recall_fscore_support(references, predictions, average="weighted")
    # Print metrics
    print(f"Accuracy: {accuracy['accuracy']:.4f}")
    print(f"F1-Score: {f1['f1']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    # Return both metrics
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"], "precision": precision, "recall": recall}

def split(data):
    return data.train_test_split(test_size=0.2)

def main():
    # Load the dataset

    df = pd.read_csv(file_path)
    df_test = pd.read_csv(test_file)
    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and test sets
    dataset = split(dataset) # split to train and validation

    dataset_test = Dataset.from_pandas(df_test)

    dataset = dataset.map(lambda batch: preprocess_data(batch, dataSetDir))
    dataset_test = dataset_test.map(lambda batch: preprocess_data(batch, testDir))

    # Load the model
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(set(df["label"])),  # Number of unique labels
    )
    #uncomment to freeze base model
    #for param in model.base_model.parameters():
    #    param.requires_grad = False

    # Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {trainable_params}")

    trainable_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    print(f"Total trainable classification parameters: {trainable_params}")

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
        num_train_epochs=5,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'], #Validation set we have separate test set
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Evaluate the model on a separate test set
    print("Evaluating on a separate test set...")
    test_results = trainer.evaluate(eval_dataset=dataset_test)
    print(test_results)

    print("Classifier Head Parameters:")
    for name, param in model.classifier.named_parameters():
        print(f"Parameter Name: {name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")

if __name__=="__main__":
    main()