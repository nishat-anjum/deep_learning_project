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

processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
# Load metrics globally
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
dataSetDir = "E:\\DeepLearning\\wav2vecPy3.8\\src\\EQUAL_TOXIC_NON_TOXIC_WAV\\"
file_path = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\equal_label_dataset.csv"
#dataSetDir = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\LARGER_NON-TOXIC_AND_LESS_TOXIC\\"
#file_path = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\larger_non-toxic_and_less_toxic.csv"
testDir = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\UNIQUE_TOXIC_NON_TOXIC_TEST_DATA\\"
test_file = "E:\\DeepLearning\\wav2vecPy3.8\\final_dataset\\unique_test_toxic_and_non_toxic.csv"

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
    print("f1:" + str(f1["f1"]))
    # Return both metrics
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

def split_over_samples(data):
    class_0 = data.filter(lambda example: example["label"] == 'non-toxic')
    class_1 = data.filter(lambda example: example["label"] == 'toxic')

    while len(class_1) < len(class_0):
        class_1 = concatenate_datasets([class_1, class_1])

    # Oversample the minority class
    oversampled_class_1 = class_1.shuffle(seed=42).select(range(len(class_0)))  # Match the size of class_0

    # Combine the datasets
    balanced_dataset = concatenate_datasets([class_0, oversampled_class_1])

    # Perform the split
    dataset_split = balanced_dataset.train_test_split(test_size=0.2, seed=42)
    return dataset_split

def split_less_samples(data):
    class_0 = data.filter(lambda example: example["label"] == 'non-toxic')
    class_1 = data.filter(lambda example: example["label"] == 'toxic')

    # Lesssample the majority class
    oversampled_class_0 = class_0.shuffle(seed=42).select(range(len(class_1)))  # Match the size of class_0

    # Combine the datasets
    balanced_dataset = concatenate_datasets([oversampled_class_0, class_1])

    # Perform the split
    dataset_split = balanced_dataset.train_test_split(test_size=0.2, seed=42)
    return dataset_split

def split(data):
    return data.train_test_split(test_size=0.2)

def main():
    # Load the dataset

    df = pd.read_csv(file_path)
    df_test = pd.read_csv(test_file)
    # Inspect the dataset
    print(df.head())
    df = pd.concat([df, df_test.head(30)])
    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split into train and test sets
    dataset = split(dataset)
    dataset_test = Dataset.from_pandas(df_test.tail(98))

    dataset = dataset.map(lambda batch: preprocess_data(batch, dataSetDir))
    dataset_test = dataset_test.map(lambda batch: preprocess_data(batch, testDir))

    # Load the model
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(set(df["label"])),  # Number of unique labels
    )
    # for param in model.base_model.parameters():
    #     param.requires_grad = False

    # Calculate the total number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {trainable_params}")

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
        eval_dataset=dataset_test, #Validation set
        tokenizer=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Evaluate the model on a separate test set
    print("Evaluating on a separate test set...")
    test_results = trainer.evaluate(eval_dataset=dataset['test'])
    print(test_results)

    print("Classifier Head Parameters:")
    for name, param in model.classifier.named_parameters():
        print(f"Parameter Name: {name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")

if __name__=="__main__":
    main()