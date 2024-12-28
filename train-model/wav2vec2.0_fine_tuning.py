from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
from datasets import load_dataset, Audio


def load_data(csv_path):
    dataset = load_dataset("csv", data_files={"train": csv_path})
    return dataset


def preprocess(batch):
    audio = batch["file"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = batch["label"]
    return batch


dataset_path = "/home/nishat/DP_Dataset_and_project/dataset.csv"  # Path to your CSV file
dataset = load_data(dataset_path)

dataset = dataset.cast_column("file", Audio(sampling_rate=16000))
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
dataset = dataset.map(preprocess, remove_columns=["file"])

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=processor,
)

trainer.train()

model.save_pretrained("./wav2vec2-toxic-bangla")
processor.save_pretrained("./wav2vec2-toxic-bangla")
