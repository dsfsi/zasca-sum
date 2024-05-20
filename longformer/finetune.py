import random
import evaluate
import pandas as pd
from datasets import ClassLabel, Dataset, features, load_metric
# from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

def process_data_to_model_inputs(batch):
  inputs = tokenizer(
    batch["main"],
    padding="max_length",
    truncation=True,
    max_length=max_input_length,
  )
  outputs = tokenizer(
    batch["media-summary"],
    padding="max_length",
    truncation=True,
    max_length=max_output_length,
  )

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask

  batch["global_attention_mask"] = len(batch["input_ids"]) * [
    [0 for _ in range(len(batch["input_ids"][0]))]
  ]

  batch["global_attention_mask"][0][0] = 1
  batch["labels"] = outputs.input_ids

  batch["labels"] = [
    [-100 if token == tokenizer.pad_token_id else token for token in labels]
    for labels in batch["labels"]
  ]

  return batch

def compute_metrics(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions

  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.pad_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  rouge_output = rouge.compute(
    predictions=pred_str, references=label_str, rouge_types=["rouge2"]
  )["rouge2"].mid

  return {
    "rouge2_precision": round(rouge_output.precision, 4),
    "rouge2_recall": round(rouge_output.recall, 4),
    "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
  }

train_dataset = Dataset.from_pandas(pd.read_csv('../data/train.tsv', sep='\t'))
val_dataset = Dataset.from_pandas(pd.read_csv('../data/dev.tsv', sep='\t'))

# print(train_dataset, val_dataset)

max_input_length = 16384
max_output_length = 1024
batch_size = 1
model_name = "allenai/led-base-16384"

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = train_dataset.map(
  process_data_to_model_inputs,
  batched=True,
  batch_size=batch_size,
  remove_columns=['type', 'year', 'j_no', 'main', 'media-summary'],
)

val_dataset = val_dataset.map(
  process_data_to_model_inputs,
  batched=True,
  batch_size=batch_size,
  remove_columns=['type', 'year', 'j_no', 'main', 'media-summary'],
)

train_dataset.set_format(
  type="torch",
  columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
  type="torch",
  columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

led = AutoModelForSeq2SeqLM.from_pretrained(model_name, gradient_checkpointing=True)

# if torch.cuda.device_count() > 1:
#   print(f"Using {torch.cuda.device_count()} GPUs")
#   led = nn.DataParallel(led)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# led.to(device)

led.config.num_beams = 2
led.config.max_length = 1024
led.config.min_length = 512
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3

rouge = load_metric('rouge')

training_args = Seq2SeqTrainingArguments(
  predict_with_generate=True,
  evaluation_strategy="steps",
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  fp16=True,
  output_dir="./",
  logging_steps=5,
  eval_steps=30,
  save_steps=30,
  save_total_limit=2,
  gradient_accumulation_steps=4,
  num_train_epochs=1,
  report_to="tensorboard",
)

trainer = Seq2SeqTrainer(
  model=led,
  tokenizer=tokenizer,
  args=training_args,
  compute_metrics=compute_metrics,
  train_dataset=train_dataset,
  eval_dataset=val_dataset,
)

try:
  trainer.train(resume_from_checkpoint = True)
except:
  trainer.train()
trainer.save_model("models/sca-longformer")