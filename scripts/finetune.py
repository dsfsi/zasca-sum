import os
import sys
import argparse
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  AutoModelForSeq2SeqLM,
  Seq2SeqTrainingArguments,
  Seq2SeqTrainer
)

def preprocess_function(examples):
  inputs = [prefix + doc for doc in examples["main"]]
  model_inputs = tokenizer(
                  inputs,
                  max_length=input_max_length,
                  truncation=True
                )

  labels = tokenizer(
            text_target=examples["media-summary"],
            max_length=label_max_length,
            truncation=True
          )

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

def postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [[label.strip()] for label in labels]

  return preds, labels

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

  result = eval_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)

  return {k: round(v, 4) for k, v in result.items()}

parser = argparse.ArgumentParser(description='Fine-tune a T5 model on the SCA Judgment and Summaries dataset')

parser.add_argument('--data_dir', type=str, default='data', help='Path to the directory containing the train, test and dev splits of the SCA Judgment and Summaries dataset')
parser.add_argument('--output_dir', type=str, default='sca_judgment_summaries', help='Path to the directory to save the fine-tuned model')

parser.add_argument('--model_name', type=str, default='t5-small', help='Name of the T5 model to use')
parser.add_argument('--eval_metric', type=str, default='rouge', help='Name of the metric to use for evaluation')
parser.add_argument('--input_max_length', type=int, default=11600, help='Maximum length of the input')
parser.add_argument('--label_max_length', type=int, default=1200, help='Maximum length of the label')

parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the fine-tuning')
parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Batch size for evaluation')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer')
parser.add_argument('--num_train_epochs', type=int, default=4, help='Number of epochs for training')
parser.add_argument('--save_total_limit', type=int, default=3, help='Limit the total amount of checkpoints')
parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Strategy for evaluation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')

parser.add_argument('--do_train', action='store_true', help='Train the model')
parser.add_argument('--do_eval', action='store_true', help='Evaluate the model')
parser.add_argument('--do_predict', action='store_true', help='Generate predictions using the model')

parser.add_argument('--overwrite_output_dir', action='store_true', help='Overwrite the output directory')
parser.add_argument('--predict_with_generate', action='store_true', help='Use generation during evaluation and prediction')

args = parser.parse_args()

dataset = DatasetDict()

data_files = [f for f in os.listdir(args.data_dir) if f.split('.')[0] in ['train', 'dev', 'test']]
if len(data_files) == 0:
  sys.exit('No train, dev and test splits found in the data directory')
if args.do_train:
  if 'train.tsv' not in data_files:
    sys.exit('No train split found in the data directory')
  dataset['train'] = Dataset.from_pandas(pd.read_csv(f'{args.data_dir}/train.tsv', sep='\t'))
if args.do_eval:
  if 'dev.tsv' not in data_files:
    sys.exit('No dev split found in the data directory')
  dataset['dev'] = Dataset.from_pandas(pd.read_csv(f'{args.data_dir}/dev.tsv', sep='\t'))
if args.do_predict:
  if 'test.tsv' not in data_files:
    sys.exit('No test split found in the data directory')
  dataset['test'] = Dataset.from_pandas(pd.read_csv(f'{args.data_dir}/test.tsv', sep='\t'))

print(dataset)

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

prefix = "summarize: "
input_max_length = args.input_max_length
label_max_length = args.label_max_length

tokenized_scaj = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

eval_metric = evaluate.load(args.eval_metric)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

if args.do_train:
  training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(args.output_dir, '_'.join(args.model_name.split('/'))),
    evaluation_strategy=args.evaluation_strategy,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=args.save_total_limit,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=str(args.predict_with_generate),
    fp16=str(args.fp16)
  )

  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_scaj["train"],
    eval_dataset=tokenized_scaj["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
  )

  train_result = trainer.train()
  trainer.save_model()

  metrics = train_result.metrics
  metrics["train_samples"] = len(tokenized_scaj["train"])

  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()

if args.do_eval:
  metrics = trainer.evaluate(
      metric_key_prefix="eval"
  )
  metrics["eval_samples"] = len(tokenized_scaj["dev"])

  trainer.log_metrics("eval", metrics)
  trainer.save_metrics("eval", metrics)

if args.do_predict:
  test_results = trainer.predict(
      tokenized_scaj["test"],
      metric_key_prefix="test"
  )
  metrics = test_results.metrics
  metrics["predict_samples"] = len(tokenized_scaj["test"])

  trainer.log_metrics("predict", metrics)
  trainer.save_metrics("predict", metrics)

  if trainer.is_world_process_zero():
    if training_args.predict_with_generate:
      predictions = tokenizer.batch_decode(
        test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
      )
      predictions = [pred.strip() for pred in predictions]
      output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
      with open(output_prediction_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(predictions))