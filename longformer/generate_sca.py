import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import LEDTokenizer, LEDForConditionalGeneration

model_name = "allenai/led-base-16384"

# load test data
sca_test = Dataset.from_pandas(pd.read_csv('../data/test.tsv', sep='\t'))

# load model and tokenizer
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name).to("cuda").half()

def generate_summary(batch):
  inputs_dict = tokenizer(batch["main"], padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_summary"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch


result = sca_test.map(generate_summary, batched=True, batch_size=4)
result.to_pandas().to_csv('sca-longformer/0_shot_data_with_predictions.tsv', sep='\t', index=False)