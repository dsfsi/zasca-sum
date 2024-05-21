import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to("cuda").half()

def generate_summary(batch):
  inputs_dict = tokenizer(batch["main"], padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  summary_ids = model.generate(input_ids, attention_mask=attention_mask,
                  num_beams=9,
                  no_repeat_ngram_size=3,
                  length_penalty=2.0,
                  min_length=512,
                  max_length=1024,
                  early_stopping=True
                )
  batch["predicted_summary"] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
  return batch

sca_test = Dataset.from_pandas(pd.read_csv('../data/test.tsv', sep='\t'))

result = sca_test.map(generate_summary, batched=True, batch_size=4)
result.to_pandas().to_csv('sca-longformer/0_shot_data_with_predictions.tsv', sep='\t', index=False)