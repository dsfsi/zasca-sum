import os
import json
import pandas as pd
from evaluate import load
from statistics import mean

from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel

file_name = 'legal_led_data_with_predictions'
data_file = pd.read_csv(f'sca-longformer/{file_name}.tsv', sep='\t')

rouge = load('rouge')
bertscore = load("bertscore")

predictions = data_file['predicted_summary']
references = data_file['media-summary']
results = rouge.compute(predictions=predictions, references=references)

results = {metric: round(score, 4) * 100 for metric, score in results.items()}

bert_result = bertscore.compute(predictions=predictions, references=references, model_type='bert-base-uncased')

results['bertscore precision'] = round(mean(bert_result['precision']), 4) * 100
results['bertscore recall'] = round(mean(bert_result['recall']), 4) * 100
results['bertscore f1'] = round(mean(bert_result['f1']), 4) * 100

print(results)

with open(f'sca-longformer/{file_name}.json', 'w') as json_file:
  json.dump(results, json_file)