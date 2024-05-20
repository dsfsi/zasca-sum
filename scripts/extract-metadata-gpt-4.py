import openai
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
client = OpenAI(
  api_key='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'
)

def extract_metadata(text, client):
  prompt = f"""Extract the exact metadata without rewording any information in the following format:
  
  case name:::
  case number:::
  location:::
  date of hearing:::
  date of judgement:::
  applicants::: []
  respondents::: []
  coram::: []
  neutral citation:::
  summary:::
  order:::
  
  from the given text: {text}
  
  Extracted Information:"""

  response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": "Extract information."},
    ],
    stop=None
  )
  return response

judgment_df = pd.read_csv('data/all_jugdments_v_2.tsv', sep='\t', lineterminator='\n')

error_j = []
metadata = []

for j_no, text in tqdm(zip(judgment_df['j_no'].values.tolist(), judgment_df.main.values.tolist())):
  extract_metadata = ''
  try:
    extracted_metadata = extract_metadata(text, client)
    metadata.append({h[0].lower().strip(): h[1].lower().strip() for h in [l.split(':::') for l in extracted_metadata.choices[0].message.content.split('\n')]})
  except:
    print(f': error extracting {j_no}')
    error_j.append({j_no: extract_metadata})

df = pd.DataFrame(metadata)
df.to_csv('data/sca-judgments-metadata.tsv', sep='\t', index=False)

df_err = pd.DataFrame(error_j)
df_err.to_csv('data/sca-judgments-error-metadata.tsv', sep='\t', index=False)