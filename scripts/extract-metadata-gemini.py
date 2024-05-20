import time
import pathlib
import textwrap
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import google.generativeai as genai

genai.configure(api_key='AIzaSyB52g2d0lm-_fezSvsMiEjDU6tB9Tlbiws')

generation_config = {
  "candidate_count": 1,
  "temperature": 1.0,
  "top_p": 0.7,
}

safety_settings=[
  {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

def generate_prompt(text):
  return f"""Extract the exact metadata without rewording any information in the following format:
    
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

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)
judgment_df = pd.read_csv('data/all_jugdments_v_2.tsv', sep='\t', lineterminator='\n')

error_j = []
uns_j = []
metadata = []

for j_no, text in tqdm(zip(judgment_df['j_no'].values.tolist(), judgment_df.main.values.tolist())):
  try:
    extracted_metadata = model.generate_content(generate_prompt(text))
    try:
      metadata.append({h[0].lower().strip(): h[1].lower().strip() for h in [l.split(':::') for l in extracted_metadata.text.split('\n')]})
    except:
      error_j.append({h[0].lower().strip(): 'not available' for h in [l.split(':::') for l in extracted_metadata.text.split('\n')]})
  except Exception as msg:
    uns_j.append(f'{j_no}\n')
    pprint(msg)
    time.sleep(5)
    continue

df = pd.DataFrame(metadata)
df.to_csv('data/sca-judgments-metadata.tsv', sep='\t', index=False)

df_err = pd.DataFrame(error_j)
df_err.to_csv('data/sca-judgments-error-metadata.tsv', sep='\t', index=False)

with open('data/sca-unsuccessful-extract.txt', 'w') as f:
  f.writelines(uns_j)