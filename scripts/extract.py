import os
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from utils import FileManager, PDFExtractor

FileManager.unzip_data('../data/raw.zip', '../data')

directories = {
  "with_summaries": {
    "path": Path('../data/raw/with_summaries'),
    "columns": ['id', 'type', 'year', 'main_judgement', 'media_summary'],
    "has_summary": True
  },
  "without_summaries": {
    "path": Path('../data/raw/without_summaries'),
    "columns": ['id', 'type', 'year', 'main_judgement'],
    "has_summary": False
  }
}

for dir_key, dir_info in directories.items():
  data = []
  pdir = dir_info["path"]

  for root, dirs, files in tqdm(os.walk(pdir)):
    if not files:
      continue
    try:
      dtails = Path(root).parts
      record = [
        dtails[-1].split('-')[0],
        dtails[3],
        dtails[4].split('-')[-1]
      ]
      record.append(PDFExtractor.extract_text_from_pdf(f'{root}/main-judgement.pdf'))
      if dir_info["has_summary"]:
        record.append(PDFExtractor.extract_text_from_pdf(f'{root}/media-summary.pdf'))
      
      data.append(record)
    except Exception as e:
      print(f"Skipping {root} due to error: {e}")
      continue

  df = pd.DataFrame(data, columns=dir_info["columns"])
  df.to_csv(f'../data/processed/judgments_{dir_key}.tsv', sep='\t', index=False)