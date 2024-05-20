import os
import pandas as pd

model_name = "models/longformer/sca-longformer"
predictions = pd.read_csv(os.path.join(model_name, 'data_with_predictions.tsv'), sep='\t')

print(predictions['predicted_summary'].values.tolist()[0])