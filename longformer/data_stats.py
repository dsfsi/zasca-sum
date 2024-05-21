import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

train_data = pd.read_csv('../data/train.tsv', sep='\t')
val_data = pd.read_csv('../data/dev.tsv', sep='\t')
test_data = pd.read_csv('../data/test.tsv', sep='\t')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

data = ['main', 'media-summary']

df = pd.concat([train_data, val_data, test_data])

d_w_l = {}

words = df[data[0]].apply(word_tokenize)
word_lengths = words.apply(lambda x: len(x))
y = np.array(word_lengths)

words = df[data[1]].apply(word_tokenize)
word_lengths = words.apply(lambda x: len(x))
x = np.array(word_lengths)

plt.scatter(x, y, s=10, label='Data points')

# Calculate the line of best fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept

# Plot the line of best fit
plt.plot(x, line, color='red', label=f'Line of Best Fit: y={slope:.2f}x + {intercept:.2f}')

plt.ylabel('Judgment length')
plt.xlabel('Media summary length')
plt.legend()

plt.savefig('../data/stats_figures/main_vs_media_summary_token_length_correlation.pdf')
plt.savefig('../data/stats_figures/main_vs_media_summary_token_length_correlation.png')