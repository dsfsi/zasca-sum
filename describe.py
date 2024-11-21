import logging
import pandas as pd
from pathlib import Path
from utils import DataLoader, SCAPlotter, TextProcessor, TopicModeling, DATA_ANALYSIS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Initialising the data loader, plotter, text processor and topic modeler')
dl = DataLoader()
plotter = SCAPlotter()
text_processor = TextProcessor(dl)
topic_modeler = TopicModeling()

# plot case distribution
logging.info('Plotting the case distribution on all data')
plotter.plot_case_distribution(dl.load_data('all'))

# get the data with summaries
logging.info('Loading the data with summaries only for further analysis.')
df = dl.load_data('with_summaries')

# prepare the text
logging.info('Preparing the text: dropping duplicates, removing null values, etc.')
df = text_processor.prepare_text(df, target_columns=['input', 'output'])

# get all stats
logging.info('Getting all stats for the text and summary')
stats_file = DATA_ANALYSIS_PATH / 'data_with_stats.csv'
if stats_file.exists():
  stats = pd.read_csv(stats_file)
  df = pd.concat([df, stats], axis=1)

stats = df.copy()
df = text_processor.get_all_stats(df)

if df.equals(stats):
  logging.info('Data and stats are the same. All stats are calculated up to date.')
else:
  stats = df.drop(columns=['text', 'summary'])
  stats.to_csv(stats_file, index=False)
  logging.info(f'Data with stats saved to {stats_file}')
  del stats

logging.info('Plotting the summary vs judgment length')
plotter.plot_summary_vs_judgment_length(df)

logging.info('Plotting the summary and judgment stats')
plotter.plot_length_distribution(df, columns=['text_sent_count', 'text_word_count', 'text_char_count'], file_name='judgment_stats')
plotter.plot_length_distribution(df, columns=['text_sent_density','text_word_density'], file_name='judgment_density_stats')

plotter.plot_length_distribution(df, columns=['sum_sent_count', 'sum_word_count', 'sum_char_count'], file_name='summary_stats')
plotter.plot_length_distribution(df, columns=['sum_sent_density','sum_word_density'], file_name='summary_density_stats')

# get the pos tags
logging.info('Getting the POS tags for the text and summary')
columns = ['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X']

# plot the pos tags
logging.info('Plotting the POS tags for the text and summary')
postags = ['ADJ','ADP','ADV','CONJ','DET','NOUN']

df_text = df[[f'text_{p}' for p in postags]]
df_text.columns = [p for p in postags]
plotter.plot_length_distribution(df_text, columns=postags, plot_boxplots=False, file_name='judgment_pos_tags')

df_summary = df[[f'sum_{p}' for p in postags]]
df_summary.columns = [p for p in postags]
plotter.plot_length_distribution(df_summary, columns=postags, plot_boxplots=False, file_name='summary_pos_tags')

del df_text, df_summary

# print some unknown words
logging.info('Printing some unknown words')
print('Unknown words: ', df['text_unknown_words'].values[5])

# plot unknown words stats in text and summary
logging.info('Plotting the unknown words stats')
unknown_words_columns = ['text_unknown_count', 'sum_unknown_count']
plotter.plot_length_distribution(df, columns=unknown_words_columns, file_name='unknown_words_stats')

# plot puncs and stopwords
logging.info('Plotting the punctuation and stopwords stats')
target_columns = ['text_stopw_count', 'sum_stopw_count', 'text_punc_count','sum_punc_count']
plotter.plot_length_distribution(df, columns=target_columns, file_name='punc_stopw_and_punc_stats')

# clean the data for topic modeling
logging.info('Cleaning the text and summary for topic modeling')
cleaned_text, cleaned_summary = text_processor.remove_stopwords(df, target_columns=['text', 'summary'])

plotter.plot_wordcloud(cleaned_text, file_name='judgment_wordcloud')
plotter.plot_wordcloud(cleaned_summary, file_name='summary_wordcloud')

# Visualise the 20 most common words in the judgment
logging.info('Visualising the 20 most common words in the judgment')
tf, tf_feature_names = text_processor.get_vectorizer_features(cleaned_text)
plotter.plot_most_common_words(tf, tf_feature_names, file_name='judgment_most_common_words')

# # perform lda analysis, this takes a lot of time
# logging.info('Performing LDA analysis on the judgment')
# topic_modeler.perform_lda_analysis(cleaned_text, tf_vectorizer, file_name='judgment_lda_analysis')

# Visualise the 20 most common words in the summary
logging.info('Visualising the 20 most common words in the summary')
tf, tf_feature_names = text_processor.get_vectorizer_features(cleaned_summary)
plotter.plot_most_common_words(tf, tf_feature_names, file_name='summary_most_common_words')

# # perform lda analysis, this takes a lot of time
# logging.info('Performing LDA analysis on the summary')
# topic_modeler.perform_lda_analysis(cleaned_summary, tf_vectorizer, file_name='summary_lda_analysis')

# perform bertopic analysis
logging.info('Performing BERTopic analysis on the judgment and summary')
topic_modeler.perform_bertopic_analysis(cleaned_text=cleaned_text, cleaned_summary=cleaned_summary, output_path='bertopic/')
judgment_model, _ = topic_modeler.perform_bertopic_analysis(cleaned_text=cleaned_text, save_topic_info=False, output_path='bertopic/judgments/')
summary_model, _ = topic_modeler.perform_bertopic_analysis(cleaned_summary=cleaned_summary, save_topic_info=False, output_path='bertopic/summaries/')

# calculate topic overlap
logging.info('Calculating the topic overlap between the judgment and summary')
overlap_matrix = topic_modeler.calculate_overlap_matrix(judgment_model, summary_model)

# plot the overlap matrix
logging.info('Plotting the overlap matrix')
plotter.plot_overlap_heatmap(overlap_matrix, file_name='overlap_matrix')