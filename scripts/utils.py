import fitz
import random
import logging
import zipfile
import re, string
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import pickle
import pyLDAvis
import pyLDAvis.lda_model as lda

from bertopic import BERTopic
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation as LDA

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

tqdm.pandas()
plt.rcParams["font.family"] = "Tahoma"
sns.set_theme(style="whitegrid", font="Tahoma")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOME_DIR = Path("..")

EXTRACTED_DATA_DIR = HOME_DIR / "data"
RAW_DATA_DIR = EXTRACTED_DATA_DIR / "raw"
PROCESSED_DATA_DIR = EXTRACTED_DATA_DIR / "processed"
GLOVE_EMBEDDINGS_FILE = EXTRACTED_DATA_DIR / "glove.6B.100d.txt"

DATA_ANALYSIS_PATH = HOME_DIR / "data_analysis"
FIGURES_DIR = DATA_ANALYSIS_PATH / "plots"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
POST_TAGS = ['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PRT','PRON','VERB','.','X']


class FileManager:
  """Handles file operations, including zip and unzipping folders and saving text to files."""

  @staticmethod
  def unzip_data(zip_path, extract_to):
    """
    Unzips a ZIP file to a specified directory.

    Parameters:
    - zip_path (str or Path): Path to the ZIP file.
    - extract_to (str or Path): Target directory to extract files to.

    Raises:
    - FileNotFoundError: If the ZIP file does not exist.
    - RuntimeError: If the file is not a valid ZIP archive.
    """
    zip_file = Path(zip_path)
    extract_to = Path(extract_to)
    if not zip_file.exists():
      raise FileNotFoundError(f"ZIP file not found: {zip_file}")

    target_dir = extract_to / zip_file.stem
    if target_dir.exists():
      logging.info(f"Directory already exists: {target_dir}")
      return

    try:
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        logging.info(f"Extracted {zip_file} to {target_dir}")
    except zipfile.BadZipFile as e:
      raise RuntimeError(f"Invalid ZIP file: {zip_file}") from e

  @staticmethod
  def save_text(text, file_path):
    """
    Saves text to a file.

    Parameters:
    - text (str): Text to save.
    - file_path (str or Path): Target file path.

    Raises:
    - IOError: If writing to the file fails.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
      logging.info(f"Saved text to {file_path}")
    except IOError as e:
      logging.error(f"Failed to save text to {file_path}: {e}")
      raise


class PDFExtractor:
  """Extracts and cleans text from PDF documents."""

  @staticmethod
  def extract_text(pdf_path):
    """
    Extracts and processes text from a PDF file.

    Parameters:
    - pdf_path (str or Path): Path to the PDF file.

    Returns:
    - str: Cleaned and processed text.

    Raises:
    - FileNotFoundError: If the PDF file does not exist.
    - RuntimeError: If the PDF cannot be opened.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
      logging.error(f"PDF file not found: {pdf_path}")
      raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
      doc = fitz.open(pdf_path)
      text_lines = [
        PDFExtractor._clean_line(page.get_text("text"))
        for page in doc
      ]
      doc.close()
      return '\n'.join(PDFExtractor._combine_paragraphs(text_lines))
    except Exception as e:
      logging.error(f"Error extracting text from {pdf_path}: {e}")
      raise RuntimeError(f"Error extracting text from {pdf_path}: {e}")

  @staticmethod
  def _clean_line(text):
    """
    Cleans a line of text by removing unwanted content.

    Parameters:
    - text (str): The text to clean.

    Returns:
    - list: List of cleaned sentences.
    """
    paragraphs = [line.strip() for line in sent_tokenize(text)]
    return [p for p in paragraphs if not PDFExtractor._is_numeric_string(p)]

  @staticmethod
  def _combine_paragraphs(lines):
    """
    Combines lines into paragraphs based on paragraph markers.

    Parameters:
    - lines (list of str): List of text lines.

    Returns:
    - list: Combined paragraphs.
    """
    combined = []
    for line in lines:
      if PDFExtractor._is_paragraph_marker(line):
        if combined:
          combined[-1] += f' {line}'
        else:
          combined.append(line)
      else:
        combined.append(line)
    return combined

  @staticmethod
  def _is_numeric_string(string):
    """
    Checks if a string is numeric and less than 1000.

    Parameters:
    - string (str): The string to check.

    Returns:
    - bool: True if numeric and less than 1000, otherwise False.
    """
    return string.isdigit() and int(string) < 1000

  @staticmethod
  def _is_paragraph_marker(line):
    """
    Determines if a line is a paragraph marker.

    Parameters:
    - line (str): The line to check.

    Returns:
    - bool: True if it matches paragraph marker criteria, otherwise False.
    """
    return line.startswith("[") and line.endswith("]") and line[1:-1].isdigit()


class DataLoader:
  """Loads and processes TSV data files into DataFrames."""

  def __init__(self, base_dir=PROCESSED_DATA_DIR, file_extension="tsv"):
    """
    Initialize the DataLoader.

    Parameters:
    - base_dir (Path): Base directory containing the processed data.
    - file_extension (str): Extension of data files to read (default: 'tsv').
    """
    self.base_dir = Path(base_dir)
    self.file_extension = file_extension

  def load_data(self, data_type, column_name=None):
    """
    Load data based on the specified type.

    Parameters:
    - data_type (str): One of ['with_summaries', 'without_summaries', 'all'].

    Returns:
    - pd.DataFrame: Concatenated DataFrame with a 'split' column.
    """
    paths = {
      'with_summaries': [self.base_dir / "with_summaries" / f"{split}.{self.file_extension}" for split in ['train', 'dev', 'test']],
      'without_summaries': [self.base_dir / "without_summaries" / f"all_data.{self.file_extension}"],
      'all': [self.base_dir / "without_summaries" / f"all_data.{self.file_extension}"] +
              [self.base_dir / "with_summaries" / f"{split}.{self.file_extension}" for split in ['train', 'dev', 'test']]
    }

    if data_type not in paths:
      raise ValueError(f"Invalid data type specified: {data_type}. Expected one of {list(paths.keys())}.")

    valid_paths = [path for path in paths[data_type] if path.exists()]
    missing_paths = [path for path in paths[data_type] if not path.exists()]

    if missing_paths:
      logging.warning(f"Missing files: {missing_paths}")

    if not valid_paths:
      raise FileNotFoundError("No valid data files found to load.")
    
    if column_name:
      return self._read_files(valid_paths)[column_name]

    return self._read_files(valid_paths)

  @staticmethod
  def _read_files(paths):
    """
    Read and concatenate data files into a single DataFrame.

    Parameters:
    - paths (list of Path): Paths to the files to read.

    Returns:
    - pd.DataFrame: Combined DataFrame with a 'split' column.
    """
    df_list = []
    for path in paths:
      logging.info(f"Loading file: {path}")
      try:
        df = pd.read_csv(path, sep='\t')
        df['split'] = path.stem
        df_list.append(df)
      except Exception as e:
        logging.error(f"Failed to read {path}: {e}")

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


class GloveVectorizer:
  """
  Maps words to GloVe embeddings and computes sentence embeddings
  by averaging word vectors.
  """

  def __init__(self, embedding_file):
    """
    Initializes the vectorizer with GloVe embeddings.

    Args:
        embedding_file (str): Path to the GloVe embedding file.
    """
    self.word2vec = {}
    self.embedding = []
    self.idx2word = []

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    try:
      logging.info("Loading word vectors...")
      with open(embedding_file, encoding='utf-8') as f:
        for line in f:
          values = line.split()
          word = values[0]
          vec = np.asarray(values[1:], dtype='float32')
          self.word2vec[word] = vec
          self.embedding.append(vec)
          self.idx2word.append(word)

      self.embedding = np.array(self.embedding)
      self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
      self.V, self.D = self.embedding.shape
      logging.info(f"Found {len(self.word2vec)} word vectors.")
    except FileNotFoundError:
      logging.error(f"Embedding file '{embedding_file}' not found.")
      raise FileNotFoundError(f"Embedding file '{embedding_file}' not found.")
    except Exception as e:
      logging.error(f"Error loading embeddings: {e}")
      raise RuntimeError(f"Error loading embeddings: {e}")

  def fit(self, data):
    """Placeholder for potential future implementation."""
    pass

  def get_vocabulary(self):
    """
    Returns the vocabulary of the embeddings.

    Returns:
        list: A list of all words in the GloVe vocabulary.
    """
    return list(self.word2vec.keys())

  def transform(self, data, return_unknowns=False):
    """
    Transforms a list of sentences into mean GloVe embeddings.

    Args:
      data (list of str): Sentences to transform.
      return_unknowns (bool): If True, also return unknown words.

    Returns:
      np.ndarray: Mean GloVe embeddings for each sentence.
      list: (Optional) List of unknown words for each sentence.
    """
    X = np.zeros((len(data), self.D))
    unknown_words = []
    emptycount = 0

    for n, sentence in enumerate(data):
      tokens = sentence.lower().split()
      vecs = []
      unknowns = []

      for word in tokens:
        if word in self.word2vec:
          vecs.append(self.word2vec[word])
        else:
          unknowns.append(word)

      if vecs:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1

      if return_unknowns:
        unknown_words.append(unknowns)

    if emptycount > 0:
      print(f"Warning: {emptycount} sentences had no known words.")

    return (X, unknown_words) if return_unknowns else X

  def fit_transform(self, data, return_unknowns=False):
    """
    Fits and transforms the data.

    Args:
      data (list of str): Sentences to transform.
      return_unknowns (bool): If True, also return unknown words.

    Returns:
      np.ndarray: Mean GloVe embeddings for each sentence.
      list: (Optional) List of unknown words for each sentence.
    """
    self.fit(data)
    return self.transform(data, return_unknowns)

class TextProcessor:
  """Processes text data for analysis and visualization."""

  def __init__(self, data_loader):
    self.data_loader = data_loader

  @staticmethod
  def tokenize_stats(df, col_name, tokenize_type):
    tokenizer = sent_tokenize if tokenize_type == 'sent' else word_tokenize
    stats = df[col_name].dropna().apply(lambda x: len(tokenizer(x)))
    return stats

  @staticmethod
  def get_punctuation():
    return string.punctuation

  @staticmethod
  def get_stopwords(language='english'):
    return set(nltk.corpus.stopwords.words(language))

  @staticmethod
  def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')

  @staticmethod
  def count_stopwords(text, stopwords):
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stopwords]
    return len(stopwords_x)

  @staticmethod
  def replace_punctuation(text, punctuation):
    table = str.maketrans(punctuation, ' ' * len(punctuation))
    return text.translate(table)

  @staticmethod
  def get_unknown_words(text, vocab):
    tokens = word_tokenize(text)
    unknown = [t for t in tokens if t not in vocab.word2vec]
    return unknown
  
  @staticmethod
  def get_pos_tags(sentences, columns, data_type, tagset='universal'):
    ''' Extract the part-of-speech taggings of the sentence
        Input:
        - sentence: string, sentence to tag
        - tagset: string, tagset or the set of tags to search for
    '''
    tags = []
    columns = [f'{data_type}_{c}' for c in columns]
    for sent in tqdm(sentences):
      pos_tags = Counter([j for _,j in nltk.pos_tag(word_tokenize(sent), tagset=tagset)])
      pos_tags = {f'{data_type}_{k}':v for k,v in dict(pos_tags).items()}
      tags.append(pos_tags)
    
    return pd.DataFrame(tags, columns=columns).fillna(0)
  
  def remove_stopwords(self, df, target_columns=None):
    ''' Apply some basic techniques for cleaning a text for an analysis of words

    Input:
      - text: text to be cleaned
    Output:
      - result: cleaned text
    '''
    def clean_text(text, stopwords):
      text = text.lower()
      pattern =  r'[^a-zA-Z\s]'    
      text = re.sub(pattern, '', text)

      tokens = nltk.word_tokenize(text)    
      tokens = [token.strip() for token in tokens]    
      text = ' '.join([token for token in tokens if token not in stopwords])
      return text
    
    if target_columns:
      logging.info(f"Removing stopwords for columns: {target_columns}")
      stopwords = self.get_stopwords()
      cleaned_text = []
      for col in target_columns:
        cleaned_text.append(df[col].progress_apply(lambda x: clean_text(x, stopwords)).tolist())
      return cleaned_text

  def prepare_text(self, df, target_columns=None, drop_duplicates=True, drop_na=True):
    if target_columns and len(target_columns) == 2:
      logging.info(f"Preparing text data for columns: {target_columns}")
      try:
        df = df[target_columns]
      except KeyError as e:
        logging.error(f"Invalid columns specified: {e}")
        raise ValueError(f"Invalid columns specified: {e}")
      if drop_duplicates:
        df.drop_duplicates(subset=target_columns[0], inplace=True)
        logging.info(f"Dropped duplicates, new shape: {df.shape}")
      if drop_na:
        df.dropna(inplace=True)
        logging.info(f"Dropped NA values, new shape: {df.shape}")
      df.reset_index(drop=True, inplace=True)
      df.columns = ['text', 'summary']
      logging.info(f"Renamed columns to 'text' and 'summary'")

      logging.info("Cleaning unicode characters and extra spaces...")
      df['text'] = df['text'].apply(lambda x: self.unicode_to_ascii(x.strip()))
      df['summary'] = df['summary'].apply(lambda x: self.unicode_to_ascii(x.strip()))

      logging.info(f"Data prepared, new shape: {df.shape}")

      return df
    else:
      logging.error("Invalid columns or number of target columns specified.")
      raise ValueError('No target columns specified, or invalid number of columns.')

  def get_vectorizer_features(self, texts, max_df=0.9, min_df=25, max_features=5000):
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    tf = tf_vectorizer.fit_transform(texts)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    return tf, tf_feature_names

  def get_all_stats(self, df):
    """
    Generate and add statistical metrics for text and summary columns in a DataFrame.
    
    Parameters:
      df (pd.DataFrame): Input DataFrame containing 'text' and 'summary' columns.

    Returns:
        pd.DataFrame: DataFrame with added statistical columns.
    """
    punc = self.get_punctuation()
    stopwords = self.get_stopwords()
    vocab = GloveVectorizer(GLOVE_EMBEDDINGS_FILE)

    def add_stat_column(column_name, compute_func, *args, **kwargs):
      if column_name not in df.columns:
        logging.info(f"Calculating {column_name}...")
        df[column_name] = compute_func(*args, **kwargs)
      else:
        logging.info(f"{column_name} already present in stats, skipping...")

    logging.info("Calculating text statistics (sentences, tokens, characters, etc.)...")
    add_stat_column('text_sent_count', self.tokenize_stats, df, 'text', 'sent')
    add_stat_column('text_word_count', self.tokenize_stats, df, 'text', 'word')
    add_stat_column('text_char_count', lambda x: x['text'].progress_apply(lambda t: len(t.replace(" ", ""))), df)
    add_stat_column('text_sent_density', lambda x: x['text_sent_count'] / (x['text_word_count'] + 1), df)
    add_stat_column('text_word_density', lambda x: x['text_word_count'] / (x['text_char_count'] + 1), df)
    add_stat_column('text_punc_count', lambda x: x['text'].progress_apply(lambda t: sum(1 for char in t if char in punc)), df)
    add_stat_column('text_stopw_count', lambda x: x['text'].progress_apply(lambda t: self.count_stopwords(t, stopwords)), df)
    add_stat_column('text_unknown_words', lambda x: x['text'].progress_apply(lambda t: self.get_unknown_words(self.replace_punctuation(t.lower(), string.punctuation), vocab)), df)
    add_stat_column('text_unknown_count', lambda x: x['text_unknown_words'].progress_apply(lambda t: len(t) if isinstance(t, list) else 0), df)    

    logging.info("Calculating summary statistics (sentences, tokens, characters, etc.)...")
    add_stat_column('sum_sent_count', self.tokenize_stats, df, 'summary', 'sent')
    add_stat_column('sum_word_count', self.tokenize_stats, df, 'summary', 'word')
    add_stat_column('sum_char_count', lambda x: x['summary'].progress_apply(lambda t: len(t.replace(" ", ""))), df)
    add_stat_column('sum_sent_density', lambda x: x['sum_sent_count'] / (x['sum_word_count'] + 1), df)
    add_stat_column('sum_word_density', lambda x: x['sum_word_count'] / (x['sum_char_count'] + 1), df)
    add_stat_column('sum_punc_count', lambda x: x['summary'].progress_apply(lambda t: sum(1 for char in t if char in punc)), df)
    add_stat_column('sum_stopw_count', lambda x: x['summary'].progress_apply(lambda t: self.count_stopwords(t, stopwords)), df)
    add_stat_column('sum_unknown_words', lambda x: x['summary'].progress_apply(lambda t: self.get_unknown_words(self.replace_punctuation(t.lower(), string.punctuation), vocab)), df)
    add_stat_column('sum_unknown_count', lambda x: x['sum_unknown_words'].progress_apply(lambda t: len(t) if isinstance(t, list) else 0), df)

    logging.info("Adding POS tags for text and summary...")
    text_columns = [f'text_{p}' for p in POST_TAGS]
    if not all(col in df.columns for col in text_columns):
      df = pd.concat([df, self.get_pos_tags(df['text'], POST_TAGS, 'text')], axis=1)
    else:
      logging.info("Text POS tags already present in stats, skipping...")
    sum_columns = [f'sum_{p}' for p in POST_TAGS]
    if not all(col in df.columns for col in sum_columns):
      df = pd.concat([df, self.get_pos_tags(df['summary'], POST_TAGS, 'sum')], axis=1)
    else:
      logging.info("Summary POS tags already present in stats, skipping...")

    logging.info("All statistics have been calculated successfully.")
    return df

class SCAPlotter:
  """Generates plots for data visualization."""

  def __init__(self):
    self.labels_dict = {
      'sum_word_count': 'Word Count of Summaries', 'text_word_count': 'Word Count of Judgments',
      'sum_char_count': 'Chararacter Count of Summaries', 'text_char_count': 'Chararacter Count of Judgments',
      'sum_word_density': 'Word Density of Summaries', 'text_word_density': 'Word Density of Judgments',
      'sum_punc_count': 'Punctuation Count of Summaries', 'text_punc_count': 'Punctuation Count of Judgments',
      'text_sent_count': 'Sentence Count of Judgments', 'sum_sent_count': 'Sentence Count of Summaries',
      'text_sent_density': 'Sentence Density of Judgments', 'sum_sent_density': 'Sentence Density of Summaries',
      'text_stopw_count': 'Stopwords Count of Judgments', 'sum_stopw_count': 'Stopwords Count of Summaries',
      'ADJ': 'adjective','ADP': 'adposition', 'ADV': 'adverb','CONJ': 'conjunction',
      'DET': 'determiner','NOUN': 'noun', 'text_unknown_count': 'Unknown words in Judgments',
      'sum_unknown_count': 'Unknown words in Summaries'
    }
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

  def plot_case_distribution(self, df):
    plt.figure(figsize=(7.5, 6))
    sns.countplot(data=df, x='type', hue='type', palette='muted', width=0.5)
    plt.ylabel('Number of Cases')
    plt.xlabel('Case Type')
    plt.xticks(rotation=0)
    plt.savefig(FIGURES_DIR / 'number_of_cases_by_type.png')
    plt.close()

  def plot_summary_vs_judgment_length(self, df):
    slope, intercept, _, _, _ = stats.linregress(df['text_word_count'], df['sum_word_count'])
    plt.figure(figsize=(7.5, 6))
    sns.scatterplot(x='text_word_count', y='sum_word_count', data=df, s=10, label='Data', color="dodgerblue")
    
    plt.xlabel('Judgment Length')
    plt.ylabel('Summary Length')
    plt.plot(df['text_word_count'], intercept + slope * df['text_word_count'], 'b', label=f'Best Fit: y = {slope:.2f}x + {intercept:.2f}')
    self._add_capacity_shading(df['text_word_count'], df['sum_word_count'])
    plt.legend()
    plt.savefig(FIGURES_DIR / 'data_summary_lengths.png')

    plt.close()

  def plot_length_distribution(self, df, columns, plot_histogram=True, plot_boxplots=True, file_name='stats'):
    if plot_histogram or plot_boxplots:
      if plot_histogram:
        self._plot_histograms(
          df, 
          np.array([columns]),
          self.labels_dict,
          show_kde=False,
          output_path=FIGURES_DIR / f'{file_name}_histograms.png'
        )
      if plot_boxplots:
        self._plot_boxplots(
          df,
          np.array([columns]),
          self.labels_dict,
          output_path=FIGURES_DIR / f'{file_name}_boxplots.png'
        )
    else:
      raise ValueError('No plots selected to be generated.')
    
  def plot_most_common_words(self, count_data, words, figsize=(15, 7), no_words=20, file_name=None, show_plot=False):
    """
    Draw a barplot showing the most common words in the data.

    Parameters:
    - count_data (sparse matrix): Document-term matrix containing word occurrences.
    - count_vectorizer (CountVectorizer): Fitted CountVectorizer object.
    - figsize (tuple): Figure size for the plot.
    - no_words (int): Number of most common words to display.
    - output_path (str): Path to save the plot.
    """
    total_counts = np.zeros(len(words))
    for t in count_data:
      total_counts += t.toarray()[0]

    count_dict = sorted(zip(words, total_counts), key=lambda x: x[1], reverse=True)[:no_words]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(figsize=figsize)
    plt.subplot(title=f'{no_words} most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x=x_pos, y=counts, palette='husl')
    plt.xticks(x_pos, words, rotation=45)
    plt.ylabel('Frequency')
    plt.tight_layout()
    if file_name:
      plt.savefig(FIGURES_DIR / f'{file_name}.png')
    if show_plot:
      plt.show()
    plt.close()

  def plot_bertopic_visualizations(self, model, output_path):
    """
    Generate and save BERTopic visualizations.
    """
    fig = model.visualize_barchart(top_n_topics=12)
    fig.write_html(output_path / "topic_barchart.html")

    hierarchical_fig = model.visualize_hierarchy()
    hierarchical_fig.write_html(output_path / "topic_hierarchy.html")

    heatmap_fig = model.visualize_heatmap()
    heatmap_fig.write_html(output_path / "topic_heatmap.html")

    word_cloud_fig = model.visualize_topics()
    word_cloud_fig.write_html(output_path / "topic_wordcloud.html")

  def plot_overlap_heatmap(self, overlap_matrix, file_name=None):
    """
    Plot a heatmap for the overlap matrix.

    Parameters:
      overlap_matrix (np.array): Overlap matrix between judgment and summary topics.
      output_path (str): Path to save the heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(overlap_matrix, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Topic Overlap Between Judgments and Summaries")
    plt.xlabel("Summary Topics")
    plt.ylabel("Judgment Topics")
    plt.savefig(FIGURES_DIR / f'{file_name}.png')
    plt.close()

  def plot_wordcloud(self, texts, background_color="white", max_words=1000, contour_width=3, contour_color='steelblue', file_name='wordcloud'):
    long_string = ','.join(texts)
    wordcloud = WordCloud(background_color=background_color, max_words=max_words, contour_width=contour_width, contour_color=contour_color)
    wordcloud.generate(long_string)
    wordcloud.to_image()
    wordcloud.to_file(FIGURES_DIR / f'{file_name}.png')

  def plot_lda_results(self, lda_model, tf, tf_vectorizer, file_name='lda_topics'):
    LDAvis_prepared = lda.prepare(lda_model, tf, tf_vectorizer)

    with open(FIGURES_DIR / f'{file_name}.pkl', 'wb') as f:
      pickle.dump(LDAvis_prepared, f)
    
    with open(FIGURES_DIR / f'{file_name}.pkl', 'rb') as f:
      LDAvis_prepared = pickle.load(f)
        
    pyLDAvis.save_html(LDAvis_prepared, FIGURES_DIR / f'{file_name}.html')

  @staticmethod
  def _plot_boxplots(data, plot_vars, labels, figsize=(15, 5), output_path=None, show_plot=False):
    """
    Plot boxplots for the specified variables with appropriate labels.

    Parameters:
    - data (pd.DataFrame): The data points to plot.
    - plot_vars (array-like): A (1, x) or (n, m) array containing column names to plot.
    - labels (dict): A dictionary mapping column names to their respective labels.
    - figsize (tuple): The size of the figure (default: (15, 5)).
    - output_path (str, optional): File path to save the plot.
    - show_plot (bool, optional): Whether to display the plot.

    Returns:
    - None
    """
    plot_vars = np.atleast_2d(plot_vars)
    nrows, ncols = plot_vars.shape

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i in range(nrows):
      for j in range(ncols):
        var = plot_vars[i, j]
        ax = axes[i, j]

        if var is not None:
          ax.set_title(labels.get(var, var))
          ax.grid(True)
          ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
          )
          if var in data.columns:
            ax.boxplot(data[var])
          else:
            ax.set_visible(False)
        else:
          ax.set_visible(False)

    fig.tight_layout()

    if output_path:
      plt.savefig(output_path)
    if show_plot:
      plt.show()
    plt.close()

  @staticmethod
  def _plot_histograms(data, plot_vars, labels, figsize=(15,5), show_kde=False, output_path=None, show_plot=False):
    ''' Function to plot the histograms of the variables in plot_vars
        Input:
        - data: a dataframe, containing the data points to plot
        - plot_vars: a (1,x) array, containing the columns to plot
        - xlim: a list, defines the max x value for every column to plot
        - labels: a dictionary, to map the column names to its label
        - figsize: a tuple, indicating the size of the figure
        - show_kde: a boolean, indicating if the kde should be shown
        - output_path: a string, indicating the path to save the file
    '''
    fig, axes = plt.subplots(1, plot_vars.shape[1], figsize=figsize, sharey=False, dpi=100)

    if plot_vars.shape[1] == 1:
      axes = [axes]

    for i in range(plot_vars.shape[1]):
      color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
      
      sns.histplot(
        data[plot_vars[0, i]], 
        color=color, 
        ax=axes[i], 
        bins=50, 
        kde=show_kde,
      )

      x_label = plot_vars[0, i].replace('sent', 'sentence')
      axes[i].set_xlabel(' '.join([l.capitalize() for l in x_label.split('_')[1:]]))
      axes[i].set_ylabel('Frequency')
      
      axes[i].set_title(labels[plot_vars[0, i]])

    fig.tight_layout()
    if output_path:
      plt.savefig(output_path)
    if show_plot:
      plt.show()
    plt.close()

  @staticmethod
  def _add_capacity_shading(input_stats, output_stats):
    model_input_length, model_output_length = 16384, 1024
    plt.gca().add_patch(
      plt.Rectangle((0, 0), model_input_length, max(output_stats) + 50,
                    color='red', alpha=0.3, linestyle='--', linewidth=1.5,
                    label=f"Judgments accommodated: {len([x for x in input_stats if x < model_input_length]):,}")
    )
    plt.gca().add_patch(
      plt.Rectangle((0, 0), max(input_stats) + 400, model_output_length,
                    color='green', alpha=0.3, linestyle='-', linewidth=1.5,
                    label=f"Summaries accommodated: {len([y for y in output_stats if y < model_output_length]):,}")
    )


class TopicModeling:
  """
  Class to perform topic modeling using LDA, UMAP, and HDBSCAN.
  """

  def __init__(self):
    self.plotter = SCAPlotter()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

  def perform_lda_analysis(self, texts, tf_vectorizer, no_top_words=8, n_components=10, max_iter=500, random_state=0, learning_method='online', file_name='lda_topics'):
    """
    Perform LDA topic modeling and save top words per topic.

    Parameters:
      texts (list of str): Input texts for LDA.
      tf_vectorizer (TfidfVectorizer or CountVectorizer): Vectorizer for text processing.
      no_top_words (int): Number of top words to display per topic.
      n_components (int): Number of topics.
      max_iter (int): Maximum number of iterations.
      random_state (int): Random state for reproducibility.
      learning_method (str): Learning method for LDA ('batch' or 'online').
      file_name (str): Name of the file to save topics.

    Returns:
      lda_model (LDA): Fitted LDA model.
    """
    logging.info("Vectorizing text data...")
    tf = tf_vectorizer.fit_transform(texts)

    logging.info("Fitting LDA model...")
    lda_model = LDA(
      n_components=n_components,
      learning_method=learning_method,
      max_iter=max_iter,
      random_state=random_state
    ).fit(tf)

    words = tf_vectorizer.get_feature_names_out()

    with open(FIGURES_DIR / f'{file_name}.txt', 'w') as f:
      for topic_idx, topic in enumerate(lda_model.components_):
        f.write(f"\nTopic #{topic_idx}:\n")
        f.write(" ".join([words[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) + "\n")

    self.plotter.plot_lda_results(lda_model, tf, tf_vectorizer, file_name)
    return lda_model

  def perform_bertopic_analysis(self, cleaned_text=None, cleaned_summary=None, output_path='bertopic', save_topic_info=True):
    """
    Perform BERTopic modeling and generate plots.

    Parameters:
      cleaned_text (list of str): List of cleaned text strings.
      cleaned_summary (list of str): List of cleaned summary strings.
      output_path (str): Directory path to save results.
      save_topic_info (bool): Save topic information as a CSV file.

    Returns:
      model (BERTopic): Trained BERTopic model.
      topic_info (pd.DataFrame): DataFrame containing topic information.
    """
    if cleaned_text is None and cleaned_summary is None:
      logging.error("No cleaned text or summary data provided.")
      raise ValueError("Please provide cleaned text and/or summary data.")
    
    if cleaned_text and cleaned_summary:
      logging.info('merging text and summary data...')
    elif cleaned_text:
      logging.info('using only text data...')
    elif cleaned_summary:
      logging.info('using only summary data...')
    
    combined_texts = cleaned_text or [] + cleaned_summary or []

    logging.info("Initializing and fitting BERTopic model...")
    model = BERTopic()
    model.fit_transform(combined_texts)

    topic_info = None
    topic_info_path = FIGURES_DIR / output_path
    topic_info_path.mkdir(parents=True, exist_ok=True)

    if save_topic_info:
      logging.info("Saving topic information to CSV file...")
      topic_info = model.get_topic_info()
      topic_info.to_csv(topic_info_path / "topic_info.csv", index=False)

    logging.info("Generating BERTopic visualizations...")
    self.plotter.plot_bertopic_visualizations(model, topic_info_path)

    return model, topic_info

  def calculate_overlap_matrix(self, judgment_model, summary_model, top_n=12):
    """
    Calculate the overlap matrix between judgment and summary topics.

    Args:
        judgment_model: The model containing judgment topics.
        summary_model: The model containing summary topics.
        top_n (int): The number of top topics to consider.

    Returns:
        np.ndarray: Overlap matrix between judgment and summary topics.
    """
    logging.info("Getting topic information from judgment and summary models.")
    
    # Get topic information
    judgment_topics = judgment_model.get_topic_info()['Topic'][:top_n].values
    summary_topics = summary_model.get_topic_info()['Topic'][:top_n].values

    logging.info("Initializing overlap matrix.")
    # Initialize overlap matrix
    overlap_matrix = np.zeros((top_n, top_n))

    for i, j_topic_id in enumerate(judgment_topics):
      if j_topic_id == -1:  # Skip outliers
        logging.info(f"Skipping outlier topic in judgment model at index {i}.")
        continue
      logging.info(f"Processing judgment topic {j_topic_id} at index {i}.")
      j_terms = {term for term, _ in judgment_model.get_topic(j_topic_id)}
      for j, s_topic_id in enumerate(summary_topics):
        if s_topic_id == -1:  # Skip outliers
          logging.info(f"Skipping outlier topic in summary model at index {j}.")
          continue
        logging.info(f"Processing summary topic {s_topic_id} at index {j}.")
        s_terms = {term for term, _ in summary_model.get_topic(s_topic_id)}
        # Calculate Jaccard similarity
        overlap_matrix[i, j] = len(j_terms & s_terms) / len(j_terms | s_terms)
        logging.info(f"Calculated Jaccard similarity for judgment topic {j_topic_id} and summary topic {s_topic_id}: {overlap_matrix[i, j]}")

    logging.info("Overlap matrix calculation complete.")
    return overlap_matrix