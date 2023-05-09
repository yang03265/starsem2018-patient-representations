#!/usr/bin/env python

import numpy
numpy.random.seed(0)
import numpy as np
import sys
sys.dont_write_bytecode = True
import configparser, os, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from keras.utils import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import i2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

FEATURE_LIST = 'Model/features.txt'
NGRAM_RANGE = (1, 1) # use unigrams for cuis
MIN_DF = 0
def run_evaluation_sparse(disease, judgement, use_svd=False):
  """Train on train set and evaluate on test set"""

  print ('disease:', disease)
  print ('judgement:', judgement)

  cfg = configparser.ConfigParser()
  cfg.read('sparse.cfg')
  base = '/Users/jasonpeng/documents/representation'
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))
  alphabet_pickle=cfg.get('data', 'alphabet_pickle')
  print(alphabet_pickle)
  # handle training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=False,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x_train, y_train = train_data_provider.load_raw()
  print ('train examples:', len(x_train))

  vectorizer = CountVectorizer(
    ngram_range=NGRAM_RANGE,
    stop_words='english',
    min_df=MIN_DF,
    vocabulary=None,
    binary=False)
  train_count_matrix = vectorizer.fit_transform(x_train)

  tf = TfidfTransformer()
  train_tfidf_matrix = tf.fit_transform(train_count_matrix)

  # now handle the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x_test, y_test = test_data_provider.load_raw()
  print ('test examples:', len(x_test))

  test_count_matrix = vectorizer.transform(x_test)
  test_tfidf_matrix = tf.transform(test_count_matrix)

  if use_svd:
    # reduce sparse vector to 300 dimensions
    svd = TruncatedSVD(n_components=300)
    train_tfidf_matrix = svd.fit_transform(train_tfidf_matrix)
    test_tfidf_matrix = svd.transform(test_tfidf_matrix)
  #print (train_tfidf_matrix)
  y_train_boolean =  all(v == 0 for v in y_train)
  if (not y_train_boolean):
    classifier = LinearSVC(class_weight='balanced')
    classifier.fit(train_tfidf_matrix, y_train)
    predictions = classifier.predict(test_tfidf_matrix)
    p = precision_score(y_test, predictions, average='macro')
    r = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    print ('unique labels in train:', len(set(y_train)))
    print ('p = %.3f' % p)
    print ('r = %.3f' % r)
    print ('f1 = %.3f\n' % f1)

    return p, r, f1
  return 0, 0, 0


def run_evaluation_dense(disease, judgement):
  """Use pre-trained patient representations"""

  print ('disease:', disease)
  print ('judgement:', judgement)

  cfg = configparser.ConfigParser()
  cfg.read('dense.cfg')
  base = '/Users/jasonpeng/documents/representation'
  train_data = os.path.join(base, cfg.get('data', 'train_data'))
  train_annot = os.path.join(base, cfg.get('data', 'train_annot'))
  test_data = os.path.join(base, cfg.get('data', 'test_data'))
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('HL').output)

  # load training data first
  train_data_provider = DatasetProvider(
    train_data,
    train_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_train, y_train = train_data_provider.load()

  classes = len(set(y_train))
  print ('unique labels in train:', classes)
  maxlen = cfg.getint('data', 'maxlen')
  x_train = pad_sequences(x_train, maxlen=maxlen)

  # make training vectors for target task
  print ('original x_train shape:', x_train.shape)
  print(x_train)
  x_train = interm_layer_model.predict(x_train)
  print ('new x_train shape:', x_train.shape)

  # now load the test set
  test_data_provider = DatasetProvider(
    test_data,
    test_annot,
    disease,
    judgement,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'),
    min_token_freq=cfg.getint('args', 'min_token_freq'))
  x_test, y_test = test_data_provider.load()
  x_test = pad_sequences(x_test, maxlen=maxlen)
  
  # make test vectors for target task
  print ('original x_test shape:', x_test.shape)
  x_test = interm_layer_model.predict(x_test)
  print ('new x_test shape:', x_test.shape)
  
  classifier = LinearSVC(class_weight='balanced')
  #print(x_train)
  #print(y_train)
  y_train_boolean =  all(v == 0 for v in y_train)
  print(x_train)
  if (not (np.isnan(x_train).any()) and (not y_train_boolean)):
    model = classifier.fit(x_train, y_train)
    
    predictions = classifier.predict(x_test)
    p = precision_score(y_test, predictions, average='macro')
    r = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    print ('p = %.3f' % p)
    print ('r = %.3f' % r)
    print ('f1 = %.3f\n' % f1)
    return p, r, f1
  return 0,0,0

def run_evaluation_all_diseases():
  """Evaluate classifier performance for all 16 comorbidities"""

  exclude = set()

  cfg = configparser.ConfigParser()
  cfg.read('sparse.cfg')
  base = '/Users/jasonpeng/documents/representation'
  judgement = cfg.get('data', 'judgement')
  evaluation = cfg.get('data', 'evaluation')
  test_annot = os.path.join(base, cfg.get('data', 'test_annot'))

  ps = []
  rs = []
  f1s = []
  for disease in i2b2.get_disease_names(test_annot, exclude):
    if evaluation == 'sparse':
      # use bag-of-word vectors
      p, r, f1 = run_evaluation_sparse(disease, judgement)
      if (p == 0):
        continue
    elif evaluation == 'svd':
      # use low dimensional vectors obtained via svd
      p, r, f1 = run_evaluation_svd(disease, judgement)
    else:
      # use learned patient vectors
      p, r, f1 = run_evaluation_dense(disease, judgement)
      if (p == 0):
        continue
    ps.append(p)
    rs.append(r)
    f1s.append(f1)
  
  print ('average p =', numpy.mean(ps))
  print ('average r =', numpy.mean(rs))
  print ('average f1 =', numpy.mean(f1s))

if __name__ == "__main__":

  run_evaluation_all_diseases()
