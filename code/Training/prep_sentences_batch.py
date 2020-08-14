import sys
sys.path.append('../')
from utils import utils_train,bias_utils
from gensim.models.word2vec import Word2Vec
import logging
import pandas as pd
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# paths
ROOT = "/kbdata/ResearchDrive"
SENT_OUTPUT = "/kbdata/Processed/Sentences"

WORKERS = 8

# training parameters
START_YEAR = 1909
END_YEAR = 1910


print(f'Collecting data from {START_YEAR} to {END_YEAR}')
for YEAR in tqdm(range(START_YEAR,END_YEAR)):
    sentences = utils_train.SentIterator(ROOT,date_range=(YEAR,YEAR),processed_path=SENT_OUTPUT,tokenized=False,n_jobs=WORKERS)
    #print(sentences._date_range)
    sentences.prepareLines()
