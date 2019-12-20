import sys
sys.path.append('../')
from utils import utils_train
from gensim.models.word2vec import Word2Vec 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SIZE = 300
WINDOW = 20
MIN_COUNT = 10
WORKERS = 16
EPOCH = 4
SEED = 42

START_YEAR = 1830
END_YEAR = 1840
ROOT = "/data/ResearchDrive"
SENT_OUTPUT = "/data/Processed/Sentences"

for YEAR in range(START_YEAR,END_YEAR):
    print(f'Writing Sentences for year {YEAR} to {SENT_OUTPUT}')
    sentences = utils_train.SentIterator(ROOT,date_range=(YEAR,YEAR),processed_path=SENT_OUTPUT,tokenized=False,n_jobs=8)
    sentences.prepareLines()
