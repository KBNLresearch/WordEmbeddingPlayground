import sys
sys.path.append('../')
from utils import utils_train,bias_utils
from gensim.models.word2vec import Word2Vec
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# paths
ROOT = "/data/ResearchDrive"
SENT_OUTPUT = "/data/Processed/Sentences"
METADATA_PATH = "../../resources/Lijst_kranten_final.xlsx"

# model hyperparameter
SIZE = 300
WINDOW = 20
MIN_COUNT = 10
WORKERS = 8
EPOCH = 4
SEED = 42

# training parameters
TRAIN_START = 1840
TRAIN_END = 1910
TRAIN_STEP = 20

# contextual paramaters
FACET = "Verspreidingsgebied"
meta_df = pd.read_excel(METADATA_PATH,sheet_name="Sheet1")


for START_YEAR,END_YEAR in range(TRAIN_STEP,TRAIN_END,TRAIN_STEP):
	for FACET,PPN in meta_df.groupby(FACET)['PPN'].apply(list).items(): 
		print(f'Training model--from {START_YEAR} to {END_YEAR}--{FACET}')
		sentences = utils_train.SentIterator(ROOT,date_range=(START_YEAR,END_YEAR),processed_path=SENT_OUTPUT,ppn=PPN,tokenized=False,n_jobs=WORKERS)
		sentences.prepareLines()
		