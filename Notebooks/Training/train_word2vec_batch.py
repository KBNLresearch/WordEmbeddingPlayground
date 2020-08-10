import sys
sys.path.append('../')
from utils import utils_train,bias_utils
from gensim.models.word2vec import Word2Vec
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# paths
ROOT = "/kbdata/ResearchDrive"
SENT_OUTPUT = "/kbdata/Processed/Sentences"
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
TRAIN_END = 1890
TRAIN_STEP = 10
TRAIN_WINDOW = 20

# contextual paramaters
FACETS = "Verspreidingsgebied"
#FACETS = "Provincie"
meta_df = pd.read_excel(METADATA_PATH,sheet_name="Sheet1")


for FACET,PPN in meta_df.groupby(FACETS)['PPN'].apply(list).items():
    print(FACET, PPN)
    for START_YEAR in range(TRAIN_START,TRAIN_END,TRAIN_STEP):
    
        print(f'Training model--from {START_YEAR} to {START_YEAR+TRAIN_WINDOW}--{FACET}');print(PPN);
        sentences = utils_train.SentIterator(ROOT,date_range=(START_YEAR,START_YEAR+TRAIN_WINDOW),processed_path=SENT_OUTPUT,ppn=PPN,tokenized=False,n_jobs=WORKERS)
        
        model = Word2Vec(size=SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, seed=SEED)
        model.build_vocab(sentences=sentences)
    
        total_examples = model.corpus_count
        print(total_examples)
        if not total_examples: continue
        model.train(sentences=sentences, total_examples=total_examples, epochs=EPOCH)
        MODEL_OUTPUT = "/kbdata/Processed/Models/{}-{}-{}.w2v.model".format(START_YEAR,START_YEAR + TRAIN_WINDOW,FACET.replace('/','_'))
        model.save(MODEL_OUTPUT)
        print(f'Saved model to {MODEL_OUTPUT}')
