#!/usr/bin/env python
__author__ = "Kaspar Beelen"
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
MIN_COUNT = 100
WORKERS = 8
EPOCH = 4
SEED = 42

# training data parameters
TRAIN_START = 1800
TRAIN_END = 1909


print(f'Training model--from {TRAIN_START} to {TRAIN_END}')
sentences = utils_train.SentIterator(ROOT,date_range=(TRAIN_START,TRAIN_END),
                                            processed_path=SENT_OUTPUT,
                                            tokenized=False,
                                            n_jobs=WORKERS)
print('Initialing model and vocabulary')
model = Word2Vec(size=SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, seed=SEED)
model.build_vocab(sentences=sentences)
    
total_examples = model.corpus_count
print(f"\n-------\nTotal number examples = {total_examples}\n-------\n")
     
model.train(sentences=sentences, total_examples=total_examples, epochs=EPOCH)
MODEL_OUTPUT = f"/kbdata/Processed/Models/BaseModel-{TRAIN_START}-{TRAIN_END}.w2v.model"
model.save(MODEL_OUTPUT)
print(f'Saved model to {MODEL_OUTPUT}')
