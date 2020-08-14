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
MIN_COUNT = 10
WORKERS = 8
EPOCH = 4
SEED = 42

# training data parameters
TRAIN_START = 1870
TRAIN_END = 1909
TRAIN_STEP = 10
TRAIN_WINDOW = 20

# contextual paramaters
FACETS =  "Politek" # "Verspreidingsgebied" |  "Provincie" | "Politek"

meta_df = pd.read_excel(METADATA_PATH,sheet_name="Sheet1",index_col=0)
meta_df["PPN"] = meta_df["PPN"].astype(str)

# finetuning parameters

BASE_MODEL = ""


for FACET,PPN in meta_df.groupby(FACETS)['PPN'].apply(list).items():
    print(FACET, PPN)
    for START_YEAR in range(TRAIN_START,TRAIN_END,TRAIN_STEP):
    
        print(f'Training model--from {START_YEAR} to {START_YEAR+TRAIN_WINDOW}--{FACET}');print(PPN);
        sentences = utils_train.SentIterator(ROOT,date_range=(START_YEAR,START_YEAR+TRAIN_WINDOW),
                                            processed_path=SENT_OUTPUT,
                                            ppn=PPN,tokenized=False,
                                            n_jobs=WORKERS)
        print('Initialing model and vocabulary')
        model = Word2Vec.load("/kbdata/Processed/Models/BaseModel-1800-1909.w2v.model")
        #model.build_vocab(sentences=sentences)
        total_examples = model.corpus_count#len([i for i in sentences.__iter__()])
        print(f"\n-------\nTotal number examples = {total_examples}\n-------\n")
        if not total_examples: continue
        model.train(sentences=sentences, total_examples=total_examples, epochs=EPOCH,) # total_examples=total_examples,
        MODEL_OUTPUT = "/kbdata/Processed/FT-Models/FT-{}-{}-{}.w2v.model".format(START_YEAR,START_YEAR + TRAIN_WINDOW,FACET.replace('/','_'))
        model.save(MODEL_OUTPUT)
        print(f'Saved model to {MODEL_OUTPUT}')
