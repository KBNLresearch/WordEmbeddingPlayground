
from gensim.models.word2vec import Word2Vec 
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from glob import glob
import pickle
from bias_utils import *
from utils_parallel import *

START_YEAR = 1860
END_YEAR = 1870
ROOT = "/home/kaspar/ResearchDrive"
MODEL_PATH = "/home/kaspar/models/{}-{}.w2v.model".format(START_YEAR,END_YEAR)
OUTPUT = "/home/kaspar/processed"
# Hyperparameters for training
EPOCH = 4
# Important: add learning rate!!


daily_articles = pd.read_csv('../../../processed/{}_{}-daily.csv'.format(START_YEAR,END_YEAR),chunksize=100)

# load model
model = Word2Vec.load(MODEL_PATH)

# for similation we now use the nearest neighbours as the lexicon for male and female words
p1 = [w for w,v in model.wv.most_similar('vrouw',topn=20)] + ['vrouw']
p2 = [w for w,v in model.wv.most_similar('man',topn=20)] + ['man']
# target is the word child
target = [w for w,v in model.wv.most_similar('kind',topn=20)] + ['kind']

cosine_sim = lambda v1,v2: 1 - cosine(v1,v2) 
euclid_dist = lambda v1,v2: - np.linalg.norm(v1-v2,ord=2)
average_vector = lambda words,model : np.mean([model.wv.__getitem__(w) for w in words if model.wv.__contains__(w)],axis=0)



update_sents = (preprocess_sent(t.text,t.doc_id)
                    for chunk in daily_articles
                        for i,t in chunk.iterrows())

model_path ='../../../models/{0}-{1}.w2v.model'.format(START_YEAR,END_YEAR)
# compute the bias scores of all sentences
scores = Parallel(n_jobs=-1)(delayed(compare_bias)(i,sent,p1,p2,target,model_path) for i,sent in tqdm(enumerate(update_sents)))



with open('{}/biasbatch.pckl'.format(OUTPUT),'wb') as out_pickle:
    pickle.dump(scores,out_pickle)
