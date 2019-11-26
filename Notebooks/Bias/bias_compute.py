
from gensim.models.word2vec import Word2Vec 
import pandas as pd
from tqdm import tqdm
import pickle
import sys
sys.path.append('../utils')
from bias_utils import *
from utils_parallel import *

START_YEAR = 1850
END_YEAR = 1860

MODEL_PATH = "../../../Processed/Models/{}-{}.w2v.model".format(START_YEAR,END_YEAR)
PROCESSED_PATH = "../../../Processed/Sentences"
OUTPUT = '../../../Processed'

# Hyperparameters for training
EPOCH = 4
# Important: add learning rate!!






# for similation we now use the nearest neighbours as the lexicon for male and female words
# replace this with a function later that loads the words from a pickle files as in
#with open('lexicon.pckl','rb') in in_pickle:
#	p1,p2,target = pickle.load(in_pickle)
# load model
model = Word2Vec.load(MODEL_PATH)
p1 = [w for w,v in model.wv.most_similar('vrouw',topn=20)] + ['vrouw']
p2 = [w for w,v in model.wv.most_similar('man',topn=20)] + ['man']
# target is the word child
target = [w for w,v in model.wv.most_similar('kind',topn=20)] + ['kind']


batches = pd.read_csv('{}/{}-{}_batched.csv'.format(PROCESSED_PATH,START_YEAR,END_YEAR),
                        index_col=0,
                        chunksize=1000)

update_sents = [preprocess_sent(t.text,(t.name,t.ppn,t.doc_id))
                    for chunk in batches
                        for i,t in chunk.iterrows()] #  !!!!!! test test change back to generator later !!!!!! 

# compute the bias scores of all sentences
scores = Parallel(n_jobs=-1)(
                delayed(compare_bias)(sent,meta,p1,p2,target,MODEL_PATH) 
                    for sent,meta in tqdm(update_sents[:100])) # !!!! test test remove later!!!!!

with open('{}/{}-{}_biass_cores.pckl'.format(OUTPUT,START_YEAR,END_YEAR),'wb') as out_pickle:
    pickle.dump(scores,out_pickle)
