import sys
sys.path.append('../')
from utils import utils_train,bias_utils
from gensim.models.word2vec import Word2Vec 
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SIZE = 300
WINDOW = 20
MIN_COUNT = 10
WORKERS = 16
EPOCH = 4
SEED = 42

df = pd.read_excel("../../resources/Lijst_kranten_final.xlsx",sheet_name="Sheet1")
PPN_NAME = "Katholiek"
PPN_FIELD = 'Politek'
PPN = df[(df[PPN_FIELD]==PPN_NAME) & (~df["PPN"].isnull())].PPN


START_YEAR = 1890
END_YEAR = 1900
ROOT = "kb/data/ResearchDrive"
SENT_OUTPUT = "/kbdata/Processed/Sentences"
MODEL_OUTPUT = "/kbdata/Processed/Models/{}-{}-{}.w2v.model".format(START_YEAR,END_YEAR,PPN_NAME)

sentences = utils_train.SentIterator(ROOT,date_range=(START_YEAR,END_YEAR),processed_path=SENT_OUTPUT,tokenized=False,n_jobs=8)
#sentences.prepareLines()

model = Word2Vec(size=SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, seed=SEED)
model.build_vocab(sentences=sentences)
total_examples = model.corpus_count
print(total_examples)

model.train(sentences=sentences, total_examples=total_examples, epochs=EPOCH)

model.save(MODEL_OUTPUT)
