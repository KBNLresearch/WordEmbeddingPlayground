import sys
sys.path.append('../')
from utils import utils_train,bias_utils
from gensim.models.word2vec import Word2Vec 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SIZE = 300
WINDOW = 20
MIN_COUNT = 10
WORKERS = 16
EPOCH = 4
SEED = 42

START_YEAR = 1850
END_YEAR = 1860
ROOT = "/data/ResearchDrive"
SENT_OUTPUT = "/data/Processed/Sentences"
MODEL_OUTPUT = "/data/Processed/Models/{}-{}.w2v.model".format(START_YEAR,END_YEAR)

sentences = utils_train.SentIterator(ROOT,date_range=(START_YEAR,END_YEAR),processed_path=SENT_OUTPUT,tokenized=False,n_jobs=8)
#sentences.prepareLines()

model = Word2Vec(size=SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, seed=SEED)
model.build_vocab(sentences=sentences)
total_examples = model.corpus_count
print(total_examples)

model.train(sentences=sentences, total_examples=total_examples, epochs=EPOCH)

model.save(MODEL_OUTPUT)
