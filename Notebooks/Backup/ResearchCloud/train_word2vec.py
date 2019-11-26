from gensim.models.word2vec import Word2Vec 
import logging
import argparse
from utils import *
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Script for training Word2Vec model for a specified year range.')
parser.add_argument('-s','--start_year', type=int, help='start year',default=1800)
parser.add_argument('-e','--end_year', type=int, help='end year',default=2000)
parser.add_argument('-p','--path', type=str, help='path to the newspaper data',required=True)
parser.add_argument('-o','--output', type=str, help='output folder',default='.')
parser.add_argument('--size', type=int, help='dimensionality of the vector space',default=100)
parser.add_argument('--window', type=int, help='window size',default=5)
parser.add_argument('--min', type=int, help='minimum word count',default=10)

args = parser.parse_args()
print(args)


sentences = SentIterator(args.path,date_range=(args.start_year,args.end_year))
model = Word2Vec(size=args.size, window=args.window, min_count=args.min)
model.build_vocab(sentences=sentences)
total_examples = model.corpus_count
model.train(sentences=sentences, total_examples=total_examples, epochs=5)
model.save(args.output)