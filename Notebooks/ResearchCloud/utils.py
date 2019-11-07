from lxml import etree
from zipfile import ZipFile
from io import StringIO,BytesIO
import unidecode
from glob import glob
import nltk.data
import re

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
years_included = lambda x: set(irange(*tuple(map(int,re.findall('1[8-9][0-9]{2}', x)[-2:]))))
#def process_sent(sent,tokenized=True):  
#    sent = re.sub(r"([.,?!:;'\"])",r" \1 ",sent) # seperate punctuation from text
#    sent = re.sub(r"[^A-Za-z\s]",r"",sent) # remove all non-alphatbetical characters
#    sent = ' '.join(sent.lower().split()) # remove superfluous whitespaces
    
#    if tokenized:
#        return sent.split()
#    return sent

def irange(start,stop):
    return range(start,stop+1)

def preprocess_sent(sent,tokenized=True):
    # Credits: Kasra Hosseini and Kaspar Beelen
    # --- replace .- and . in the middle of the word
    sent = re.sub(r'(?<=\w)(\.-)(?=\w)', '-', sent)
    sent = re.sub(r'(?<=\w)(\.)(?=\w)', '', sent)
    # --- remove accent
    sent = unidecode.unidecode(sent)
    # --- remove 2 or more .
    sent = re.sub(r'[.]{2,}', '.', sent)
    # --- add a space before and after a list of punctuations
    sent = re.sub(r"([.,!?:;\"\'])", r" \1 ", sent)
    # --- remove everything except:
    sent = re.sub(r"([^a-zA-Z\-.:;,!?\d+]+)", r" ", sent)
    # --- replace numbers with <NUM>
    sent = re.sub(r'\b\d+\b', '<NUM>', sent)
    sent = re.sub(r'--', '', sent)
    # --- normalize white spaces
    sent = re.sub(r'\s+', ' ', sent)
    # --- lowercase
    sent = sent.lower()

    if tokenized:
        return sent.split()
    return sent

def sent_split(article_text):
    tree = etree.parse(BytesIO(article_text))
    text = etree.tostring(tree,method="text",encoding="unicode")
    global sent_detector
    sents = sent_detector.tokenize(text)
    return sents

class SentIterator(object):

    def __init__(self,root,date_range=(1800,2000),sample_docs=None,tokenized=True):
        """Iterates over archive
        Arguments:
            root (string): folder where all the zip files are located
            date_range (tuple): tuple with first and last year on which language model is trained
                                Important: data range is inclusive (includes start and end year)
            sample_docs (int): use first _n_ documents in each .zip file
            tokenized (boolean): defines whether the preprocessing includes tokenization
        """
        self.root = root
        self._zip_files = glob(f'{self.root}/*.zip')
        self._path2year = {f:years_included(f) for f in self._zip_files }
        self._date_range = irange(*date_range)
        self._sample_docs = sample_docs
        self._tokenized = tokenized
        if self._sample_docs  is None:
            self._sample_docs = -1


    def _select_zip_by_date_range(self):
        """Select zip files based on date range
        """
        return set(k for k,v in self._path2year.items() if set(self._date_range).intersection(v))
        
    def __iter__(self):
        """Iterate over files: select all files within a specific date range
        """
        selected = self._select_zip_by_date_range()
        
        for file in selected: 
            with ZipFile(file, 'r') as zipdata:
                article_text = [f for f in zipdata.namelist() 
                                    if f.endswith(".xml") and int(re.findall(r"/([0-9]{4})/",'/' + f)[0]) 
                                        in set(self._date_range)]

                for at in article_text:
                    for sent in sent_split(zipdata.read(at)):
                        yield preprocess_sent(sent,self._tokenized)


#class SentIterator(object):
#    def __init__(self,location,sample_docs=None,tokenized=True):
#        self.location = location
#        self.sample_docs = sample_docs
#        self.tokenized = tokenized
#        if sample_docs is None:
#            self.sample_docs = -1
#        
#    def __iter__(self):
#        with ZipFile(self.location, 'r') as zipdata:
#            article_text = [f for f in zipdata.namelist() if f.endswith("articletext.xml")][:self.sample_docs]
#            for at in article_text:
#                for sent in sent_split(zipdata.read(at)):
#                    yield preprocess_sent(sent,self.tokenized)
        
    
