from lxml import etree
from zipfile import ZipFile
from io import StringIO,BytesIO
import nltk.data
import re

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')

def process_sent(sent,tokenized=True):  
    sent = re.sub(r"([.,?!:;'\"])",r" \1 ",sent) # seperate punctuation from text
    sent = re.sub(r"[^A-Za-z\s]",r"",sent) # remove all non-alphatbetical characters
    sent = ' '.join(sent.lower().split()) # remove superfluous whitespaces
    
    if tokenized:
        return sent.split()
    return sent

def process_article_text(article_text):
    tree = etree.parse(BytesIO(article_text))
    text = etree.tostring(tree,method="text",encoding="unicode")
    global sent_detector
    sents = sent_detector.tokenize(text)
    return sents

class SentIterator(object):
    def __init__(self,location,sample_docs=None,tokenized=True):
        self.location = location
        self.sample_docs = sample_docs
        self.tokenized = tokenized
        if sample_docs is None:
            self.sample_docs = -1
        
    def __iter__(self):
        with ZipFile(self.location, 'r') as zipdata:
            article_text = [f for f in zipdata.namelist() if f.endswith("articletext.xml")][:self.sample_docs]
            for at in article_text:
                for sent in process_article_text(zipdata.read(at)):
                    yield process_sent(sent,self.tokenized)
        
    
