from lxml import etree
from zipfile import ZipFile
from io import StringIO,BytesIO
import unidecode
from joblib import Parallel, delayed
from tqdm import tqdm
from glob import glob
import nltk.data
import os
import re

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
years_included = lambda x: set(irange(*tuple(map(int,re.findall('1[8-9][0-9]{2}', x)[-2:]))))

def irange(start,stop):
    return range(start,stop+1)

def preprocess_sent(sent,sent_id=None,tokenized=True): # check if setting sent_id to None effects anything?
    # Credits: Kasra Hosseini and Kaspar Beelen
    # --- replace .- and . in the middle of the word
    sent = re.sub(r'(?<=\w)(\.-)(?=\w)', '-', sent)
    sent = re.sub(r'(?<=\w)(\.)(?=\w)', '', sent)
    # --- remove accent
    #sent = unidecode.unidecode(sent)
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
        return (sent.split(),sent_id)
    return sent,sent_id

def read_doc(zipdoc,path):
    doc_id = path.split('/')[-1].rstrip('.xml')
    try:
        text = etree.tostring(
            etree.parse(
                BytesIO(zipdoc)
                    ), method="text",encoding="unicode"
            ) 
    except Exception as e:
        print(f'Error "{e}" for "{doc_id}"')
        text = ''
    return (text,doc_id)



class SentIterator(object):

    def __init__(self,root,date_range=(1800,2000),sample_docs=None,tokenized=True,processed_path='',n_jobs=-1):
        """Iterates over archive
        Arguments:
            root (string): folder where all the zip files are located
            date_range (tuple): tuple with first and last year on which language model is trained
                                Important: data range is inclusive (includes start and end year)
            sample_docs (int): use first _n_ documents in each .zip file
            tokenized (boolean): defines whether the preprocessing includes tokenization
        """
        self.root = root
        self._zip_files = glob('{}/*.zip'.format(self.root))
        self._path2year = {f:years_included(f) for f in self._zip_files }
        self._date_range = irange(*date_range)
        self._sample_docs = sample_docs
        self._tokenized = tokenized
        self._n_jobs = n_jobs
        self.count = None
        self.processed_path = processed_path
        if self._sample_docs  is None:
            self._sample_docs = -1


    def _select_zip_by_date_range(self):
        """Select zip files based on date range
        """
        return set(k for k,v in self._path2year.items() if set(self._date_range).intersection(v))
    
    
    
    def _processZipParallel(self):
        """Iterate over files: select all files within a specific date range
        """
        selected = self._select_zip_by_date_range()
        print(selected)
        self.count = 0
        for file in selected: 
            with ZipFile(file, 'r') as zipdata:
                
                article_text = [(zipdata.read(f),f) for f in zipdata.namelist() 
                                        if f.endswith(".xml") and int(re.findall(r"/([0-9]{4})/",'/' + f)[0]) 
                                            in set(self._date_range)]
                
                article_text = Parallel(n_jobs=self._n_jobs)(delayed(read_doc)(zipdoc,f) for (zipdoc,f) in article_text)
                
                #print(len(article_text))
                for sent,doc_id in Parallel(n_jobs=self._n_jobs)(delayed(preprocess_sent)(at,doc_id,tokenized=False) 
                                    for at,doc_id in article_text):
                    self.count+=1
                    yield doc_id+"<SEP>"+sent

    def _processZip(self):
        """Iterate over files: select all files within a specific date range
        """
        selected = self._select_zip_by_date_range()
        print(selected)
        self.count = 0
        for file in selected: 
            with ZipFile(file, 'r') as zipdata:
                
                article_text = [(zipdata.read(f),f) for f in zipdata.namelist() 
                                        if f.endswith(".xml") and int(re.findall(r"/([0-9]{4})/",'/' + f)[0]) 
                                            in set(self._date_range)]
                
                article_text = [read_doc(zipdoc,f) for zipdoc,f in article_text]
                
                #print(len(article_text))
                for at,doc_id in article_text:
                    sent,doc_id = preprocess_sent(at,doc_id,tokenized=False)
                    self.count+=1
                    yield doc_id+"<SEP>"+sent
       
        
    def __len__(self):
        if not self.count:
            return 'Iterate over corpus first'
        else:
            return self.count
    
    def filter_lines(self,regex,name='filtered'):
        """filter lines whose content match a give regular expression
        Arguments:
            regex (regular expression): regular expression used find words in the content of an article
            name (string): extension that mark the seperated content
        """
        pattern = re.compile(regex)
        in_sents = "{}/{}-{}.txt".format(self.processed_path,self._date_range[0],self._date_range[-1])
        out_sents = "{}/{}-{}_{}.txt".format(self.processed_path,self._date_range[0],self._date_range[-1],name)
        with open(in_sents,'r') as in_lines:
            with open(out_sents,'w') as out_lines:
                for line in in_lines:
                    if pattern.findall(line):
                        out_lines.write(line+'\n')
        return out_sents

                                
    def prepareLines(self):
        out_sents = "{}/{}-{}.txt".format(self.processed_path,self._date_range[0],self._date_range[-1])
        if not os.path.isfile(out_sents): # change again later
            print('Processing zip files')
            with open(out_sents,'w') as out_file:
                 for s in self._processZip():
                    out_file.write(s + "\n")
        print('Zip files processed and stored in {}'.format(out_sents))
            
    def __iter__(self):
        """for a given year range, iterate over txt files
        and yield article content
        
        """
        for year in self._date_range:
            in_sents = "{}/{}-{}.txt".format(self.processed_path,year,year)
            with open(in_sents,'r') as in_lines:
                for line in in_lines:
                    doc_id,tokens = line.split('<SEP>')
                    yield tokens.split()
        
