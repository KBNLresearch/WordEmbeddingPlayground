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
from typing import Union

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
years_included = lambda x: set(irange(*tuple(map(int,re.findall('1[8-9][0-9]{2}', x)[-2:]))))

def irange(start,stop):
    """inclusive range: includes the stop element"""
    return range(start,stop+1)

def preprocess_sent(sent: str,sent_id:str='',tokenized: bool=True) -> Union[list,str]: # check if setting sent_id to None effects anything?
    """preprocessing function for formatting raw text before training word2vec
    # Credits: Kasra Hosseini and Kaspar Beelen
    Arguments:
        sent (string): input sentence
        sent_id (string): idx of the inpute sentence
        tokenized (boolean): if True then return the string as a list of tokens
    Returns:

    """
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

def read_doc(zipdoc,path:str) -> tuple:
    """reads docuemnts from zip archive
    Arguments:
        zipdoc (element of ziparchive object?): ??
        path (str) : file path
    """
    # extract file idx from file path
    file_id = path.split('/')[-1].rstrip('.xml')
    # extract  ppn from path
    ppn = path.split('/')[-2]
    # construct docid by merging ppn and file idx
    doc_id = f'{ppn}<SEP>{file_id}'
    # try reading content from XML encoded file
    try:
        text = etree.tostring(
            etree.parse(
                BytesIO(zipdoc)
                    ), method="text",encoding="unicode"
            ) 
    except Exception as e:
        print(f'Error "{e}" for "{doc_id}"')
        text = ''
    # return text with the document idx
    return (text,doc_id)



class SentIterator(object):

    def __init__(self,
                root: str,
                date_range: tuple=(1800,2000),
                sample_docs: int=-1,
                tokenized:bool=True,
                processed_path: str='',
                ppn: Union[None,str]=None,
                n_jobs: int=-1):

        """Iterates over archive
        Arguments:
            root (string): folder where all the zip files are located
            date_range (tuple): tuple with first and last year on which language model is trained
                                Important: data range is inclusive (includes start and end year)
            sample_docs (integer): use first _n_ documents in each .zip file
            tokenized (boolean): defines whether the preprocessing includes tokenization
            processed_path (string): path to line by line processed data
            ppn (string): newspaper ppn
            n_jobs (integer): number of parallell processes to run
        """
        self.root = root
        # get path of all zip files
        self._zip_files = glob('{}/*.zip'.format(self.root))
        # map path of zip files to a year range encoded as a tuple
        self._path2year = {f:years_included(f) for f in self._zip_files }
        # define year range (inclding the last year)
        self._date_range = irange(*date_range)
        # for testing purposes only use first n years
        self._sample_docs = sample_docs
        self._tokenized = tokenized
        self._n_jobs = n_jobs
        # count the number of elements (documents or sentences)
        self.count = None
        self.ppn = ppn
        self.processed_path = processed_path
        

    def _select_zip_by_date_range(self):
        """Select zip files based on date range.
        Returns:
            set of path to zip files relevant for the selected data range
        """
        return set(k for k,v in self._path2year.items() if set(self._date_range).intersection(v))
    
    
    
    def _processZipParallel(self):
        """Iterate over files: select all files within a specific date range.
        Returns:
            a string separated by a <SEP> token.
        """
        # select relevant zip files
        selected = self._select_zip_by_date_range()
        print(selected)
        self.count = 0
        for file in selected: 
            with ZipFile(file, 'r') as zipdata:
                
                # iterate over path to XML files in zip files
                # if the first sequenace of four integers, surrounded by '/' is in the set of included years (defined by _date_range)
                #  ## Need to check if this works
                article_text = [(zipdata.read(f),f) for f in zipdata.namelist() 
                                        if f.endswith(".xml") and int(re.findall(r"/([0-9]{4})/",'/' + f)[0]) 
                                            in set(self._date_range)]
                
                # read the XML files 
                article_text = Parallel(n_jobs=self._n_jobs)(delayed(read_doc)(zipdoc,f) for (zipdoc,f) in article_text)
                
                #print(len(article_text))
                # process documents
                for sent,doc_id in Parallel(n_jobs=self._n_jobs)(delayed(preprocess_sent)(at,doc_id,tokenized=False) 
                                    for at,doc_id in article_text):
                    self.count+=1
                    yield doc_id+"<SEP>"+sent

    #def _processZip(self):
    #    """Iterate over files: select all files within a specific date range
    #    """
    #    selected = self._select_zip_by_date_range()
    #    print(selected)
    #    self.count = 0
    #    for file in selected: 
    #        with ZipFile(file, 'r') as zipdata:
    #            
    #            article_text = [(zipdata.read(f),f) for f in zipdata.namelist() 
    #                                    if f.endswith(".xml") and int(re.findall(r"/([0-9]{4})/",'/' + f)[0]) 
    #                                        in set(self._date_range)]
    #            
    #            article_text = [read_doc(zipdoc,f) for zipdoc,f in article_text]
    #            
    #            #print(len(article_text))
    #            for at,doc_id in article_text:
    #                sent,doc_id = preprocess_sent(at,doc_id,tokenized=False)
    #                self.count+=1
    #                yield doc_id+"<SEP>"+sent
       
        
    def __len__(self):
        """Number of elements processed
        Returns:
            self.count (integer) if object applied to the ziparchives
        """
        if not self.count:
            return 'Iterate over corpus first'
        else:
            return self.count
    
    def filter_lines(self,regex: str,name: str='filtered'):
        """filter lines whose content match a give regular expression
        Arguments:
            regex (regular expression): regular expression used find words in the content of an article
            name (string): extension that mark the seperated content
        Returns:
            path (string) to filtered lines
        """
        # compile regular expression
        pattern = re.compile(regex)
        # path to input sentences
        in_sents = "{}/{}-{}.txt".format(self.processed_path,self._date_range[0],self._date_range[-1])
        # where to store the sentences
        out_sents = "{}/{}-{}_{}.txt".format(self.processed_path,self._date_range[0],self._date_range[-1],name)
        with open(in_sents,'r') as in_lines:
            with open(out_sents,'w') as out_lines:
                for line in in_lines:
                    if pattern.findall(line):
                        out_lines.write(line+'\n')
        return out_sents

                                
    def prepareLines(self):
        """save sentences for selected date range in a txt file
        Check first if file doesn't alread exist.
        """
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
                    ppn,file_id,tokens = line.split('<SEP>')
                    if self.ppn:
                        if ppn in self.ppn:
                            yield tokens.split()
                    else:
                        yield tokens.split()
        
