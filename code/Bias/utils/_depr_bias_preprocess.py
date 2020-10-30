import sys
sys.path.append('..')
from utils import *
from glob import glob
import pandas as pd



START_YEAR = 1850
END_YEAR = 1860
ROOT = "../../../Data/ResearchDrive"
IDENTIFIERS_PATH = "../../../Data/Identifiers/Identifiers_18*.csv"
MODEL_PATH = "../../../Processed/Models/{}-{}.w2v.model".format(START_YEAR,END_YEAR)
PROCESSED = "../../../Processed/Sentences"
OUTPUT = "../../../Processed"


sentences = SentIterator(ROOT,date_range=(START_YEAR,END_YEAR),processed_path=PROCESSED,tokenized=False,n_jobs=-1)

filtered_sents_path = sentences.filter_lines('(?:vrouw*|moeder*)')
filtered_sents_lines = open(filtered_sents_path,'r').read().split('\n\n')
sent_df = pd.DataFrame([s.split('<SEP>') for s in filtered_sents_lines],columns=['doc_id','text'])

df = pd.concat([pd.read_csv(f,sep=';',index_col=0) for f in glob(IDENTIFIERS_PATH)],axis=0, sort=True)    
df['doc_id'] = df.identifier.apply(get_doc_id)
#print(np.sum(df.doc_id=='NaN'))
df_merged = sent_df.merge(df,how='left',right_on='doc_id',left_on='doc_id')
batches = pd.DataFrame(
			df_merged.groupby(['date','ppn']).agg(
								{"text":'<SEP>'.join,
								"doc_id":'<SEP>'.join}
                                    	))

batches.to_csv('{}/{}-{}_batched.csv'.format(PROCESSED,START_YEAR,END_YEAR))