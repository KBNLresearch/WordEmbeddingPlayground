# Word embedding playground
Code created during KB Research in Residence project "Why girls smile and boys don't cry.". This repository provides tools for training and fine-tuning word embeddings (Word2Vec and FastText) on a selected subset of Dutch Newspapers available in Delpher. It also provides tools for exploring diachronic embeddings along the dimension of political leaning and place.

The code in this repository allows you to train word embedding models, explore their content and analyse bias.

## Training

These scripts provide function for training Word2Vec and FastText models, either individually or in batch. The first step requires proprecessing the zipped newspaper data, which is handled by the `prep_sentences_batch.py` script.

Before running the script adjust the hyperparameters listed below.


```python
ROOT = "/path/to/zip/files"
SENT_OUTPUT = "path/to/output/folder"

START_YEAR = 1850
END_YEAR = 1910

WORKERS = 8 # the number of cores used for preprocessing the data
```

Then run

```bash
python prep_sentences_batch.py
```

This will produce one large `.txt` file for each year, with one (processed) newspaper article per line. For inspecting, the exact procedure for processing the text files, please consult the `preprocess_sent` function in `utils_train.py`.

After preprocessing data, you can train a batch of embedding models on the newspapers. Below we show how to train Word2Vec models in batch, but the same procedure applies to creating FastText models. 

`train_word2vec_batch.py` is the main workhorse. Before running it, adjust the hyperparamters listed below. The `METADATA_PATH` refers to excel file in the `resources` folder, which provides information on the political leaning, publication and circulation of newspapers.

```python
# paths
ROOT = "/path/to/root/folder" # 
SENT_OUTPUT = "/path/to/processed/data"
METADATA_PATH = "../../resources/Lijst_kranten_final.xlsx"
```

Next, define the model hyperparameters. More information can be found in the gensim documentation.

```
# model hyperparameter
SIZE = 300 # size of word vector
WINDOW = 20 # window size 
MIN_COUNT = 10 # remove words that appear less than n time
WORKERS = 8 # use n number of cores
EPOCH = 4 # train for n epochs
SEED = 42
```