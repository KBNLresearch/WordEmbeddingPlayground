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