<p align="center">
    <a href="https://github.com/Living-with-machines/DeezyMatch/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <br/>
</p>



# Word embedding playground

Code created during KB Research in Residence project "Why girls smile and boys don't cry". This repository provides tools for **training and fine-tuning word embedding models** (Word2Vec and FastText) on a selected subset of Dutch Newspapers available in Delpher. 

It also comes with various functions to explore the trained embeddings. **Lexicon expansion**, allows you to "travel through a vector space" and interactively create a lexicon of conceptually related words in the process. In the **Bias** folder, you find various tools for analysing bias over time and other dimensions such as political leaning and place.

- [Training WEM Models](#training-models)
- [Lexicon Expansion](#explore-word-embedings)
- [Analyse bias](#analyse-bias)



## Training models

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

Next, define the model hyperparameters. More information can be found in the [gensim documentation](https://radimrehurek.com/gensim/models/word2vec.html).

```python
# model hyperparameter
SIZE = 300 # size of word vector
WINDOW = 20 # window size 
MIN_COUNT = 10 # remove words that appear less than n time
WORKERS = 8 # use n number of cores
EPOCH = 4 # train for n epochs
SEED = 42
```


After setting the model hyperparameters, define the training routine, which moves as a sliding window over a selected date range..
The first step entials setting a time range, i.e. the period for which to generate Word2Vec models. `TRAIN_START` and `TRAIN_END` are the first and last year. `TRAIN_WINDOW` refers to number of years included in each step, `TRAIN_STEP` sets the step size. For the parameters selected below, the train routine will start at 1840, train a model for the period 1840-1860, and then move the window with five years to 1845-1865 etc.

```python
# training data hyperparameters
TRAIN_START = 1840
TRAIN_END = 1909
TRAIN_STEP = 5
TRAIN_WINDOW = 20
```

The last hyperparamater is `FACETS`. This allows you to add another dimension to the training routine (besides time). These facets refer to colums in the metadata file, and you can easily change the cell values, or simply add another column, depending on your research question. The standard options are: 

- Politek: Political leaning of the newspaper;
- Verspreidingsgebied: Indicates whether this is a national or regional newspaper;
- Provincie: The province in which the newspaper is published.


## Explore word embedings

Lexicon expansion provides some functionality to interactively explore (i.e. travel through) word vector spaces. The screencast gives a quick overview of the process, but more function are availble in this [Notebook](./code/LexiconExpansion/LexiconExpansion.ipynb). To read more about the interactive lexicon expansion go [here](./code/LexiconExpansion/README.md) for more information.



![Annotation Procedure](./code/LexiconExpansion/img/annotation.gif)


The different steps covered in the screencast are:
- Select seed words: in this case we chose "vrouw" and "vrouwen" as the seed query
- Select Sampling strategy: : `"average"` selects the simplest method which samples the closest neighbours to the query vector, other option are `"query_tokens"`, `"entropy"` and `"distance"`.
- Annotate: `Core` words will be added to lexicon and influence constructing the query vector. `Peripheral` words will be saved but don't influence the sampling. In this scenario I added unambiguously "female" words to the `Core` lexicon and OCR variants to the `Peripheral` word list. These words are saved, in case they are need later. I ignored For all other words (`Ignore`)
- Update lexicon with annotations: the next code blocks, updated the lexicon with the annotations. You can now go back to the previous step to harvest more words (but don't forget to save afterwards!) or you can plot the results.
- Plot the lexicon and surround words: the visualisation plots all the selected words on a 2D plane. The re
- Save lexicon: save the results of the annotation process for later use.

The expansion normally consists of multiple iterations. The figure below plots the of multiple annotation rounds that aimed to harvest different words referring to women in newspapers.

## Analyse bias 

Inspect bias over different Word2Vec model. See the bias [README](./code/Bias/README.md) for more information.
