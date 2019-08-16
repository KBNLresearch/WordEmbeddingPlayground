# KB-RiR
Code for KB RiR

## Updates 19 - 07

Created Notebooks for Training Word/Character Embedding

- Fasttext.ipynb
- Word2Vec.ipynb
- utils.py

Created a Notebook for Lexicon Expansion

- SemAxis - Lexicon Expansion.ipynb

Experimented with Flair

- Flair.ipynb

## Updates 26 - 07

- Skype with Melvin on possible collaboration

- Theory: Read Chapter 10 "Pre-trained Word Representation", in "Neural Network Methods for Natural Language Processing" by Yoav Goldberg

- Focus on Deep Contextual Word Embedding (FLAIR/BERT/ELMo)
	- Experiment with existing APIs
			- FLAIR: Tutorial see ./Notebooks/FLAIR/FLAIR Tutorial.ipynb
			- BERT as a service 
			
	- Optimizing/Fine-tuning models:
			- Optimize FLAIR embedding with KB newspaper data: ./Notebooks/FLAIR/FineTuneModel.ipynb

## Updates 02 - 08

Holiday.

## Updates 09 - 08

Theory: 
     - Stanford [CS224N  Lecture 13](https://www.youtube.com/watch?v=S-CspeZ8FHc&feature=youtu.be) – Contextual Word Embeddings
    - Blog post explaining the [BERT Model](http://jalammar.github.io/illustrated-bert/)
    
Focussed on interactive lexicon expansion, i.e. generating a lexicon based on only a minimal number of seed words. Implemented two approaches:

	- [SemAxis](https://github.com/kasparvonbeelen/KB-RiR/blob/master/Notebooks/LexiconExpansion/Interactive-Lexicon-Expansion-SemAxis.ipynb), generates a male-female dimension based on user input. Resulted in a list of 636 "female" words, which can be found [here](https://github.com/kasparvonbeelen/KB-RiR/blob/master/Notebooks/LexiconExpansion/result/female_lexicon.txt). 
			- To do: ensure maximum recall (i.e. are we not missing relevant axis); avoid being stuck in one latent dimension (i.e. female names)
				
- [Active Learning](https://github.com/kasparvonbeelen/KB-RiR/blob/master/Notebooks/LexiconExpansion/Interactive-Lexicon-Expansion-ActiveLearning.ipynb): Uses an active learning pipeline to train a Linear Support Vector Machines that trains a model to classify words as "male" or "female". The confidence for each word is taken as the "gender score"
			- To do: works less than SemAxis. Create at test set to evaluate performance, and determine the optimal hyperparameters of the model.
