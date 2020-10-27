# Lexicon Expansion Instruction

The Lexicon Expansion code allows you to create word list by systematically navigating embedding spaces. All code is available via the `LexiconExpansion.ipynb` Make sure that you have installed `ipyannotate` before annotation.

Lexicon Expansion contains three different methods for navigating word embeddings, which are explained in more detail below.

- Unidirectional Expansion
- Contrastive Expansion
- Expansion with Active Learning

Please consult the notebook for more information and instruction. The README only gives a consice overview of the expansion procedure and results.

## Unidirectional Expansion

This method allows you to explore the region around a couple of selected seed word. For example, in the notebook we selected the words "vrouw" (woman) and "moeder" (mother) to start querying for female words in the vector space.

```python
core = {'vrouw','moeder'} # select seed words
core_init = average_vector(core.copy(),model) # average the vector representation of the selected seed words
```

After specifying the seed words, you need to select a **sampling strategy** which defines how we will navigate the vector space. Here we use the simple `average` procedure, which simply looks at the words currently in the lexicon `L`, and explores the area surroundig `avg(v(L))` the avarage of the vector representation of the word in `L`.