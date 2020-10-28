import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import cosine
from pathlib import Path
from tqdm.notebook import tqdm
from collections import defaultdict

def select_model_by_facet_value(facet_value,root=Path("/kbdata/Processed/Models/")):
    models = root.glob(f"*-{facet_value}.w2v.model")
    
    out = {}
    for m in models:
        
        start = m.stem.lstrip("FT-").split('-')[0]
        out[int(start)] = m
        
    return out

cosine_sim = lambda v1,v2: 1 - cosine(v1,v2) 
average_vector = lambda words,model : np.mean([model.wv.__getitem__(w) for w in words if model.wv.__contains__(w)],axis=0)

def inspect_bias(p1,p2,target,model,metric=cosine_sim):
    p1 = [p for p in p1 if  model.wv.__contains__(p)]
    p2 = [p for p in p2 if  model.wv.__contains__(p)]
    target = [t for t in target if  model.wv.__contains__(t)]
    p1_scores, p2_scores = defaultdict(float),defaultdict(float)
    for t in target:
        for p in p1:
            p1_scores[(p,target[0])] += metric(model.wv.__getitem__(p),model.wv.__getitem__(t))
        for p in p2:
            p2_scores[(p,target[0])] += metric(model.wv.__getitem__(p),model.wv.__getitem__(t))
    return (p1_scores,p2_scores)

def compute_bias_average_vector(p1,p2,target,model,metric=cosine_sim):
    """computes bias given two poles and and a target word list
    bias is the average distance of each target word to the poles
    Arguments:
        p1 (list): list of pole words
        p2 (list): lost of pole words
        target (list): list of target words
        metric (funtion): distance function, either cosine or euclidean
    Returns:
        bias (float): the bias score of the target to each of the poles
    """
    av_v1 = average_vector(p1,model); av_v2 = average_vector(p2,model)
    return np.mean([metric(av_v1,model.wv.__getitem__(w)) - \
                      metric(av_v2,model.wv.__getitem__(w)) for w in target 
                           if w in model.wv])

def compute_bias(p1,p2,target,model,metric=cosine_sim):
    p1 = [p for p in p1 if  model.wv.__contains__(p)]
    p2 = [p for p in p2 if  model.wv.__contains__(p)]
    target = [t for t in target if  model.wv.__contains__(t)]
    p1_scores, p2_scores = [],[]
    for t in target:
        for p in p1:
            p1_scores.append(metric(model.wv.__getitem__(p),model.wv.__getitem__(t)))
        for p in p2:
            p2_scores.append(metric(model.wv.__getitem__(p),model.wv.__getitem__(t)))
    return np.mean(p1_scores) - np.mean(p2_scores)