
from gensim.models.word2vec import Word2Vec 
from scipy.spatial.distance import cosine
import numpy as np

cosine_sim = lambda v1,v2: 1 - cosine(v1,v2) 
euclid_dist = lambda v1,v2: - np.linalg.norm(v1-v2,ord=2)
average_vector = lambda words,model : np.mean([model.wv.__getitem__(w) for w in words if model.wv.__contains__(w)],axis=0)


def compare_bias(batch,meta_data,p1,p2,target,model_path,epoch=5):
    """function that compares the bias scores before and after updating the model weights.
    Arguments:
        i (int): row index # to do: improve here
        batch (list): list of strings that contains the document on which to retrain the model
        p1 (list): list of pole words
        p2 (list): lost of pole words
        target (list): list of target words
    Returns:
        a tuple with i, batch and difference in bias caused by updating the model
        
    """
    model = Word2Vec.load(model_path)
    model.train([batch],total_examples=len([batch]),epochs=epoch)
    orig_model = Word2Vec.load(model_path)
    return (*meta_data,compute_bias(p1,p2,target,model) - compute_bias(p1,p2,target,orig_model))

def compute_bias(p1,p2,target,model,metric=cosine_sim):
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


def get_doc_id(identifier):
    """reconstruct id from row in Identifier files
    these ids match one reported in the xml.
    """
    try:
        _,_,i,_,j,_ = identifier.split(":") 
        return '_'.join([i,j])
    except:
        return 'NaN'