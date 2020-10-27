import numpy as np
import pandas as pd
import random
import datetime
from collections import defaultdict
from collections import OrderedDict
from scipy.spatial.distance import cosine, euclidean
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.special import softmax





def expand_lexicon(core,model,method,args={}):
    """Core function for lexicon expansion. Defines the procedure 
    for generating words to create the average vector. 
    Essentially defines the area that will be explored, and the variability.
    """

    words = list(core)

    if not method:
        method = average_vector
    print('Using "{}" as sample method.'.format(method.__name__))
    sampled_words = method(words,model,**args)#,cutoff=sample_args.get('cutoff',-1))
    av_vec = average_vector(sampled_words,model)

    return dict(model.wv.similar_by_vector(av_vec,topn=100))


# return an average vector given a list of words and a model
average_vector = lambda words,model : np.mean([model.wv.__getitem__(w) 
                                               for w in words if model.wv.__contains__(w)],axis=0)
def average_all(words,model):
    return words

def query_tokens(words,model,tokens,merge=False):
    if merge:
        tokens+=words
    print(tokens)
    return tokens

def random_sample(words,model,cutoff=None): # cutoff=0.5 # ,**kwargs
    """resample from a wordlist
    randomize the sequence of the words, and selects sample of size n defined by the cutoff 
    Arguments:
        wordlist (list): list of primary lexicon terms
        cutoff (int or float): the sample size n
                if float it selects #words * n 
                if int it select the n first words
    Returns:
    """
    if isinstance(cutoff,float):
        cutoff = int(len(words)*cutoff)
    elif isinstance(cutoff,int):
        pass
    else:
        raise ValueError(f"Variable 'cutoff' has to be of type {type(0)} or {type(.0)}, got {type(cutoff)}")
    random.shuffle(words)
    print(cutoff)
    return words[:cutoff]


def entropy_sample(words,model,topn=2,init_vec=None,reverse=True):
    if not isinstance(init_vec,np.ndarray):
        init_vec = average_vector(words,model) 

    ranked = [n for n,v in sorted({w : entropy(softmax(init_vec), softmax(model.wv[w])) for w in words}.items(),
                                key = lambda x : x[1], reverse = reverse)][:topn]
    print('Selected terms = {}'.format(', '.join(ranked)))
    return ranked


def distance_sample(words,model,topn=2, method=cosine, init_vec=None, reverse=True):
    
    if not isinstance(init_vec,np.ndarray):
        init_vec = average_vector(words,model)

    ranked = [n for n,v in sorted({w : method(init_vec, model.wv[w]) for w in words}.items(),
                                key = lambda x : x[1], reverse = reverse)][:topn]
    print('Selected terms = {}'.format(', '.join(ranked)))
    return ranked


# contrastive expansion
def project_matrix_on_vec(vec_normalized, matrix_normalized):
    """
    Project list of vectors arranged in a matrix (row-wise) onto a vector (vec_normalized)
    Note: both vector and matrix should be normalized beforehand
    """
    proj_vecs = np.dot(matrix_normalized, vec_normalized)
    return proj_vecs

def select_proj_vecs(proj_vecs, embedding_model, show_negative=False, num_vec2show=20, verbose=True):
    """
    Select first num_vec2show from projected vectors, 
    These vectors can be selected on the either side of a vector
    
    Testing:
    #embedding_model.wv.index2word.index("machine")
    #matrix_normalized[5035, :][:10]
    #(embedding_model.wv.word_vec("machine")/np.linalg.norm(embedding_model.wv.word_vec("machine")))[:10]
    """
    if show_negative:
        found_vecs = np.array(embedding_model.wv.index2word)[np.argsort(proj_vecs)][:num_vec2show]
        found_vecs_mag = np.sort(proj_vecs)[:num_vec2show]
    
    else:
        found_vecs = np.array(embedding_model.wv.index2word)[np.argsort(proj_vecs)[::-1]][:num_vec2show]
        found_vecs_mag = np.sort(proj_vecs)[::-1][:num_vec2show]
    
    if verbose:
        print(15*" " + "count" + 4*" " + "score" + 3*" " + "word")
        print(40*"-")
        for i in range(len(found_vecs)):
            count_in_corpus = str(embedding_model.wv.vocab[found_vecs[i]].count)
            count_in_corpus = '{0: >20}'.format(count_in_corpus)
            print("%s|\t%.3f|\t%s" % (count_in_corpus, found_vecs_mag[i], found_vecs[i]))
    return found_vecs, found_vecs_mag

def contrastive_expansion(core,model,antipode=None,direction='core'):
    show_negative = {'core':False,'antipode':True}[direction]
    # read all vectors in embedding_model, normalize them
    matrix_normalized = model.wv.vectors/np.linalg.norm(model.wv.vectors, axis=1)[:, None]   
    # Average over vectors
    mean_v1 = np.mean([model.wv[w] for w in core], axis=0)
    mean_v2 = np.mean([model.wv[w] for w in antipode], axis=0)
    # defining a semantic axis (refer to equation 1 in the reference)
    semantic_axis = (mean_v1 - mean_v2)
    # normalize
    semantic_axis /= max(1e-10, np.linalg.norm(semantic_axis))
    # Project all vectors in embedding_model onto semantic_axis
    proj_vecs = project_matrix_on_vec(semantic_axis, matrix_normalized)

    found_vecs, found_vecs_mag = select_proj_vecs(proj_vecs, 
                                              model, 
                                              show_negative=show_negative, 
                                              verbose= False,
                                              num_vec2show=100)

    return dict(zip(found_vecs,found_vecs_mag))

#helper functions
def update_log(log,rounds,seen,core,peripheral,sampling_procedure):
    log[rounds]['timestamp'] = datetime.datetime.now()           
    log[rounds]['seen'] = seen.copy(); log[rounds]['core'] = core.copy(); log[rounds]['peripheral'] = peripheral.copy()
    log[rounds]['sample_method'] = sampling_procedure['method']
    log[rounds]['sample_args'] = sampling_procedure.get('args',{})
    return log

def obtain_negatives(positives,model):
    return [model.wv.most_similar(negative=p)[0][0] for p in positives]


def sort_scores(scores,topn=-1,ascending=False):
    """sort items of a dictionary
    Arguments:
        - scores (dict): dictionary that maps words to similarity scores
        - topn (int): number of items to return (-1 if to return all words)
    Returns:
        - sorted item list by score (n first)
    """
    return sorted(scores.items(),key = lambda x: x[1],reverse=not ascending)[:topn]

def topn_new(neighbours,seen,topn=10,reverse=True):
    """ensure that annotation candidates haven't been observed earlier
    Arguments:
        - neighbours (dict): list of neighbours of the query 
        - seen (list): words observed earlier
        - topn (int): number of words to return
    """
    neighbours = sorted(neighbours.items(),key=lambda x: x[1], reverse=reverse)
    c = 0
    candidates = []
    while len(candidates) < topn and c < len(neighbours):
        if neighbours[c][0] not in seen:
            candidates.append(neighbours[c])
        c+=1
    return dict(candidates)

# plot functions
def plot_travel_distance(log,model,core_init,method=np.mean):
    """plot distance travelled from the original set of seed words
    Arguments:
        - log (dict): logged decisions
        - model (models.word2vec.Word2Vec): the word2vec model
        - method (function): how to compute the distance (np.max or np.mean)
    """
    dists = {}
    for i,l in sorted(log.items()):
        #print(log[i]['core'])
        dists[i] = method([euclidean(core_init,model.wv[w]) for w in log[i]['core']])
    
    return pd.Series(list(dists.values()),index=list(dists.keys())).plot()


def plot_2d(log,model,figsize=None,include_neighbours=False):
    """Plot annotations on a 2 dimensional plane using TSNE
    Arguments:
        - log (dict): logged decisions
        - model (models.word2vec.Word2Vec): the word2vec model
    """
    last_round = sorted(log.items())[-1]
   
    core,periph = list(last_round[1].get('core',[])),list(last_round[1].get('peripheral',[]))
    words = core + periph

    if include_neighbours:
        neighbours = {n for w in words for n,v in model.wv.most_similar(w,topn=10) if n not in words}
        words += list(neighbours)
    X = [model.wv[w] for w in words]
    if figsize:
        fig = plt.figure(figsize = figsize)

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    
    for i,w in enumerate(words):
        c = 'blue'
        if i < len(core):
            c = 'red'
        if i > len(core + periph):
            c = 'green'
        plt.scatter(X_2d[i, 0], X_2d[i, 1],c=c)
        plt.text(X_2d[i, 0],X_2d[i, 1],w,size=14)
    
    
    return plt.show()


sampling_options = {'average': {
                        'method': average_all },
                    
                    'query_tokens': {
                        'method': query_tokens,
                        'args' : {
                            "tokens" : [],
                            "merge" : False }
                                    },
                    
                    'random': {
                        'method': random_sample,
                        'args': {
                            "cutoff": 5 }
                                    },
                   
#                    'entropy': {
#                        'method': entropy_sample,
#                         'args': {
#                             'topn':2,
#                             'init_vec': core_init,
#                             'reverse': True}
#                                     },
                   
                   'distance' : {
                       'method': distance_sample,
                        'args': {
                            'topn': 5,
                            'method': cosine,
                            'init_vec': None,
                            'reverse': True
                                        }
                                    }
                                }