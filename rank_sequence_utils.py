import sys
sys.path.append("./keras")
import theano
from theano import tensor
import numpy as np
import pickle
import json
import os
import scipy.io
from operator import itemgetter

def model_output(sentseq,model):
    get_activations = theano.function([model.layers[0].input], model.layers[-1].output(train=False), allow_input_downcast=True)
    return get_activations(sentseq)

def model_score(sentseq,imgseq,model):
    return model.test(sentseq, imgseq)

def rank_sequence(sentseqs,imgseqs,keylist,model):
    score_list={}
    for i,sentseq in enumerate(sentseqs):
        sentseq=sentseq.reshape(1,sentseq.shape[0],sentseq.shape[1])
        imgseq=imgseqs[i]
        imgseq=imgseq.reshape(1,imgseq.shape[0],imgseq.shape[1])
        score_list[i]=model_score(sentseq,imgseq,model)
        #print score_list[i]
    sorted_list=sorted(score_list.iteritems(), key=itemgetter(1), reverse=True)
    #[(index,score),..]
    #print "Small: ",sorted_list[-1]
    #print "Top: ",sorted_list[0]
    sorted_key_list=[]
    for sort_index in sorted_list:
        index=sort_index[0]
        sorted_key_list.append(keylist[index])
    return sorted_key_list
def rank_sequence_entity(sentseqs,imgseqs,entity_feat,keylist,model):
    score_list={}
    for i,sentseq in enumerate(sentseqs):
        sentseq=sentseq.reshape(1,sentseq.shape[0],sentseq.shape[1])
        entity=np.pad(entity_feat[i],(0,sentseq.shape[2]-64),'constant', constant_values=0).reshape(1,1,sentseq.shape[2])
        sentseq=np.concatenate((sentseq,entity),axis=1)
        imgseq=imgseqs[i]
        imgseq=imgseq.reshape(1,imgseq.shape[0],imgseq.shape[1])
        score_list[i]=model_score(sentseq,imgseq,model)
        #print score_list[i]
    sorted_list=sorted(score_list.iteritems(), key=itemgetter(1), reverse=True)
    #[(index,score),..]
    #print "Small: ",sorted_list[-1]
    #print "Top: ",sorted_list[0]
    sorted_key_list=[]
    for sort_index in sorted_list:
        index=sort_index[0]
        sorted_key_list.append(keylist[index])
    return sorted_key_list
