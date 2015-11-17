from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-15

def rcn_cost_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    ypred_copy=y_pred
    ytrue_copy=y_true
    def seq_score(out_matrix,img_matrix):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        #k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        #j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)
        return T.sum(T.dot(out_matrix,img_matrix.T)*eye)

        #return T.sum(T.dot(out_matrix,img_matrix.T)/(T.sqr(k_mat-j_mat)+1))
    def iter_k(out_matrix,k_img_matrix,ytrue,ypred):
        def iter_j(in_matrix, j_out_matrix,out_matrix,k_img_matrix):
            score_img_seq=T.maximum(0,seq_score(out_matrix,in_matrix)+1- seq_score(out_matrix, k_img_matrix))
            score_sent_seq=T.maximum(0,seq_score(j_out_matrix,k_img_matrix)+1- seq_score(out_matrix, k_img_matrix))
            return score_img_seq+score_sent_seq
        (inner_totalsum,updates)=theano.scan(fn=iter_j,sequences=[ytrue, ypred],non_sequences=[out_matrix, k_img_matrix])
        return T.sum(inner_totalsum)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true],non_sequences=[ytrue_copy,ypred_copy])
    return T.sum(sumscores)

def crcn_cost_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    ypred_copy=y_pred
    ytrue_copy=y_true
    def seq_score(out_matrix,img_matrix,entity):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        #k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        #j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        entityscore=T.dot(entity,img_matrix.T)
        #division_mat=1/(T.abs_(k_mat-j_mat)+1)
        #division_mat=division_mat/T.sum(division_mat) #normalize: sum of all coefficient =1
        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)
        return T.sum(T.dot(out_matrix,img_matrix.T)*eye) +T.sum(entityscore)
    def iter_k(out_matrix,k_img_matrix,ytrue,ypred):
        out_len=out_matrix.shape[0]
        entity=out_matrix[out_len-1:]
        out_matrix=out_matrix[:out_len-1]
        def iter_j(in_matrix, j_out_matrix,out_matrix,k_img_matrix):
            j_out_len=j_out_matrix.shape[0]
            jentity=j_out_matrix[j_out_len-1:]
            j_out_matrix=j_out_matrix[:j_out_len-1]
            score_img_seq=T.maximum(0,seq_score(out_matrix,in_matrix,entity)+1- seq_score(out_matrix,k_img_matrix,entity))
            score_sent_seq=T.maximum(0,seq_score(j_out_matrix,k_img_matrix,jentity)+1- seq_score(out_matrix,k_img_matrix,entity))
            return score_img_seq+score_sent_seq
        (inner_totalsum,updates)=theano.scan(fn=iter_j,sequences=[ytrue, ypred],non_sequences=[out_matrix, k_img_matrix])
        return T.sum(inner_totalsum)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true],non_sequences=[ytrue_copy,ypred_copy])
    return T.sum(sumscores)

def crcn_cohevec_cost_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    ypred_copy=y_pred
    ytrue_copy=y_true
    def seq_score(out_matrix,img_matrix,entity):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        #k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        #j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        entityscore=T.dot(entity[0],img_matrix.T)
        bazialiascore=T.dot(entity[1],img_matrix.T)
        #division_mat=1/(T.abs_(k_mat-j_mat)+1)
        #division_mat=division_mat/T.sum(division_mat) #normalize: sum of all coefficient =1
        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)
        return T.sum(T.dot(out_matrix,img_matrix.T)*eye) +T.sum(entityscore)+T.sum(bazialiascore)
    def iter_k(out_matrix,k_img_matrix,ytrue,ypred):
        out_len=out_matrix.shape[0]
        entity=out_matrix[out_len-2:]
        out_matrix=out_matrix[:out_len-2]
        def iter_j(in_matrix, j_out_matrix,out_matrix,k_img_matrix):
            j_out_len=j_out_matrix.shape[0]
            jentity=j_out_matrix[j_out_len-2:]
            j_out_matrix=j_out_matrix[:j_out_len-2]
            score_img_seq=T.maximum(0,seq_score(out_matrix,in_matrix,entity)+1- seq_score(out_matrix,k_img_matrix,entity))
            score_sent_seq=T.maximum(0,seq_score(j_out_matrix,k_img_matrix,jentity)+1- seq_score(out_matrix,k_img_matrix,entity))
            return score_img_seq+score_sent_seq
        (inner_totalsum,updates)=theano.scan(fn=iter_j,sequences=[ytrue, ypred],non_sequences=[out_matrix, k_img_matrix])
        return T.sum(inner_totalsum)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true],non_sequences=[ytrue_copy,ypred_copy])
    return T.sum(sumscores)


def crcn_score_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    def seq_score(out_matrix,img_matrix,entity):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        #k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        #j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        entityscore=T.dot(entity,img_matrix.T)

        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)

        return T.sum(T.dot(out_matrix,img_matrix.T)*eye) +T.sum(entityscore)
    # number=y_true.shape[0]
    # k_iter_list=T.arange(number)
    # j_iter_list=T.arange(number)
    def iter_k(out_matrix,img_matrix):
        out_len=out_matrix.shape[0]
        out_matrix_copy=out_matrix
        out_matrix=out_matrix_copy[:out_len-1]
        entity=out_matrix_copy[out_len-1:]
        #seq_score(out_matrix,img_matrix,entity)
        return seq_score(out_matrix,img_matrix,entity)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true])
    return T.sum(sumscores)

def crcn_cohevec_score_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    def seq_score(out_matrix,img_matrix,entity):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        #k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        #j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        entityscore=T.dot(entity[0],img_matrix.T)
        bazialiascore=T.dot(entity[1],img_matrix.T)

        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)

        return T.sum(T.dot(out_matrix,img_matrix.T)*eye) +T.sum(entityscore)+T.sum(bazialiascore)
    # number=y_true.shape[0]
    # k_iter_list=T.arange(number)
    # j_iter_list=T.arange(number)
    def iter_k(out_matrix,img_matrix):
        out_len=out_matrix.shape[0]
        out_matrix_copy=out_matrix
        out_matrix=out_matrix_copy[:out_len-2]
        entity=out_matrix_copy[out_len-2:]
        #seq_score(out_matrix,img_matrix,entity)
        return seq_score(out_matrix,img_matrix,entity)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true])
    return T.sum(sumscores)


def rcn_score_func(y_true, y_pred):
    # y_pred = (batch nb, vector nb,  dimension)
    # y_true = image vector = (batch nb, vector nb,  dimension)
    def seq_score(out_matrix,img_matrix):
        out_len=out_matrix.shape[0]
        img_len=img_matrix.shape[0]
        k_mat=T.repeat(T.arange(out_len).reshape((1,out_len)),img_len,axis=0)
        j_mat=T.repeat(T.arange(img_len).reshape((img_len,1)),out_len,axis=1)
        #entityscore=T.dot(entity,img_matrix.T)
        eye=T.eye(out_len,img_len)
        eye=eye/T.sum(eye)
        return T.sum(T.dot(out_matrix,img_matrix.T)*eye)

        #return T.sum(T.dot(out_matrix,img_matrix.T)/(T.sqr(k_mat-j_mat)+1))#+T.sum(entityscore)
    # number=y_true.shape[0]
    # k_iter_list=T.arange(number)
    # j_iter_list=T.arange(number)
    def iter_k(out_matrix,img_matrix):
        #out_len=out_matrix.shape[0]
        #out_matrix_copy=out_matrix
        #out_matrix=out_matrix_copy[:out_len-1]
        #entity=out_matrix_copy[out_len-1:]
        #seq_score(out_matrix,img_matrix)
        return seq_score(out_matrix,img_matrix)
    (sumscores,updates)=theano.scan(fn=iter_k,sequences=[y_pred,y_true])
    return T.sum(sumscores)

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

def to_categorical(y):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
