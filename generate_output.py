import json
import sys
from topk_utils import *
import numpy as np
import re
from gensim import models
from keras.models import Sequential
from keras.layers.recurrent import CRCN,RCN
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Flatten



def create_rcn_model():
    model=Sequential()
    model.add(RCN(300, 300, return_sequences=True,activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 4096,init='normal'))
    model.add(Dropout(0.7))
    return model
def create_crcn_model():
    model=Sequential()
    model.add(CRCN(300, 300, return_sequences=True,activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 512,init='normal'))
    model.add(Dropout(0.5))
    model.add(Embedding(300, 4096,init='normal'))
    model.add(Dropout(0.7))
    return model

jsonfile = open('./data/example_tree.json', 'r')
json_data=jsonfile.read()
jsondata=json.loads(json_data)
jsonfile.close()
json_imgs=jsondata['images']


features_path = os.path.join('./data/', 'example.mat')
features_struct = scipy.io.loadmat(features_path)['feats'].transpose() # this features array length have to be same with images length

DOC2VEC_MODEL_PATH='./model/example.doc2vec'


jsonfile = open('./data/example_test.json', 'r')
json_data=jsonfile.read()
jsondata=json.loads(json_data)
jsonfile.close()
json_imgs_test=jsondata['images']


features_path = os.path.join('./data/', 'example_test.mat')
features_struct_test = scipy.io.loadmat(features_path)['feats'].transpose() # this features array length have to be same with images length

RCN_MODEL_PATH='./model/rcn_5.hdf5'
CRCN_MODEL_PATH='./model/crcn_5.hdf5'

contents={}

for i,json_img in enumerate(json_imgs_test):
    pageurl=os.path.basename(json_img['docpath']).encode('ascii','ignore')
    feature=features_struct_test[i]
    if contents.has_key(pageurl):
        pass#already in
    else:
        contents[pageurl]=[]
    contents[pageurl].append({'imgid':str(i),'filename':json_img['filename'],'sentences':json_img['sentences'],'feature':feature})


MAX_SEQ_LEN=15


contents_filtered = {}

for key, item in contents.iteritems():
    if len(item) > 4:
        contents_filtered[key] = item[:MAX_SEQ_LEN]


testset=contents_filtered.items()

count=0

model_loaded_entity = create_crcn_model()
model_loaded_entity.load_weights(CRCN_MODEL_PATH)
model_loaded_entity.compile(loss='crcn_score_func',optimizer='rmsprop')


model_loaded = create_rcn_model()
model_loaded.load_weights(RCN_MODEL_PATH)
model_loaded.compile(loss='rcn_score_func',optimizer='rmsprop')


doc2vecmodel = models.Doc2Vec.load(DOC2VEC_MODEL_PATH)


crcn_output_list=[]
rcn_output_list=[]



for i,tests in enumerate(testset):

    count+=1
    crcn_output=output_list_topk_crcn(tests[1],json_imgs,features_struct,doc2vecmodel,model_loaded_entity)
    crcn_output_list.append(crcn_output)

    rcn_output=output_list_topk_rcn(tests[1],json_imgs,features_struct,doc2vecmodel,model_loaded)
    rcn_output_list.append(rcn_output)
    print i

pickle.dump(crcn_output_list,open('./output_crcn.p','w'))
pickle.dump(rcn_output_list,open('./output_rcn.p','w'))



