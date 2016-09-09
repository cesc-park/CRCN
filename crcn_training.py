import sys
sys.path.append("./keras")
sys.path.append("./entity")
import theano
import numpy as np
import pickle
from gensim import models
import json
import os
import scipy.io
from entity_score import *
from load_models import *

MAX_SEQ_LEN= 10

model = create_crcn_blstm()
model.compile(loss='crcn_cost_func', optimizer='rmsprop')
# "images" is a numpy array of shape (nb_samples, nb_channels=3, width, height)
# "captions" is a numpy array of shape (nb_samples, max_caption_len=16, embedding_dim=256)
# captions are supposed already embedded (dense vectors).

#fisrt sentenceseq (training data nb, vector nb,  dimension)

DOC2VEC_MODEL_PATH='./model/example.doc2vec'
doc2vecmodel = models.Doc2Vec.load(DOC2VEC_MODEL_PATH)

jsonfile = open('./data/example_tree.json', 'r')


json_data=jsonfile.read()
jsondata=json.loads(json_data)
json_imgs=jsondata['images']


features_path = os.path.join('./data/', 'example.mat')
features_struct = scipy.io.loadmat(features_path)['feats'].transpose()


contents={}

for i,json_img in enumerate(json_imgs):
    pageurl=os.path.basename(json_img['docpath']).encode('ascii','ignore')
    imgfeature=features_struct[i] #1 * 4096
    concatstring=""
    concattree=""
    for sentence in json_img['sentences']:
        if sentence['raw'] not in concatstring:
            concatstring+=sentence['raw']#re.sub('\.+','.',sentence['raw'].encode('ascii','ignore')).replace('.','.\n')
            concattree+=sentence['tree']
    if contents.has_key(pageurl):
        #already in
        contents[pageurl].append({'imgid':str(i),'filename':json_img['filename'],'feature':imgfeature,'raw':concatstring,'tree':concattree})
    else:
        contents[pageurl]=[]
        contents[pageurl].append({'imgid':str(i),'filename':json_img['filename'],'feature':imgfeature,'raw':concatstring,'tree':concattree})


#data cleaning image must have pair of sentence

data_list=[]
for k,item in contents.iteritems():
    itemcopy=[]
    for imgpair in item:
        try:
            doc2vecmodel.docvecs[imgpair['imgid']]
            itemcopy.append(imgpair)
        except:
            pass
    if len(itemcopy)>3:
        if len(itemcopy)>MAX_SEQ_LEN:
            iternum=len(itemcopy)/MAX_SEQ_LEN
            for i in range(0,iternum):
                data_list.append(itemcopy[i*MAX_SEQ_LEN:(i+1)*MAX_SEQ_LEN])
            if len(itemcopy)- iternum*MAX_SEQ_LEN >4:
                data_list.append(itemcopy[iternum*MAX_SEQ_LEN:])
        else:
            data_list.append(itemcopy)

document_trees=[]
for tdata_seq in data_list:
    document_tree=""
    for tdata in tdata_seq:
        document_tree+=tdata['tree']
    document_trees.append(document_tree)
print "entity feature extracting..."

entity_feat=entity_feature(document_trees)

training_num=len(data_list)
Sentenceseq=np.zeros((training_num,MAX_SEQ_LEN+1,300))
Imageseq=np.zeros((training_num,MAX_SEQ_LEN,4096))


for i,seq_list in enumerate(data_list):
    for j,seq_elem in enumerate(seq_list):
        Imageseq[i][j]=seq_elem['feature']
        Sentenceseq[i][j]=doc2vecmodel.docvecs[seq_elem['imgid']]
    Sentenceseq[i][MAX_SEQ_LEN]=np.pad(entity_feat[i],(0,300-64),'constant', constant_values=0)


for i in range(1,20):
    print "Number of stage", i
    model.fit(Sentenceseq, Imageseq, batch_size=100, nb_epoch=5,validation_split=0.1,shuffle=True)
    print "Checkpoint saved"
    model.save_weights('./model/crcn_'+str(i)+'.hdf5')



