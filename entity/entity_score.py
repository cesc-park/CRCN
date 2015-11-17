#import entity_grid
from entity_grid import *
from parsetree import *
import pickle
import os
import re
from operator import itemgetter
import numpy as np
# from easydict import EDict as edict
# args = edict()
# args.jobs = 1
# args.mem = 10

args={}
args['jobs']=2
args['mem']=5
args['parser']="./stanford-parser/stanford-parser.jar"
args['models']="./stanford-parser/stanford-parser-3.5.2-models.jar"
args['grammar']="./stanford-parser/stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz"
args['threads']=2
args['max_length']=1000
testgrid_path="./browncoherence/bin64/TestGrid"

def entity_score(content_list,weights_path):

    # input content_list ["This is cspark code paragraph1", "2"...]
    # output sorted (key,score) pair list [(key,score), ..]
    global args
    global testgrid_path
    ntsb_weights=pickle.load(open( weights_path, "r" ))
    uniq_list=range(0,len(content_list)) #for multiple process
    trees_list=get_parsed_trees_multi_documents(content_list,args)
    trees_key_list=[]
    for trees, key in itertools.izip(trees_list, uniq_list):
        if trees is not None:
            trees_key_list.append({'trees':trees,'key':str(key)})
    grids=get_grids_multi_documents(testgrid_path,trees_key_list,args['jobs'])
    dict_score={}
    for grid, trees_and_key  in itertools.izip(grids, trees_key_list):
        if grid and grid.strip()!="":
            model=new_entity_grid(grid, syntax=True,max_salience=0, history=2)
            key=int(trees_and_key['key'])
            score=np.dot(ntsb_weights,model.get_trans_prob_vctr())
            dict_score[key]=score
        else:
            dict_score[key]=0
    for key in uniq_list:
        if dict_score.has_key(key):
            pass
        else:
            dict_score[key]=0

    return sorted(dict_score.iteritems(), key=itemgetter(1), reverse=True)


def entity_feature(trees_list):

    # input content_list ["This is cspark code paragraph1", "2"...]
    # output sorted (key,score) pair list [(key,score), ..]
    global args
    global testgrid_path
    uniq_list=range(0,len(trees_list)) #for multiple process
    trees_key_list=[]
    for trees, key in itertools.izip(trees_list, uniq_list):
        if len(trees.strip())!=0:
            trees_key_list.append({'trees':trees.encode('ascii','ignore'),'key':str(key)})
    grids=get_grids_multi_documents(testgrid_path,trees_key_list,args['jobs'])
    feature_vec_list={}
    for grid, trees_and_key  in itertools.izip(grids, trees_key_list):
        if grid and grid.strip()!="":
            model=new_entity_grid(grid, syntax=True,max_salience=0, history=3)
            key=int(trees_and_key['key'])
            feature_vec_list[key]=model.get_trans_prob_vctr()
        else:
            key=int(trees_and_key['key'])
            feature_vec_list[key]=np.zeros(64)

    for key in uniq_list:
        if feature_vec_list.has_key(key):
            pass
        else:
            feature_vec_list[key]=np.zeros(64)
    return feature_vec_list


# trees_key_list=[]
# for trees, key in itertools.izip(trees_list, key_list):
#     if trees is not None:
#         trees_key_list.append({'trees':trees,'key':key})
# print "disney trees ans key dump"
# pickle.dump( trees_key_list, open( "./data/disney_trees_key_list.p", "wb" ) )
# grids=get_grids_multi_documents(testgrid_path,trees_key_list,3) #jobs is 3
# print "disney grid dump"
# pickle.dump( grids, open( "./data/disney_grid_list.p", "wb" ) )
# #data_entities=[]

# for grid, trees_and_key  in itertools.izip(grids, trees_key_list):
#     if grid and grid.strip()!="":

#         model=new_entity_grid(grid, syntax=True,max_salience=0, history=2)
#         #data_entities.append({'key':trees_and_key['key'],'entities':model.entities})
#         trainer.add_model(model)
# print "training start"
# trainer.train()
# ntsb_weights = trainer.weights
# print ntsb_weights
# pickle.dump( ntsb_weights, open( "./data/disney_weights.p", "wb" ) )
