#import entity_grid
from entity_grid import *
from parsetree import *
import svm as disvm
import json
import pickle
import os
import re
# from easydict import EDict as edict
# args = edict()
# args.jobs = 1
# args.mem = 10

args={}
args['jobs']=4
args['mem']=5
args['parser']="/data/cspark/stanford-parser-full-2014-10-31/stanford-parser.jar"
args['models']="/data/cspark/stanford-parser-full-2014-10-31/stanford-parser-3.5.0-models.jar"
args['grammar']="/data/cspark/stanford-parser-full-2014-10-31/englishPCFG.caseless.ser.gz"
args['threads']=2
args['max_length']=1000
testgrid_path="/data/cspark/browncoherence/bin64/TestGrid"
trainer = disvm.RankSVMTrainer(20)

jsonfile = open('./data/example_tree.json', 'r')


json_data=jsonfile.read()
jsondata=json.loads(json_data)
json_imgs=jsondata['images']
contents={}

for json_img in json_imgs:
	pageurl=os.path.basename(json_img['docpath']).encode('ascii','ignore')
	concattree=""
	if contents.has_key(pageurl):
		concattree=contents[pageurl]
	for sentence in json_img['sentences']:
		encode_tree=sentence['raw'].encode('ascii','ignore')
		if encode_tree in concattree:
			pass
		else:
			concattree=concattree+encode_tree
	contents[pageurl]=concattree
tree_list=[]
key_list=[]
count=0
for k,content in contents.iteritems():
	#if count <300:
	key_list.append(k)
	#content={'content':content,'key':k}
	tree_list.append(content)
	count+=1


trees_key_list=[]
for trees, key in itertools.izip(trees_list, uniq_list):
    if len(trees.strip())!=0:
        trees_key_list.append({'trees':trees.encode('ascii','ignore'),'key':str(key)})
grids=get_grids_multi_documents(testgrid_path,trees_key_list,args['jobs'])


# trees_key_list=[]
# for trees, key in itertools.izip(trees_list, key_list):
# 	if trees is not None:
# 		trees_key_list.append({'trees':trees,'key':key})
# print "disney trees ans key dump"
# pickle.dump( trees_key_list, open( "./data/disney_trees_key_list.p", "wb" ) )
# grids=get_grids_multi_documents(testgrid_path,trees_key_list,3) #jobs is 3
# print "disney grid dump"
# pickle.dump( grids, open( "./data/disney_grid_list.p", "wb" ) )
#data_entities=[]
for grid, trees_and_key  in itertools.izip(grids, trees_key_list):
	if grid and grid.strip()!="":

		model=new_entity_grid(grid, syntax=True,max_salience=0, history=3)
		trainer.add_model(model)
print "training start"
trainer.train()
ntsb_weights = trainer.weights
print ntsb_weights
pickle.dump( ntsb_weights, open( "./data/example_weights.p", "wb" ) )
