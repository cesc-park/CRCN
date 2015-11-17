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
args['jobs']=1
args['mem']=10
args['parser']="./stanford-parser/stanford-parser.jar"
args['models']="./stanford-parser/stanford-parser-3.5.0-models.jar"
args['grammar']="./stanford-parser/englishPCFG.caseless.ser.gz"
args['threads']=4
args['max_length']=1000
testgrid_path="./browncoherence/bin64/TestGrid"
trainer = disvm.RankSVMTrainer(20)

jsonfile = open('../data/example_tree.json', 'r')


json_data=jsonfile.read()
jsondata=json.loads(json_data)
json_imgs=jsondata['images']
contents={}
totallen=len(json_imgs)
sentence_list=[]
for i,json_img in enumerate(json_imgs):
	#pageurl=os.path.basename(json_img['docpath']).encode('ascii','ignore')
	#concatstring=""
	#if contents.has_key(pageurl):
	#	concatstring=contents[pageurl]
	for j,sentence in enumerate(json_img['sentences']):
		string=re.sub('\.+','.',sentence['raw'].encode('ascii','ignore')).replace('.','.\n')
		sentence_list.append(string)
		# trees=get_parsed_trees_a_document(string,args)
		# json_img['sentences'][j]['entities']=[]
		# if trees is not None:
		# 	grid=get_grids_a_document(testgrid_path,trees)
		# 	if grid and grid.strip()!="":
		# 		model=new_entity_grid(grid, syntax=True,max_salience=0, history=2)
		# 		json_img['sentences'][j]['entities']=model.entities
trees_list=get_parsed_trees_multi_documents(sentence_list,args)
trees_key_list=[]
for key,trees in enumerate(trees_list):
	if trees is not None:
		trees_key_list.append({'trees':trees,'key':str(key)})
grids=get_grids_multi_documents(testgrid_path,trees_key_list,3) #jobs is 3
entities_list=[]
for grid, trees_and_key  in itertools.izip(grids, trees_key_list):
	if grid and grid.strip()!="":
		model=new_entity_grid(grid, syntax=True,max_salience=0, history=2)
		entities_list.append({'entities':model.entities,'key':trees_and_key['key']})

count =0
entity_idx=0
for i,json_img in enumerate(json_imgs):
	for j,sentence in enumerate(json_img['sentences']):
		if count==int(entities_list[entity_idx]['key']):
			json_img['sentences'][j]['entities']=entities_list[entity_idx]['entities']
			entity_idx=entity_idx+1
		else:
			json_img['sentences'][j]['entities']=[]
		count =count+1
	json_imgs[i]=json_img
jsondata['images']=json_imgs
fname = open('../data/example_entities.json', 'w')
json.dump(jsondata, fname,ensure_ascii=False)
fname.close()
