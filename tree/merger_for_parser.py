import json
import pickle
import os
import re

import glob
def dirGlob(dir, pattern):
    fullPattern = os.path.join(dir,pattern)
    return sorted(glob.glob(fullPattern))
def get_files(input_path):
    files = dirGlob(input_path, 'example_tree_*.json')
    return files


jsonfile = open('../data/example.json', 'r')


json_data=jsonfile.read()
jsondata=json.loads(json_data)

files_tree=get_files('../data/')

for treedata in files_tree:
	treef=open(treedata, 'r')
	tree_data=treef.read()
	tree_json=json.loads(tree_data)
	for tree in tree_json:
		imgid=tree['imgid']
		paraid=tree['paraid']
		tree_raw='\n'.join(tree['tree_list'])
		jsondata['images'][imgid]['sentences'][paraid]['tree']=tree_raw
	treef.close()
count=0
for i,images in enumerate(jsondata['images']):
	for j,sentence in enumerate(images['sentences']):
		if sentence.has_key('tree'):
			count+=1
		else:
			jsondata['images'][i]['sentences'][j]['tree']=""

print len(jsondata['images']),count
fname = open('../data/example_tree.json', 'w')
json.dump(jsondata, fname)
fname.close()
