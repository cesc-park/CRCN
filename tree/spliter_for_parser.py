import json
import pickle
import os
import re

SPLITNUM = 200

jsonfile = open('../data/example.json', 'r')


json_data=jsonfile.read()
jsondata=json.loads(json_data)
json_imgs=jsondata['images']
para_list=[]
for imgid,json_img in enumerate(json_imgs):
	for paraid,sentence in enumerate(json_img['sentences']):
		content=re.sub(' +',' ',re.sub('\.','. ',re.sub('\.+','.',sentence['raw'].encode('ascii','ignore'))))
		para_list.append({'raw':content,'imgid':imgid,'paraid':paraid})

for i in range(0,len(para_list)):
	if i % SPLITNUM == 0 and i != 0:
		fname = open('../data/example_split_'+str(i/SPLITNUM)+'.json', 'w')
		#print i,str(i/SPLITNUM)
		json.dump(para_list[i-SPLITNUM:i], fname,ensure_ascii=False)
		fname.close()
	elif i == len(para_list)-1:
		print (i/SPLITNUM)*SPLITNUM, str(i/SPLITNUM+1)
		print "Change last iterator in StanfordCoreNlpTreeAdder.java which is i <= XX to i <= ", str(i/SPLITNUM+1)
		fname = open('../data/example_split_'+str(i/SPLITNUM+1)+'.json', 'w')
		json.dump(para_list[(i/SPLITNUM)*SPLITNUM:], fname,ensure_ascii=False)
		fname.close()
