
import numpy as np
import math

import os
import json
import sys
sys.path.append("./keras")
sys.path.append("./entity")


from scipy.spatial import distance
from operator import itemgetter
from rank_sequence_utils import *
from entity_score import *

def make_combine_list(combined_list,split_list,count,max_c):
    #input combined_list=[],
    #split_list=[[imgid1 imgid2 ..imgidk ], ..max_c numb]
    #output = [[imgid1 , ...max_c], another combination...]
    if count==max_c:
        return combined_list
    if len(combined_list):
        new_combined_list=[]
        for listin in combined_list:
            for topk in split_list[count]:
                newlist=list(listin)
                newlist.append(topk) #add one
                new_combined_list.append(newlist)
        return make_combine_list(new_combined_list,split_list,count+1,max_c)
    else:
        new_combined_list=[]
        for topk in split_list[count]:
            newlist=[]
            newlist.append(topk) #add one
            new_combined_list.append(newlist)
        return make_combine_list(new_combined_list,split_list,count+1,max_c)

def make_merge_list(merged_list,rank_comb_list,count,max_c):
    #input combined_list=[],
    #split_list=[[imgid1 imgid2 ..imgidk ], ..max_c numb]
    #output = [[imgid1 , ...max_c], another combination...]
    if count==max_c:
        return merged_list
    if len(merged_list):
        new_merged_list=[]
        for listin in merged_list:
            for topkseq in rank_comb_list[count]:
                newlist=list(listin)
                newlist+=topkseq #add one
                new_merged_list.append(newlist)
        return make_merge_list(new_merged_list,rank_comb_list,count+1,max_c)
    else:
        new_merged_list=[]
        for topkseq in rank_comb_list[count]:
            newlist=[]
            newlist+=topkseq #add one
            new_merged_list.append(newlist)
        return make_merge_list(new_merged_list,rank_comb_list,count+1,max_c)

def output_topk_crcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded):
    return ' '.join(output_list_topk_crcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded))
def output_list_topk_crcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded):
    SENT_DIM=300
    CNN_DIM=4096
    SPLIT_VAL=5
    TOPK=3
    SECOND_SELECT_TOP=15
    BRNN_FINAL_SELECT_TOP=5


    assert len(json_imgs)==len(features_struct), 'Dataset error: Image count is %d Feature count is %d.' % (len(json_imgs),len(features_struct), )
    img_match_index_list=[] #[[{"12":23.1},{"144":32},..top k], .. img seq num.. ]
    dict_dst={}
    paragraph_list=[]
    testdata_index_list=[testimg['imgid'] for testimg in testdata]
    image_seq_features=[testimg['feature'] for testimg in testdata]
    for seq_feature in image_seq_features:
        for i,data_feature in enumerate(features_struct):
            dst = distance.euclidean(seq_feature,data_feature)
            dict_dst[i]=dst
            #we considered euclidean not cosine similarity
        sorted_list=sorted(dict_dst.iteritems(), key=itemgetter(1), reverse=False)
        #img_match_index_list.append(sorted_list[:TOPK])
        top_paragraph=[]
        count=0
        for dict_top in sorted_list:
            if count==TOPK:
                break
            try:
                index_match=dict_top[0]
                if index_match in testdata_index_list or dict_top[1]<0.01: #remove test set in index_match
                    continue
                imgid=json_imgs[index_match]['imgid']
                doc2vecmodel[str(imgid)] #check sentence exits
                count+=1
                top_paragraph.append(imgid)
            except:
                pass
        paragraph_list.append(top_paragraph)

    #print paragraph_list
    #paragraph_list=[[imgid1 imgid2 ..imgidk ], ..seq numb]
    # divide and concat
    #have to change variable split number
    if len(paragraph_list)>=SPLIT_VAL+1:

        rank_comb_list=[]
        split_num=len(paragraph_list)/SPLIT_VAL
        if len(paragraph_list)%SPLIT_VAL==0:
            range_max=split_num
        else:
            range_max=split_num+1
        for sp in range(0,range_max):
            split_list=[]
            if sp==split_num:
                split_list=paragraph_list[sp*SPLIT_VAL:]
            else:
                split_list=paragraph_list[sp*SPLIT_VAL:(sp+1)*SPLIT_VAL]
            combined_list=make_combine_list([],split_list,0,len(split_list))
            caselen=len(combined_list)
            content_len=len(split_list)
            sentseqs=np.zeros((caselen, content_len,SENT_DIM))
            imgseqs=np.zeros((caselen, content_len,CNN_DIM))
            #vectorize data
            key_seq_list=[]
            document_trees=[]
            for i,imgid_seq in enumerate(combined_list):
                key_seq=[]
                document_tree=""
                for j,imgid in enumerate(imgid_seq):

                    imgseqs[i][j]=image_seq_features[sp*SPLIT_VAL+j]
                    key_seq.append(imgid)
                    json_img=json_imgs[imgid]
                    for sentence in json_img['sentences']:
                        if sentence['tree'] not in document_tree:
                            document_tree+=sentence['tree']
                    sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
                key_seq_list.append(key_seq)
                document_trees.append(document_tree)
            entity_feat=entity_feature(document_trees)

            rank_comb_list.append(rank_sequence_entity(sentseqs,imgseqs,entity_feat,key_seq_list,model_loaded)[:SECOND_SELECT_TOP])
        #rank_comb_list=[[(index,score]..casenum]..range_max]
        #merge Phase
        merged_list=make_merge_list([],rank_comb_list,0,range_max)

    else:#for not divided case
        combined_list=make_combine_list([],paragraph_list,0,len(paragraph_list))
        caselen=len(combined_list)
        content_len=len(paragraph_list)
        sentseqs=np.zeros((caselen, content_len,SENT_DIM))
        imgseqs=np.zeros((caselen, content_len,CNN_DIM))
        #vectorize data
        key_seq_list=[]
        document_trees=[]
        for i,imgid_seq in enumerate(combined_list):
            key_seq=[]
            document_tree=""
            for j,imgid in enumerate(imgid_seq):
                key_seq.append(imgid)
                imgseqs[i][j]=image_seq_features[j]
                json_img=json_imgs[imgid]
                for sentence in json_img['sentences']:
                    if sentence['tree'] not in document_tree:
                        document_tree+=sentence['tree']
                sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
            key_seq_list.append(key_seq)
            document_trees.append(document_tree)
        entity_feat=entity_feature(document_trees)
        merged_list=rank_sequence_entity(sentseqs,imgseqs,entity_feat,key_seq_list,model_loaded)[:SECOND_SELECT_TOP]
    del sentseqs
    del imgseqs
    caselen=len(merged_list)
    content_len=len(paragraph_list)
    sentseqs=np.zeros((caselen, content_len,SENT_DIM))
    imgseqs=np.zeros((caselen, content_len,CNN_DIM))
    key_seq_list=[]
    document_trees=[]
    for i,imgid_seq in enumerate(merged_list):
        key_seq=[]
        document_tree=""
        for j,imgid in enumerate(imgid_seq):
            key_seq.append(imgid)
            imgseqs[i][j]=image_seq_features[j]
            json_img=json_imgs[imgid]
            for sentence in json_img['sentences']:
                if sentence['tree'] not in document_tree:
                    document_tree+=sentence['tree']
            sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
        key_seq_list.append(key_seq)
        document_trees.append(document_tree)
    entity_feat=entity_feature(document_trees)
    final_list=rank_sequence_entity(sentseqs,imgseqs,entity_feat,key_seq_list,model_loaded)

    final_list=final_list[:BRNN_FINAL_SELECT_TOP]
    final_content_list=[]
    for imgid in final_list[0]:
        sentence_concat=""
        for sentence in json_imgs[imgid]['sentences']:
            ensent=sentence['raw'].encode('ascii','ignore')
            if ensent not in sentence_concat:
                sentence_concat+=ensent
        final_content_list.append(sentence_concat)

    return final_content_list
def output_topk_rcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded):
    return ' '.join(output_list_topk_rcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded))

def output_list_topk_rcn(testdata,json_imgs,features_struct,doc2vecmodel,model_loaded):
    SENT_DIM=300
    CNN_DIM=4096
    SPLIT_VAL=5
    TOPK=3
    SECOND_SELECT_TOP=15
    BRNN_FINAL_SELECT_TOP=5


    assert len(json_imgs)==len(features_struct), 'Dataset error: Image count is %d Feature count is %d.' % (len(json_imgs),len(features_struct), )
    img_match_index_list=[] #[[{"12":23.1},{"144":32},..top k], .. img seq num.. ]
    dict_dst={}
    paragraph_list=[]
    testdata_index_list=[testimg['imgid'] for testimg in testdata]
    image_seq_features=[testimg['feature'] for testimg in testdata]
    # This code can cover TOPK not only TOPK=1
    for seq_feature in image_seq_features:
        for i,data_feature in enumerate(features_struct):
            dst = distance.euclidean(seq_feature,data_feature)
            dict_dst[i]=dst
            #we have to consider cosine similarity not euclidean
        sorted_list=sorted(dict_dst.iteritems(), key=itemgetter(1), reverse=False)
        #img_match_index_list.append(sorted_list[:TOPK])
        top_paragraph=[]
        count=0
        for dict_top in sorted_list:
            if count==TOPK:
                break
            try:
                index_match=dict_top[0]
                if index_match in testdata_index_list or dict_top[1]<0.01: #remove test set in index_match
                    continue
                imgid=json_imgs[index_match]['imgid']
                doc2vecmodel[str(imgid)] #check sentence exits
                count+=1
                top_paragraph.append(imgid)
            except:
                pass
        paragraph_list.append(top_paragraph)

    #print paragraph_list

    #paragraph_list=[[imgid1 imgid2 ..imgidk ], ..seq numb]

    # divide and concat
    #have to change variable split number
    if len(paragraph_list)>=SPLIT_VAL+1:

        rank_comb_list=[]
        split_num=len(paragraph_list)/SPLIT_VAL
        if len(paragraph_list)%SPLIT_VAL==0:
            range_max=split_num
        else:
            range_max=split_num+1
        for sp in range(0,range_max):
            split_list=[]
            if sp==split_num:
                split_list=paragraph_list[sp*SPLIT_VAL:]
            else:
                split_list=paragraph_list[sp*SPLIT_VAL:(sp+1)*SPLIT_VAL]
            combined_list=make_combine_list([],split_list,0,len(split_list))
            caselen=len(combined_list)
            content_len=len(split_list)
            sentseqs=np.zeros((caselen, content_len,SENT_DIM))
            imgseqs=np.zeros((caselen, content_len,CNN_DIM))
            #vectorize data
            key_seq_list=[]
            for i,imgid_seq in enumerate(combined_list):
                key_seq=[]
                for j,imgid in enumerate(imgid_seq):
                    if type(imgid)!=tuple:
                        imgseqs[i][j]=image_seq_features[sp*SPLIT_VAL+j]
                        key_seq.append(imgid)
                        sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
                    else:
                        print "error"
                key_seq_list.append(key_seq)
            #print len(sentseqs)
            rank_comb_list.append(rank_sequence(sentseqs,imgseqs,key_seq_list,model_loaded)[:SECOND_SELECT_TOP])
        #rank_comb_list=[[(index,score]..casenum]..range_max]
        #merge Phase
        merged_list=make_merge_list([],rank_comb_list,0,range_max)
    else:#for not divide case
        combined_list=make_combine_list([],paragraph_list,0,len(paragraph_list))
        caselen=len(combined_list)
        content_len=len(paragraph_list)
        sentseqs=np.zeros((caselen, content_len,SENT_DIM))
        imgseqs=np.zeros((caselen, content_len,CNN_DIM))
        #vectorize data
        key_seq_list=[]
        for i,imgid_seq in enumerate(combined_list):
            key_seq=[]
            for j,imgid in enumerate(imgid_seq):
                key_seq.append(imgid)
                imgseqs[i][j]=image_seq_features[j]
                sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
            key_seq_list.append(key_seq)
        merged_list=rank_sequence(sentseqs,imgseqs,key_seq_list,model_loaded)[:SECOND_SELECT_TOP]
    del sentseqs
    del imgseqs

    caselen=len(merged_list)
    content_len=len(paragraph_list)
    sentseqs=np.zeros((caselen, content_len,SENT_DIM))
    imgseqs=np.zeros((caselen, content_len,CNN_DIM))
    key_seq_list=[]
    for i,imgid_seq in enumerate(merged_list):
        key_seq=[]
        for j,imgid in enumerate(imgid_seq):
            key_seq.append(imgid)
            imgseqs[i][j]=image_seq_features[j]
            sentseqs[i][j]=doc2vecmodel[str(imgid)] #always has paragraph cleaned data
        key_seq_list.append(key_seq)
    final_list=rank_sequence(sentseqs,imgseqs,key_seq_list,model_loaded)

    final_list=final_list[:BRNN_FINAL_SELECT_TOP]
    #print final_list
    # final_list is senseq list [[imgid1.., imgidend]..,.. FINAL_SELECT_TOP]
    final_content_list=[]
    for imgid in final_list[0]:
        sentence_concat=""
        for sentence in json_imgs[imgid]['sentences']:
            ensent=sentence['raw'].encode('ascii','ignore')
            if ensent not in sentence_concat:
                sentence_concat+=ensent
        final_content_list.append(sentence_concat)

    return final_content_list

