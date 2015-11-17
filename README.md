
# CRCN
Coherence + Recurrent Neural Network +  Convolutional Neural Network

This project hosts the code for our NIPS 2015 paper.

+ Cesc Chunseong Park and Gunhee Kim. Expressing an Image Stream with a Sequence of Natural Sentences. In NIPS 2015
[[pdf](http://www.cs.cmu.edu/~gunhee/publish/nips15_stream2text.pdf)]

##Reference

If you use this code as part of any published research, please acknowledge the following paper.

```
@inproceedings{Cesc:2015:NIPS,
author    = {Cesc Chunseong Park and Gunhee Kim},
title     = "{Expressing an Image Stream with a Sequence of Natural Sentences}",
booktitle = {NIPS},
year      = 2015
}
```


#Running Code

git clone https://github.com/cesc-park/CRCN.git crcn

##Pre-requisite

1. stanford NLP

	Download stanford-parser.jar, stanford-parser-3.5.2-models.jar and englishPCFG.caseless.ser.gz
	```
	wget http://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip
	wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
	unzip stanford-parser-full-2015-04-20.zip
	unzip stanford-corenlp-full-2015-04-20.zip
	mv stanford-parser-full-2015-04-20 stanford-parser
	mv stanford-corenlp-full-2015-04-20 stanford-core
	cd stanford-parser
	jar xvf stanford-parser-3.5.2-models.jar
	```
2. Brown courpus

	We need "browncourpus" package to extract entity feature.
	```
	wget https://bitbucket.org/melsner/browncoherence/get/d46d5cd3fc57.zip -O browncoherence.zip
	unzip browncoherence.zip
	mv melsner-browncoherence-d46d5cd3fc57 browncoherence
	cd browncoherence
	mkdir lib64
	mkdir bin64
	vim Makefile
	```

	We have to change some lines in Makefile.
	Change from up to bottom.
	
	```
	WORDNET = 1
	WORDNET = 0
	```
	
	```
	CFLAGS = $(WARNINGS) -Iinclude $(WNINCLUDE) $(TAO_PETSC_INCLUDE) $(GSLINCLUDE)
	CFLAGS = $(WARNINGS) -Iinclude $(WNINCLUDE) $(TAO_PETSC_INCLUDE) $(GSLINCLUDE) -fpermissive 
	```
	
	```
	WNLIBS = -L$(WNDIR)/lib -lWN
	WNLIBS = -L$(WNDIR)/lib -lwordnet
	```
	
	Then build TestGrid.
	```
	make TestGrid
	cd ..
	```
3.  python modules

	Install all dependencies.
	```
	for req in $(cat python_requirements.txt); do pip install $req; done
	```


##Make New Dataset

1. Prepare Dataset

	Check out data format.
	```
	less json_data_format.txt
	```

2. Get Parsed Tree
	We use StanfordCoreNLP tools implemented with java to extract parsedtree. 
	```
	cd tree
	python spliter_for_parser.py
	javac -d . -cp .:./json-simple-1.1.1.jar:../stanford-core/stanford-corenlp-3.5.2.jar:../stanford-core/xom.jar:../stanford-core/stanford-corenlp-3.5.2-models.jar:../stanford-core/joda-time.jar:../stanford-core/jollyday.jar: StanfordCoreNlpTreeAdder.java
	java -cp .:./json-simple-1.1.1.jar:../stanford-core/stanford-corenlp-3.5.2.jar:../stanford-core/xom.jar:../stanford-core/stanford-corenlp-3.5.2-models.jar:../stanford-core/joda-time.jar:../stanford-core/jollyday.jar: parser.StanfordCoreNlpTreeAdder
	python merger_for_parser.py
	```


##Training

Make directory for training

```
mkdir model
```


1. Doc2Vec
	First, we have to training doc2vec model.

	```
	python doc2vec_training.py
	```

2. RCN
	Trainig RCN model.
	If you want to use GPU (In this example device is 0), excute below code.

	```
	CUDA_VISIBLE_DEVICES=0 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python rcn_training.py
	```
	If you want to use CPU, excute below code instead of above code.

	```
	python rcn_training.py
	```

3. CRCN
	Trainig RCN model.
	If you want to use GPU (In this example device is 0), excute below code.
	```
	CUDA_VISIBLE_DEVICES=0 THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python crcn_training.py
	```
	If you want to use CPU, excute below code instead of above code.

	```
	python crcn_training.py
	```


##Generate Output

Generating output is easy. The program will load training and test datasets, then automatically generate outputs.

```
python generate_output.py
```

##Acknowledge

We use keras package (We changed it to make our model). Thanks for keras developers.


## Authors

[Cesc Chunseong Park](http://vision.snu.ac.kr/cesc/) and [Gunhee Kim](http://www.cs.cmu.edu/~gunhee/),  
[Vision and Learning Lab](http://vision.snu.ac.kr/), 
Seoul National University
 
## License
MIT license

