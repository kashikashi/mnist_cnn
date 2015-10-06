#!/bin/bash

data_dir=./mnist
export PATH=$PATH:/usr/local/cuda/bin
step=4

# (0) Get pdnn scripts
if [ $step -le 0 ];then
    echo "Start setup."

    if [ ! -d tools ]; then
	mkdir -p tools
	cd tools

	###### Set up KALDI.
	if [ ! -d kaldi-maxout ]; then
            # (1) Download
	    echo "Checking out KALDI."
	    svn co https://svn.code.sf.net/p/kaldi/code/trunk kaldi-maxout
	    
            # (2) Change revesion to 4985
	    cd kaldi-maxout
	    svn update -r 4960
	    
            # (3) Complie tools.
	    cd tools
	    make -j 4 || exit 1;
	    cd ../
	     
            # (4) Compile src dirs.
	    cd src/nnet
	    mv nnet-component.h nnet-component.h.buckup
	    mv nnet-component.cc nnet-component.cc.buckup
	    mv nnet-activation.h nnet-activation.h.buckup
	    wget http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-component.h
	    wget http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-component.cc
	    wget http://www.cs.cmu.edu/~ymiao/codes/kaldipdnn/nnet-activation.h
	    cd ../
	    
	    ./configure
	    make depend
	    make
	    cd ../
	    
	    cd ../
	fi
	
	
	svn co https://github.com/yajiemiao/pdnn/trunk pdnn || exit 1

	wget ftp://ftp.icsi.berkeley.edu/pub/real/davidj/quicknet.tar.gz || exit 1
	tar -xvzf quicknet.tar.gz
	cd quicknet-v3_33/
	./configure --prefix=`pwd`  || exit 1
	make install  || exit 1
	cd ..

	
	wget http://www.icsi.berkeley.edu/ftp/pub/real/davidj/pfile_utils-v0_51.tar.gz  || exit 1
	tar -xvzf pfile_utils-v0_51.tar.gz  || exit 1
	cd pfile_utils-v0_51 
	./configure --prefix=`pwd` --with-quicknet=`pwd`/../quicknet-v3_33/lib || exit 1
	make -j 4 || exit 1
	make install || exit 1
	cd ../../
    fi
    echo "Finish setup"  
fi

# (1) data prep
if [ $step -le 1 ]; then
    echo "Start data prep."
    mkdir -p data/train
    mkdir -p data/test

    od -An -v -tu1 -j16 -w784 $data_dir/train-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' > data/train/train-images.txt || exit 1
    od -An -v -tu1 -j8 -w1 $data_dir/train-labels-idx1-ubyte | tr -d ' ' > data/train/train-labels.txt || exit 1
    
    od -An -v -tu1 -j16 -w784 $data_dir/t10k-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' > data/test/test-images.txt || exit 1
    od -An -v -tu1 -j8 -w1 $data_dir/t10k-labels-idx1-ubyte | tr -d ' ' > data/test/test-labels.txt || exit 1
    echo "Finish data prep."
fi

# (2) make PFile format
if [ $step -le 2 ]; then
    echo "Start conversion to PFile."

#    paste -d " " data/train/train-images.txt data/train/train-labels.txt | awk '{print NR-1 " 0 " $0}'  > data/train/train.data || exit 1

    paste -d " " data/train/train-images.txt data/train/train-labels.txt | shuf > data/tmp.data
    mkdir -p data/train_cv10 data/train_tr90
    head -n 1000 data/tmp.data | awk '{print NR-1 " 0 " $0}' > data/train_cv10/train_cv10.data
    tail -n +1000 data/tmp.data | awk '{print NR-1 " 0 " $0}' > data/train_tr90/train_tr90.data
    rm data/tmp.data

    tools/pfile_utils-v0_51/bin/pfile_create -i data/train_tr90/train_tr90.data -o data/train_tr90/train_tr90.pfile -f 784 -l 1 || exit 1
    tools/pfile_utils-v0_51/bin/pfile_create -i data/train_cv10/train_cv10.data -o data/train_cv10/train_cv10.pfile -f 784 -l 1 || exit 1

    echo "Finish conversion to PFile."
fi

# (3) train deep neural networks
if [ $step -le 3 ]; then
    echo "Start training."
    mkdir -p dnn

    export PYTHONPATH=:$(pwd)/tools/pdnn/ ; export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ; 
    python tools/pdnn/cmds/run_DNN.py --train-data data/train_tr90/train_tr90.pfile,partition=10m,random=true,stream=false \
	--valid-data data/train_cv10/train_cv10.pfile,partition=10m,random=true,stream=false \
	--nnet-spec 784:300:10 --activation maxout:3 --lrate D:0.0008:0.5:0.01,0.01:8 \
	--wdir dnn/ --kaldi-output-file dnn/nnet

    echo "Finish training."
fi

# (4) forward network to calc. posterior
if [ $step -le 4 ]; then
    echo "Forward network to calc. posterior."

    awk '{print NR " [ " $0 " ]"}' data/test/test-images.txt > data/test/test-images.feats || exit 1

    tools/kaldi-maxout/src/nnetbin/nnet-forward dnn/nnet ark,t:data/test/test-images.feats ark,t:data/test/test-images.posterior || exit 1

fi

# (5) Calcurate scores (option).
if [ $step -le 5 ]; then
    echo "Calc. score"

    awk '{print NR " [ " $0 " ]"}' data/test/test-labels.txt > data/test/test-labels.feats || exit 1

    tools/kaldi-maxout/src/nnetbin/nnet-forward dnn/nnet ark,t:data/test/test-images.feats ark:- \
	| tools/kaldi-maxout/src/featbin/append-feats ark:- ark,t:data/test/test-labels.feats ark,t:- \
	| python calc_score.py || exit 1

fi

