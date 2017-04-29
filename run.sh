#!/bin/bash

data_dir=mnist
export PATH=$PATH:/usr/local/cuda/bin
step=0

# (0) Setup scripts and dataset
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
	    
            # (2) Complie tools.
	    cd kaldi-maxout/tools
	    make -j 4 || exit 1;
	    cd ../
	     
            # (3) Compile src dirs.
	    cd src
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

    if [ ! -d mnist ]; then

	mkdir -p mnist
	cd mnist
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ; gunzip train-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ; gunzip train-labels-idx1-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz ; gunzip t10k-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz	; gunzip t10k-labels-idx1-ubyte.gz	
	cd ../

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

    paste -d " " data/train/train-images.txt data/train/train-labels.txt | shuf --random-source=data/train/train-images.txt > data/tmp.data
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
    python tools/pdnn/cmds/run_DNN.py --train-data data/train_tr90/train_tr90.pfile,partition=10m,random=true,stream=true \
	--valid-data data/train_cv10/train_cv10.pfile,partition=10m,random=true,stream=true \
	--nnet-spec 784:500:500:10  --lrate D:0.0008:0.5:0.01,0.01:8 \
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
	| python scripts/calc_score.py 10 || exit 1

fi

# (6) Training convlutional neural network
if [ $step -le 6 ] ; then
    echo "Start traing CNN."

    mkdir -p cnn
    export PYTHONPATH=:$(pwd)/tools/pdnn/ ; export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ;
    python tools/pdnn/cmds/run_CNN.py --train-data data/train_tr90/train_tr90.pfile,partition=10m,random=true,stream=true \
	--valid-data data/train_cv10/train_cv10.pfile,partition=10m,random=true,stream=true \
        --conv-nnet-spec "1x28x28:256,9x9,p2x2,f" \
        --nnet-spec "300:10" \
        --lrate "D:0.08:0.5:0.2,0.2:4" --momentum 0.9 \
        --wdir cnn/ --param-output-file cnn/nnet.param \
        --cfg-output-file cnn/nnet.cfg --kaldi-output-file cnn/cnn.nnet || exit 1;

    echo "Finish training CNN"

fi

# (7) forward CNN to calc. posterior
if [ $step -le 7 ]; then
    echo "Forward CNN to calc. posterior."

    tools/kaldi-maxout/src/featbin/copy-feats ark,t:data/test/test-images.feats ark,scp:data/test/test-images.ark,data/test/test-images.scp
    export PYTHONPATH=:$(pwd)/tools/pdnn/ ; export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ;
    python tools/pdnn/cmds2/run_CnnFeat.py --in-scp-file data/test/test-images.scp --out-ark-file data/test/test-images.conv.forward  --cnn-param-file cnn/nnet.param --cnn-cfg-file cnn/nnet.cfg

    tools/kaldi-maxout/src/nnetbin/nnet-forward cnn/cnn.nnet ark:data/test/test-images.conv.forward ark:- \
        | tools/kaldi-maxout/src/featbin/append-feats ark:- ark,t:data/test/test-labels.feats ark,t:- \
        | python scripts/calc_score.py 10 || exit 1
    
fi
