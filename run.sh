#!/bin/bash

data_dir=./mnist
export PATH=$PATH:/usr/local/cuda/bin
step=3

# (0) Get pdnn scripts
if [ $step -le 0 ];then
    echo "Start setup."

    if [ ! -d tools ]; then
	mkdir -p tools
	cd tools
	
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

    paste -d " " data/train/train-images.txt data/train/train-labels.txt | awk '{print NR-1 " 0 " $0}'  > data/train/train.data || exit 1
#    paste -d " " data/test/test-images.txt data/test/test-labels.txt | awk '{print NR-1 " 0 " $0}'  > data/test/test.data || exit 1

#    shuf data/train/train.data > data/tmp.data
    paste -d " " data/train/train-images.txt data/train/train-labels.txt | shuf > data/tmp.data
    mkdir -p data/train_cv10 data/train_tr90
    head -n 1000 data/tmp.data | awk '{print NR-1 " 0 " $0}' > data/train_cv10/train_cv10.data
    tail -n +1000 data/tmp.data | awk '{print NR-1 " 0 " $0}' > data/train_tr90/train_tr90.data
    rm data/tmp.data

#    tools/pfile_utils-v0_51/bin/pfile_create -i data/train/train.data -o data/train/train.pfile -f 784 -l 1 || exit 1
    tools/pfile_utils-v0_51/bin/pfile_create -i data/train_tr90/train_tr90.data -o data/train_tr90/train_tr90.pfile -f 784 -l 1 || exit 1
    tools/pfile_utils-v0_51/bin/pfile_create -i data/train_cv10/train_cv10.data -o data/train_cv10/train_cv10.pfile -f 784 -l 1 || exit 1
#    tools/pfile_utils-v0_51/bin/pfile_create -i data/test/test.data -o data/test/test.pfile -f 784 -l 1 || exit 1

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

exit

# (4) forward deep generative model and plot numbers
if [ $step -le 4 ]; then

    . path.sh

    nnet-forward final.mdl ark,t:scripts/eval.param ark,t:exp/eval.out

    ii=-1
    while read line ;
    do
	if [ ! $ii -eq -1 ] ;then
	    echo $ii
	    echo "----------------------------------------------------------------------------------"
	    echo $line | awk '{ for (i = 1; i <= NF; i++) printf("%s%s", $i > 50 ? "+" : " ", i % 28 ? "" : "\n") }'
	    echo "----------------------------------------------------------------------------------"
	fi
	ii=$((ii+1))
    done < exp/eval.out
fi

# (5) convert to png file
if [ $step -le 5 ]; then

    mkdir -p images

    ii=-1
    while read line ;
    do
        if [ ! $ii -eq -1 ] ;then
	    echo $line | sed s%]%% | awk 'BEGIN { print "P2 28 28 255" } { for (i = 1; i <= NF; i++) printf("%d%s", $i < 0 ? 0 : $i > 255 ? 255 : int($i) , i % 14 ? " " : "\n") }' | pnmtopng - > images/image.$ii.png
	fi
	ii=$((ii+1))
    done < exp/eval.out
fi