#!/bin/bash

data_dir=./mnist
export PATH=$PATH:/usr/local/cuda/bin
step=0

# (0) Get pdnn scripts
if [ $step -le 0 ];then

    if [ ! -d tools ]; then
	mkdir -p tools
	cd tools
	
	svn co https://github.com/yajiemiao/pdnn/trunk pdnn || exit 1
	
	wget http://www.icsi.berkeley.edu/ftp/pub/real/davidj/pfile_utils-v0_51.tar.gz  || exit 1
	tar -xvzf pfile_utils-v0_51.tar.gz  || exit 1
	cd pfile_utils-v0_51 
	./configure --prefix=`pwd` --with-quicknet=`pwd`/../quicknet-v3_33/lib || exit 1
	make -j 4 || exit 1
	make install || exit 1
	cd ../../
    fi
    
fi

# (1) data prep
if [ $step -le 1 ]; then
    mkdir -p data

    od -An -v -tu1 -j16 -w784 $data_dir/train-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' > data/train/train-images.txt
    od -An -v -tu1 -j8 -w1 $data_dir/train-labels-idx1-ubyte | tr -d ' ' > data/train/train-labels.txt
    
    od -An -v -tu1 -j16 -w784 $data_dir/t10k-images-idx3-ubyte | sed 's/^ *//' | tr -s ' ' > data/test/test-images.txt
    od -An -v -tu1 -j8 -w1 $data_dir/t10k-labels-idx1-ubyte | tr -d ' ' > data/test/test-labels.txt
fi

# (2) make PFile format
if [ $step -le 2 ]; then

#    paste -d " " data/train-images.txt data/train-labels.txt | awk '{print "0 " NR-1 " " $0}'  > data/train.data
#    paste -d " " data/test-images.txt data/test-labels.txt | awk '{print "0 " NR-1 " " $0}'  > data/test.data

    paste -d " " data/train/train-images.txt data/train/train-labels.txt | awk '{print NR-1 " 0 " $0}'  > data/train/train.data
    paste -d " " data/test/test-images.txt data/test/test-labels.txt | awk '{print NR-1 " 0 " $0}'  > data/test/test.data

    tools/pfile_utils-v0_51/bin/pfile_create -i data/train/train.data -o data/train/train.pfile -f 784 -l 1
    tools/pfile_utils-v0_51/bin/pfile_create -i data/test/test.data -o data/test/test.pfile -f 784 -l 1

fi

exit

# (3) train deep generative model
if [ $step -le 3 ]; then

    mkdir -p exp

    export PYTHONPATH=:$(pwd)/pdnn/ ; export THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 ; 
    python pdnn/cmds/run_DGM.py --train-data data/train.pfile,partition=10m,random=true,stream=false \
	--valid-data data/test.pfile,partition=10m,random=true,stream=false \
	--nnet-spec 784+10:300:64 --generative-nnet-spec 64+10:500:784 --activation maxout:3 --lrate D:0.008:0.5:0.001,0.001:8 \
	--input-scaling 0.00001 --variance-ignore True --wdir exp --kaldi-output-file exp/dgm

    cp exp/dgm.gen final.mdl

fi

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