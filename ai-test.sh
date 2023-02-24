#!/bin/bash
echo "------------------------------------------------------"
echo " ai-test.sh "
echo "------------------------------------------------------"
FILE_PATH=$(pwd)
export AI_HOME="/home/plukasik/ai/ai-daemon"
#export PYTHONPATH="~/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="~/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0
eval "$(conda shell.bash hook)"
conda activate tf

cd $AI_HOME/src
rm -r ./result/test*

# -t  <nazev-testu>
# -u1 <poc.neuronu 1>
# -u2 <poc.neuronu 2>
# -m1 <model 1>
# -m2 <model 2>
# -e  <poc.epoch>
# -l1 <poc.vrstev 1>
# -l2 <poc.vrstev 2>
# -b  <batch>
# -db <debug, True False>
# -p  <interpolate True False>
./ai-part01.sh -t=test01 -u1=230 -l1=2 -m1=DENSE   -u2=90 -l2=0 -m2=GRU -e=500 -b=256 -db=True -ip=False
./ai-part01.sh -t=test02 -u1=230 -l1=2 -m1=DENSE   -u2=90 -l2=0 -m2=GRU -e=500 -b=256 -db=True -ip=False
./ai-part01.sh -t=test03 -u1=230 -l1=2 -m1=DENSE   -u2=90 -l2=0 -m2=GRU -e=500 -b=256 -db=True -ip=False
./ai-part01.sh -t=test04 -u1=230 -l1=2 -m1=DENSE   -u2=90 -l2=0 -m2=GRU -e=500 -b=256 -db=True -ip=False

src="./result/test*.csv"
dst="./br_data/archiv/RESULTS/res_group.CSV"
cat $src >>$dst
echo $src$dst" ok....."
