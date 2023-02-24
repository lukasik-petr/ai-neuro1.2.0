#!/bin/bash
source=$1
TESTNAME=$2
RES_TESTNAME=$3
FILE_PATH=$(pwd)
echo "------------------------------------------------------"
echo " ai-part02.sh "
echo "------------------------------------------------------"
#export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
echo "Start ulohy: "$curr_timestamp
eval "$(conda shell.bash hook)"
conda activate tf

echo "Vysledek ulohy do grafu"
wd="./result"

filenames="`find $wd | grep 'res_' | grep '.csv'`"
#[ -d $TESTNAME ] || mkdir -p "$TESTNAME"

for file in $filenames;
do
    python3 ./py-src/ai-graf01.py $file $curr_timestamp
    python3 ./py-src/ai-group.py $file $curr_timestamp
done;

grafname1=$wd"/"$TESTNAME"_rmse.pdf"
grafname2=$wd"/"$TESTNAME"_stab.pdf"
filenames="*group.csv"
filename=$TESTNAME"group.csv"
fname="`ls -1t $wd/*/*group.csv | head -1`"
header="`sed -n '1p' $fname`"
echo $header >$wd/$filename
find . -name '*_group.csv'  -exec sed -n 2p   {} \; | sort -n -k1 >> $wd/$filename 
python3 ./py-src/ai-graf02.py  $wd/$filename $grafname1 $curr_timestamp
python3 ./py-src/ai-graf03.py  $wd/$filename $grafname2 $TESTNAME $curr_timestamp

cd $source
echo "TESTNAME :"$TESTNAME" RES_TESTNAME:"$RES_TESTNAME
mv $TESTNAME*.pdf ./$RES_TESTNAME
#mv $res_*.pdf ./$RES_TESTNAME
#rm -r $TESTNAME*
cd $FILE_PATH
