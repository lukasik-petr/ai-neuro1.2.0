#!/bin/bash
echo "------------------------------------------------------"
echo " ai-part01.sh "
echo "------------------------------------------------------"
FILE_PATH=$(pwd)
START=0
END=2
export AI_HOME="~/ai/ai-daemon"
#export PYTHONPATH="~/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="~/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0
eval "$(conda shell.bash hook)"
conda activate tf

#----------------------------------------------------------------------
# implicitni parametry - mozno zmenit  z 'cmd lajny'
#----------------------------------------------------------------------
STATUS="run"        # run,start,....
TESTNAME="test01"   # nazev testu
UNITS_1="79"        # LSTM=91, DENSE=79
UNITS_2="0"         # LSTM=91, DENSE=79
MODEL_1="DENSE"     # typ vrstvy_1 LSTM DENSE GRU CONV1D
MODEL_2=""          # typ vrstvy_2 LSTM DENSE GRU CONV1D
EPOCHS="500"        # Poc. treninkovych cyklu
LAYERS_1="2"        # pocet vrstev v prvni sekci
LAYERS_2="0"        # pocet vrstev v druhe sekci
BATCH="128"         # pocet vzorku do predikce
DBMODE="True"       # implicitne v debug modu - nezapisuje do PLC
INTERPOLATE="False" # TRUE FALSE -interpolace splinem
LRNRATE="0.007"     # learning rate <0.0002, 0.02>

source="./result"
VYSL="vysl_pdf"
curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`

for i in "$@"; do
  case $i in
     -t=*|--test=*)
        TESTNAME="${i#*=}"
        shift # past argument=value
        ;;
    -u1=*|--units_1=*)
	UNITS_1="${i#*=}"
	shift # past argument=value
        ;;
    -u2=*|--units2=*)
	UNITS_2="${i#*=}"
	shift # past argument=value
        ;;
    -m1=*|--model_1=*)
        MODEL_1="${i#*=}"
        shift # past argument=value
        ;;
    -m2=*|--model_2=*)
        MODEL_2="${i#*=}"
        shift # past argument=value
        ;;
    -e=*|--epochs=*)
        EPOCHS="${i#*=}"
        shift # past argument=value
        ;;
    -l1=*|--layers_1=*)
	LAYERS_1="${i#*=}"
	shift # past argument=value
        ;;
    -l2=*|--layers_2=*)
	LAYERS_2="${i#*=}"
	shift # past argument=value
        ;;
    -b=*|--batch=*)
	BATCH="${i#*=}"
	shift # past argument=value
        ;;
    -db=*|--dbmode=*)
	DBMODE="${i#*=}"
	shift # past argument=value
        ;;
    -ip=*|--interpolate=*)
	INTERPOLATE="${i#*=}"
	shift # past argument=value
        ;;
    -*|--*)
	echo "bash: Neznamy parametr $i"
	ai-help_
	exit 1
        ;;
    *)
      ;;
  esac
done


#hyperparametry 
parms="`cat ./cfg/ai-parms.cfg | grep ^[^#] | grep [X]`"
text=""
text=$text",\n  jmeno testu="$TESTNAME" "
text=$text",\n       status="$STATUS" "
text=$text",\n      units_1="$UNITS_1" "
text=$text",\n      units_2="$UNITS_2" "
text=$text",\n      model_1="$MODEL_1" "
text=$text",\n      model_2="$MODEL_2" "
text=$text",\n        epoch="$EPOCHS" "
text=$text",\n     layers_1="$LAYERS_1" "
text=$text",\n     layers_2="$LAYERS_2" "
text=$text",\n        batch="$BATCH" "
text=$text",\n   debug mode="$DBMODE" "
text=$text",\n  interpolace="$INTERPOLATE" "
text=$text",\n      lrnrate="$LRNRATE" "
text=$text",\n        datum="$curr_timestamp" "
text=$text",\n        parms="$parms" "
text=$text",\n"

RES_TESTNAME="res_"$TESTNAME

if [[ ! -e $source ]]; then 
    mkdir $source
fi

if [[ ! -e $source/$RES_TESTNAME ]]; then 
    mkdir $source/$RES_TESTNAME
fi

LSCSV="./br_data/*.csv"
printf "$text" >> $source/$RES_TESTNAME/parametry.txt
echo "testovaci soubory....................:" >> $source/$RES_TESTNAME/parametry.txt
COUNT=1 # pocet testu
for i in $(seq $START $END); do
    test=$"v$i";
    target="$source/"$TESTNAME$test
    mkdir --parents $target
    rm ./br_data/*.csv
    cp ./br_data/archiv/VALIDACNI_ZKOUSKY/$test"_src"/tm-ai*.csv ./br_data
    cp ./br_data/archiv/VALIDACNI_ZKOUSKY/$test"_src"/predict-debug.csv ./br_data
    ls -l $LSCSV  >>  $source/$RES_TESTNAME/parametry.txt
    echo "testovaci soubory....................:" >> $source/$RES_TESTNAME/parametry.txt
    for j in $(seq 1 $COUNT); do
	./ai-daemon.sh \
	    --status="$STATUS" \
	    --dbmode="$DBMODE"\
	    --model_1="$MODEL_1" \
	    --model_2="$MODEL_2" \
	    --epochs="$EPOCHS" \
	    --batch="$BATCH" \
	    --units_1="$UNITS_1" \
	    --units_2="$UNITS_2" \
	    --layers_1="$LAYERS_1" \
	    --layers_2="$LAYERS_2" \
	    --interpolate="$INTERPOLATE" \
	    --lrnrate="$LRNRATE"
  
    done;    
    echo "move "$source" to "$target
    mv -n -f $source/*.csv $target
done;

./ai-part02.sh $source $TESTNAME $RES_TESTNAME


