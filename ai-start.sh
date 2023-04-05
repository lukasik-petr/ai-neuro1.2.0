#!/bin/bash
FILE_PATH=$(pwd)
export AI_HOME="~/ai/ai-daemon"
#export PYTHONPATH="~/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="~/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0


STATUS="start"      # run,start,....
UNITS_1="280"       # GRU,LSTM=91, DENSE=330
UNITS_2="0"         # GRU,LSTM=91, DENSE=330
MODEL_1="DENSE"     # typ vrstvy_1 LSTM DENSE GRU CONV1D
MODEL_2=""          # typ vrstvy_2 LSTM DENSE GRU CONV1D
EPOCHS="500"        # Poc. treninkovych cyklu
LAYERS_0="True"     # vrstva DENSE ve vstupu <True, False>
LAYERS_1="2"        # pocet vrstev v prvni sekci
LAYERS_2="0"        # pocet vrstev v druhe sekci
BATCH="128"         # pocet vzorku do predikce
DBMODE="False"      # implicitne v debug modu - nezapisuje do PLC
INTERPOLATE="False" # TRUE FALSE -interpolace splinem
LRNRATE="0.0005"    # learning rate <0.0002, 0.002>
SHUFFLE="True"      #True, False
WINDOW="3"          # timeseries window <0,24>
N_IN="0"            # timeseries- n_in <0, 6>
N_OUT="3"           # timeseries+ n_out <0, 6>
RETVAL=0



eval "$(conda shell.bash hook)"
conda activate tf
cd $AI_HOME/src

ai-help() {
    echo " Start v produkcnim modu. Program se uhnizdi v pameti  "
    echo " jako demon a zacne generovat navrhy korekci pro PLC.  "  
    echo " ./ai-start.sh nastavi hlavni hyperparametry pro       "  
    echo " neuronovou sit.                                       "  
    echo "-------------------------------------------------------"
    echo " popis parametru:                                      "  
    echo " -s <--status>     - typ behu programu, muze nabyvat   "
    echo "                     hodnot 'start' nebo 'run'.        "
    echo "                     'start' - spusten jako demon      "
    echo "                     'run'   - spusten jako program    "
    echo "                                                       "
    echo " -db <--debug-mode> - mod debug/nodebug                 "
    echo "                     pozor !!! nodebug je ostry provoz "
    echo "-------------------------------------------------------"
    echo " popis hyperparametru:                                 "  
    echo " pozor jsou implicitne nastaveny pro optimalni beh site"
    echo "    UNITS_1='96'   - pocet neuronu v prvni sekci       "
    echo "    UNITS_2='96'   - pocet neuronu v druhe sekci       "
    echo "    MODEL_1='DENSE'- typ  site  v prvni sekci          "
    echo "    MODEL_2='GRU'  - typ  site  v prvni sekci          "
    echo "                     (DENSE, LSTM, GRU, CONV1D)        "
    echo "    EPOCHS='97'    - pocet treninkovych epoch          "
    echo "    LAYERS_1='2'   - pocet vrstev v prvni sekci        "
    echo "    LAYERS_2='1'   - pocet vrstev v prvni sekci        "
    echo "                                                       "
    echo " pouziti: ./ai-start.sh -s=start -db=nodebug           "
    echo "                                                       "
}

echo "-------------------------------------------------------"
echo " ./ai-start.sh                                         "
echo "-------------------------------------------------------"

for i in "$@"; do
  case $i in
    -s=*|--status=*)
        STATUS="${i#*=}"
        shift # past argument=value
        ;;
   -db=*|--debug-mode=*)
        DBMODE="${i#*=}"
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


echo ""
echo "----------------------------------------------------------------"
echo "Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z"
echo "----------------------------------------------------------------"
./ai-daemon.sh \
  --status="$STATUS" \
  --units_1="$UNITS_1" \
  --units_2="$UNITS_2" \
  --model_1="$MODEL_1" \
  --model_2="$MODEL_2" \
  --epochs="$EPOCHS" \
  --layers_0="$LAYERS_0" \
  --layers_1="$LAYERS_1" \
  --layers_2="$LAYERS_2" \
  --batch="$BATCH" \
  --dbmode="$DBMODE"\
  --interpolate="$INTERPOLATE" \
  --lrnrate="$LRNRATE" \
  --window="$WINDOW" \
  --n_in="$N_IN" \
  --n_out="$N_OUT" \
  --shuffle="$SHUFFLE" 

echo "ai-daemon start..."
exit 0



