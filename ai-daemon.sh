#!/bin/bash
#cd ~/workspaces/eclipse-python-workspace/ai-daemon/src
#cd /home/plukasik/ai/ai-daemon/src
echo "------------------------------------------------------"
echo " ai-daemon.sh "
echo "------------------------------------------------------"


#----------------------------------------------------------------------
# ai-ad     Startup skript pro ai-ad demona
# optimalni parametry
#           DENSE  |  CONV1D |   GRU |  LAYERS |
#----------------------------------------------------------------------
# UNITS    |    79 |   179   |  179  | 3       |
# EPOCHS   |    57 |    67   |   57  | 3       |
# ACTF     |   ELU |   ELU   |  ELU  | 3       |
# SHUFFLE  |  true |  true   | true  | 3       |
#----------------------------------------------------------------------
FILE_PATH=$(pwd)
# Environment Miniconda.
#export PYTHONPATH="/home/plukasik/miniconda3/pkgs:$PYTHONPATH"
export CONDA_HOME="/home/plukasik/miniconda3"
export PATH=${CONDA_HOME}/bin:${PATH}
export TF_ENABLE_ONEDNN_OPTS=0

prog=$FILE_PATH"/ai-daemon.py"
pidfile=${PIDFILE-$FILE_PATH/pid/ai-daemon.pid}
logfile=${LOGFILE-$FILE_PATH/log/ai-daemon.log}

#----------------------------------------------------------------------
# implicitni parametry - mozno zmenit  z 'cmd lajny'
#----------------------------------------------------------------------
STATUS="run"        # run,start,....
UNITS_1="330"       # GRU,LSTM=91, DENSE=330
UNITS_2="0"         # GRU,LSTM=91, DENSE=330
MODEL_1="DENSE"     # typ vrstvy_1 LSTM DENSE GRU CONV1D
MODEL_2="GRU"       # typ vrstvy_2 LSTM DENSE GRU CONV1D
EPOCHS="500"        # Poc. treninkovych cyklu
LAYERS_1="2"        # pocet vrstev v prvni sekci
LAYERS_2="0"        # pocet vrstev v druhe sekci
BATCH="128"         # pocet vzorku do predikce
DBMODE="True"       # implicitne v debug modu - nezapisuje do PLC
INTERPOLATE="False" # TRUE FALSE -interpolace splinem
LRNRATE="0.001"     # learning rate <0.0002, 0.002>
SHUFFLE="False"     #True, False
WINDOW="3"          # timeseries window <0,24>
N_IN="0"            # timeseries- n_in <0, 6>
N_OUT="3"           # timeseries+ n_out <0, 6>
RETVAL=0

#----------------------------------------------------------------------
# parametry z  'cmd lajny'
#----------------------------------------------------------------------
for i in "$@"; do
  case $i in
     -s=*|--status=*)
        STATUS="${i#*=}"
        shift # past argument=value
        ;;
    -u1=*|--units_1=*)
	UNITS_1="${i#*=}"
	shift # past argument=value
        ;;
    -u2=*|--units_2=*)
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
    -lr=*|--lrnrate=*)
	LRNRATE="${i#*=}"
	shift # past argument=value
        ;;
    -w=*|--window=*)
	WINDOW="${i#*=}"
	shift # past argument=value
        ;;
    -ni=*|--n_in=*)
	N_IN="${i#*=}"
	shift # past argument=value
        ;;
    -no=*|--n_out=*)
	N_OUT="${i#*=}"
	shift # past argument=value
        ;;
    -sh=*|--shuffle=*)
	SHUFFLE="${i#*=}"
	shift # past argument=value
        ;;
      -*|--*)
	echo "bash: Neznamy parametr $i"
	ai-help
	exit 1
        ;;
    *)
      ;;
  esac
done

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# implicitni parametry - nejsou zadavany z 'cmd lajny'
#----------------------------------------------------------------------
# selu     - vyborna Z (z, RMSE=3) dobra Y (y, RMSE=6)
# sigmoid  - horsi predikce (Y,Z, RMSE=8)
# elu      - dobre vysledky (Y,Z, RMSE=5)
# relu     - dobra Z (Z, RMSE=5) horsi Y (y, RMSE=8)
# softmax  - nepouzitelna (RMSE > 50)
# tanh     - dobra Z (Z, RMSE=4) horsi Y (y, RMSE=8)
# softsign - dobra Z (Z, RMSE=4) horsi Y (y, RMSE=8), dlouhy trenink
# softsign - spatna  (Z, RMSE=5) spatnaY (y, RMSE=9),
# swish    - dobra Z (Z, RMSE=4) horsi Y (y, RMSE=8), dlouhy trenink
#----------------------------------------------------------------------
ACTF="selu"  #elu, relu, sigmoid ....
TXDAT1="2022-01-01 00:00:01"
TXDAT2=`date +%Y-%m-%d -d "yesterday"`" 23:59:59"
OPTIONS=""
ILCNT="1"           #1 - 8

echo "bash: Spusteno s parametry:"  
echo "      STATUS="$STATUS
echo "     UNITS_1="$UNITS_1
echo "     UNITS_2="$UNITS_2
echo "     MODEL_1="$MODEL_1
echo "     MODEL_2="$MODEL_2
echo "      EPOCHS="$EPOCHS
echo "    LAYERS_1="$LAYERS_1
echo "    LAYERS_2="$LAYERS_2
echo "       BATCH="$BATCH
echo "      DBMODE="$DBMODE
echo " INTERPOLATE="$INTERPOLATE
echo "     SHUFFLE="$SHUFFLE
echo "        ACTF="$ACTF
echo "     LRNRATE="$LRNRATE
echo "      TXDAT1="$TXDAT1
echo "      TXDAT2="$TXDAT2
echo "       ILCNT="$ILCNT
echo "      WINDOW="$WINDOW
echo "        N_IN="$N_IN
echo "       N_OUT="$N_OUT

#----------------------------------------------------------------------
# start_daemon - aktivace miniconda a start demona
#----------------------------------------------------------------------
start_daemon(){
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo ""
    echo "----------------------------------------------------------------"
    echo "Demon pro kompenzaci teplotnich anomalii na stroji pro osy X,Y,Z"
    echo "Start ulohy: "$curr_timestamp
    echo "Treninkova mnozina v rozsahu: "$TXDAT1" : "$TXDAT2
    echo "----------------------------------------------------------------"
    eval "$(conda shell.bash hook)"
    conda activate tf
    python3 ./py-src/ai-daemon.py \
	    --status="$STATUS" \
	    --dbmode="$DBMODE"\
            --pidfile="$pidfile" \
	    --logfile="$logfile" \
	    --model_1="$MODEL_1" \
	    --model_2="$MODEL_2" \
	    --epochs="$EPOCHS" \
	    --batch="$BATCH" \
	    --units_1="$UNITS_1" \
	    --units_2="$UNITS_2" \
	    --layers_1="$LAYERS_1" \
	    --layers_2="$LAYERS_2" \
	    --actf="$ACTF" \
	    --txdat1="$TXDAT1" \
	    --txdat2="$TXDAT2" \
	    --ilcnt="$ILCNT" \
	    --shuffle="$SHUFFLE" \
	    --interpolate="$INTERPOLATE" \
	    --lrnrate="$LRNRATE" \
	    --window="$WINDOW" \
	    --n_in="$N_IN" \
	    --n_out="$N_OUT" 
    
    conda deactivate
    curr_timestamp=`date "+%Y-%m-%d %H:%M:%S"`
    echo "ai-daemon start: "$curr_timestamp

}

#----------------------------------------------------------------------
# start
#----------------------------------------------------------------------
start() {
        echo -n $"Starting $prog: as daemon... "

        if [[ -f ${pidfile} ]] ; then
            pid=$( cat $pidfile  )
            isrunning=$( ps -elf | grep  $pid | grep $prog | grep -v grep )

            if [[ -n ${isrunning} ]] ; then
                echo $"$prog already running"
                return 0
            fi
        fi
	start_daemon
        RETVAL=$?
        [ $RETVAL = 0 ]
        echo
        return $RETVAL
}


#----------------------------------------------------------------------
# run 
#----------------------------------------------------------------------
run() {
        echo -n $"Starting $prog: "

        if [[ -f ${pidfile} ]] ; then
            pid=$( cat $pidfile  )
            isrunning=$( ps -elf | grep  $pid | grep $prog | grep -v grep )

            if [[ -n ${isrunning} ]] ; then
                echo $"$prog already running"
                return 0
            fi
        fi
	start_daemon
        RETVAL=$?
        [ $RETVAL = 0 ]
        echo
        return $RETVAL
}


#----------------------------------------------------------------------
# stop  
#----------------------------------------------------------------------
stop() {
    if [[ -f ${pidfile} ]] ; then
        pid=$( cat $pidfile )
        isrunning=$( ps -elf | grep $pid | grep $prog | grep -v grep | awk '{print $4}' )

        if [[ ${isrunning} -eq ${pid} ]] ; then
            echo -n $"Stop $prog: "
            kill $pid
	    rm -f $pidfile 
        else
            echo -n $"STOP $prog: "
        fi
        RETVAL=$?
    fi
    echo
    return $RETVAL
}

#----------------------------------------------------------------------
# reload
#----------------------------------------------------------------------
reload() {
    echo -n $"Reloading $prog: "
    echo
}


#----------------------------------------------------------------------
# help   
#----------------------------------------------------------------------
ai-help_() {
      echo "--------------------------------------------------------------------------------"
      echo "Hlavni parametry"
      echo "--------------------------------------------------------------------------------"
      echo "            -s<--status>  -spusteno v rezimu program nebo demon "
      echo "                           muze nabyvat hodnot:                    "
      echo "                           start|run|stop|restart|force-reload|reload|status"
      echo ""
      echo ""
      echo "          -u1<--units_1>  -pocet neuronu v prvni sekci skryte vrstvy <32,1024>"
      echo "                           <0, 1024>  u1=0 - prvni sekce hidden vrstvy disable"
      echo ""
      echo ""
      echo "          -u2<--units_2>  -pocet neuronu v druhe sekci skryte vrstvy"
      echo "                           <0, 1024>  u2=0 - druha sekce hidden vrstvy disable"
      echo ""
      echo ""
      echo "          -m1<--model_1>  -typ prvni sekce skryte vrstvy site:"
      echo "                             'DENSE' - sit typu DENSE - zakladni model"
      echo "                             'GRU'   - sit typu GRU "
      echo "                             'LSTM'  - sit typu LSTM"
      echo "                             'CONV1D'- sit typu Konvoluce"
      echo ""
      echo ""
      echo "          -m2<--model_2>  -typ druhe sekce skryte vrstvy site:"
      echo "                             'DENSE' - sit typu DENSE - zakladni model"
      echo "                             'GRU'   - sit typu GRU "
      echo "                             'LSTM'  - sit typu LSTM"
      echo "                             'CONV1D'- sit typu Konvoluce"
      echo "                             ''      - druha vrstva disable..."
      echo "                                         "
      echo "           -e <--epochs>  -pocet epoch"
      echo "                           <64, 1024>"
      echo ""
      echo ""
      echo "            -b <--batch>  -delka tenzoru vstupujiciho do predikce"
      echo "                           optimalni velikost batch je v intervalu"
      echo "                           <32,2048>.                             "
      echo ""
      echo "--------------------------------------------------------------------------------"
      echo "Vedlejsi parametry"
      echo "--------------------------------------------------------------------------------"
      echo ""
      echo "          -sh<--shuffle>  -data budou pred rozdelenim na treninkova   "
      echo "                           a validacni shufflovana"
      echo "                           <True, False>"
      echo ""
      echo ""
      echo "            -w<--window>  -velikost okna pro 3D tenzor"
      echo "                           <0,24>"
      echo ""
      echo ""
      echo "             -ni<--n_in>  -timeseries- (minus)"
      echo "                           <0, 6>"
      echo ""
      echo ""
      echo "            -no<--n_out>  -timeseries+ (plus)"
      echo "                           <0, 6>"
      echo ""
      echo ""
      echo "      -ip<--interpolate>  -interpolace univariantni splajnou"
      echo "                           <True,False>"
      echo ""
      echo "          -lr<--lrnrate>  -rychlost uceni "
      echo "                           <0.0001, 0.02>"
      echo ""
      echo ""
      echo "PRIKLAD: ./ai-daemon.sh -s=run -m1=DENSE -e=64 -b=128 -u1=72"
}

	    --shuffle="$SHUFFLE" \
	    --interpolate="$INTERPOLATE" \
	    --lrnrate="$LRNRATE" \
	    --window="$WINDOW" \
	    --n_in="$N_IN" \
	    --n_out="$N_OUT" 


#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------
case "$STATUS" in
  start)
      start
    ;;
  stop)
    stop
    ;;
  run)
    run
    ;;
  status)
    status -p $pidfile $eg_daemon
    RETVAL=$?
    ;;
  restart)
    stop
    start
    ;;
  force-reload|reload)
    reload
    ;;
  *)
    echo $"Usage: $prog {start|run|stop|restart|force-reload|reload|status}"
    RETVAL=2
esac
exit $RETVAL

