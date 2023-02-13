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
UNITS_1="79"        # LSTM=91, DENSE=79
UNITS_2="0"         # LSTM=91, DENSE=79
MODEL_1="DENSE"     # typ vrstvy_1 LSTM DENSE GRU CONV1D
MODEL_2=""          # typ vrstvy_2 LSTM DENSE GRU CONV1D
EPOCHS="49"         # Poc. treninkovych cyklu
LAYERS_1="2"        # pocet vrstev v prvni sekci
LAYERS_2="0"        # pocet vrstev v druhe sekci
BATCH="128"         # pocet vzorku do predikce
DBMODE="True"       # implicitne v debug modu - nezapisuje do PLC
INTERPOLATE="False" # TRUE FALSE -interpolace splinem
LRNRATE="0.0005"    # learning rate <0.0002, 0.002>
RETVAL=0

#----------------------------------------------------------------------
# modifikace parametru z  'cmd lajny'
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
# implicitni parametry - nejsou zadavany z 'cmd lajny'
#----------------------------------------------------------------------
ACTF="elu"          #elu , relu, sigmoid ....
TXDAT1="2022-01-01 00:00:01"
TXDAT2=`date +%Y-%m-%d -d "yesterday"`" 23:59:59"
OPTIONS=""
ILCNT="1"           #1 - 8
SHUFFLE="True"      #True, False

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
	    --lrnrate="$LRNRATE" 
    
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
      echo "  parametry  -s<--status>  spusteno v rezimu program nebo demon "
      echo "                           muze nabyvat hodnot:                    "
      echo "                           start|run|stop|restart|force-reload|reload|status"
      echo ""
      echo ""
      echo "            -u1<--units_1> pocet neuronu v prvni sekci skryte vrstvy <32,1024>"
      echo ""
      echo ""
      echo "            -u2<--units_2> pocet neuronu v druhe sekci skryte vrstvy <0,1024>"
      echo "                              -u2 = 0 - druha sekce skryte vrstvy disable...  "
      echo ""
      echo ""
      echo "            -m1<--model_1> typ prvni sekce skryte vrstvy site:"
      echo "                             'DENSE' - sit typu DENSE - zakladni model"
      echo "                             'GRU'   - sit typu GRU "
      echo "                             'LSTM'  - sit typu LSTM"
      echo "                             'CONV1D'- sit typu Konvoluce"
      echo ""
      echo ""
      echo "            -m2<--model_2> typ druhe sekce skryte vrstvy site:"
      echo "                             'DENSE' - sit typu DENSE - zakladni model"
      echo "                             'GRU'   - sit typu GRU "
      echo "                             'LSTM'  - sit typu LSTM"
      echo "                             'CONV1D'- sit typu Konvoluce"
      echo "                             ''      - druha vrstva disable..."
      echo "                                         "
      echo "            -e <--epochs> -pocet treninkovych epoch <64,128>"
      echo ""
      echo ""
      echo "            -b <--batch>  -velikost vzorku dat  <32,2048>"
      echo "                           optimalni velikost batch je v intervalu"
      echo "                           cca.<32,2048>.                         "
      echo "                           delka tenzoru vstupujiciho do predikce"
      echo ""
      echo ""
      echo "PRIKLAD: ./ai-daemon.sh -t=predict -m1=DENSE -e=64 -b=128 -u1=72"
}


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

