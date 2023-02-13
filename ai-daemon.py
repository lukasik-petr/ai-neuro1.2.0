#!/usr/bin/python3
#------------------------------------------------------------------------
#    ai-daemon V.1.2.0
#   
#    predikce vlivu teploty na presnost obrabeni
#
#    @author plukasik
#    
#    Tento program je volny software; muzete jej sirit a modifikovat podle
#    ustanoveni Obecne verejne licence GNU, vydavane Free Software
#    Foundation; a to bud verze 2 teto licence anebo (podle vaseho
#    uvazeni) kterekoli pozdejsi verze.
#    
#    Tento program je rozsirovan v nadeji, ze bude uzitecny, avsak BEZ
#    JAKEKOLI ZARUKY; neposkytuji se ani odvozene zaruky PRODEJNOSTI anebo
#    VHODNOSTI PRO URCITY UCEL. Dalsi podrobnosti hledejte ve Obecne
#    verejne licenci GNU.
#    
#    Kopii Obecne verejne licence GNU jste meli obdrzet spolu s timto
#    programem; pokud se tak nestalo, napiste o ni Free Software
#    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
#    
#    Program je ABSOLUTNE BEZ ZARUKY. Jde o volny software a jeho sireni za
#    jistych podminek je vitano. pripominky na plukasik@tajmac-zps.cz
#    
#    Tento produkt obsahuje software vyvijeny v ramci Apache Software
#    Foundation <http://www.apache.org/>.
#    
#    pro potreby projektu TM-AI vyrobil Petr Lukasik , 2022
#    plukasik@tajmac-zps.cz
#
#------------------------------------------------------------------------
#    Prerekvizity: linux Debian-11 nebo Ubuntu-20.04,);
#               miniconda3,
#               python 3.9,
#               tensorflow 2.8,
#               mathplotlib,
#               scikit-learn-intelex,
#               pandas,
#               numpy,
#               keras,
#
#    Pozor pro instalaci je nutno udelat nekolik veci
#     1. instalace prostredi miniconda 
#          a. stahnout z webu miniconda3 v nejnovejsi verzi
#          b. chmod +x Miniconda3-py39_4.11.0-Linux-x86_64.sh
#          c. ./Miniconda3-py39_4.11.0-Linux-x86_64.sh
#   
#     2. update miniconda
#          conda update conda
#   
#     3. vyrobit behove prostredi 'tf' v miniconda
#          conda create -n tf python=3.9
#   
#     4. aktivovat behove prostredi tf (preX tim je nutne zevrit a znovu
#        otevrit terminal aby se conda aktivovala.
#          conda activate  tf
#   
#     5. instalovat tyto moduly 
#          conda install tensorflow=2.9 
#          conda install tf-nightly=2.9 
#          conda install mathplotlib
#          conda install scikit-learn-intelex
#          conda install pandas
#          conda install numpy
#          conda install keras
#          conda install lockfile
#          conda install pickle
#   
#     6. v prostredi tf jeste upgrade tensorflow 
#          pip3 install --upgrade tensorflow
#------------------------------------------------------------------------
# import vseho co souvisi s demonem...
#------------------------------------------------------------------------
import sys;
import os;
import getopt;
import errno; 
import traceback;
import time; 
import atexit; 
import signal; 
import socket;


try:
    import daemon, daemon.pidfile;
except ImportError:
    daemon = None

import lockfile;
import random;
import webbrowser;
import glob as glob;
import pandas as pd;
import seaborn as sns;
import tensorflow as tf;
import math;
import numpy as np;
import shutil;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
#mpl.use('Agg');
import platform;
import pandas.api.types as ptypes;
import pickle;

from matplotlib import cm;
from os.path import exists;

from dateutil import parser
from sklearn.preprocessing import MinMaxScaler;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import max_error;
from sklearn.utils import shuffle;
from sklearn.utils import assert_all_finite;
from numpy import asarray;

from dataclasses import dataclass;
from datetime import datetime, timedelta, timezone;
from tabulate import tabulate;
from pathlib import Path;
from pandas.core.frame import DataFrame;
from pandas import concat;
from concurrent.futures import ThreadPoolExecutor;
from signal import SIGTERM;

from tensorflow.keras import models;
from tensorflow.keras import layers;
from tensorflow.keras import optimizers;
from tensorflow.keras import losses;
from tensorflow.keras import metrics;
from tensorflow.keras import callbacks;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import InputLayer;
from tensorflow.keras.layers import Input;
from tensorflow.keras.layers import Dense;
from tensorflow.keras.layers import LSTM;
from tensorflow.keras.layers import GRU;
from tensorflow.keras.layers import Conv1D;
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

from _cffi_backend import string
#scipy
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import InterpolatedUnivariateSpline;
from scipy.interpolate import UnivariateSpline;
from scipy.interpolate import interp1d;

#opcua
from opcua import ua
from opcua import *
from opcua.common.ua_utils import data_type_to_variant_type

from subprocess import call;
from plistlib import InvalidFileException

try:
    from tensorflow.python.eager.function import np_arrays
except ImportError:
    np_arrays = None
    
from pandas.errors import EmptyDataError

try:
    from keras.saving.utils_v1.mode_keys import is_train
except ImportError:
    is_train = None

#logger
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from logging import handlers

#------------------------------------------------------------------------
# nastaveni globalnich parametru logu pro demona
#------------------------------------------------------------------------
logger = None;
log_handler = None;
df_debug_count=0;
df_debug_header=[];

#------------------------------------------------------------------------
# OPCAgent
#------------------------------------------------------------------------
class OPCAgent():

    @dataclass
    class PLCData:
    # osy XYZ
        CompX:     object              #Aktualni hodnota kompenzace v ose X
        CompY:     object              #Aktualni hodnota kompenzace v ose Y
        CompZ:     object              #Aktualni hodnota kompenzace v ose Z
    # osy rotace AC
        CompA:     object              #Aktualni hodnota kompenzace v ose A
        CompC:     object              #Aktualni hodnota kompenzace v ose C
        
    # osy XYZ
        setCompX:     object           #Predikovana hodnota kompenzace v ose X
        setCompY:     object           #Predikovana hodnota kompenzace v ose Y
        setCompZ:     object           #Predikovana hodnota kompenzace v ose Z
    # osy rotace AC
        setCompA:     object           #Predikovana hodnota kompenzace v ose A
        setCompC:     object           #Predikovana hodnota kompenzace v ose C
        
    
    # konstrukter    
    def __init__(self, batch, debug_mode):
        
        self.logger     = logging.getLogger("ai");
        self.prefix     = "opc.tcp://";
        self.host1      = "opc998.os.zps"; # BR-PLC
        self.port1      = "4840";
        self.host2      = "opc999.os.zps";# HEIDENHANIN-PLC
        self.port2      = "48010";
        self.is_ping    = False;
        self.batch      = batch;
        self.debug_mode = debug_mode;
        
        self.df_debug   = pd.DataFrame();
        self.uri1       = self.prefix+self.host1+":"+self.port1;
        self.uri2       = self.prefix+self.host2+":"+self.port2;
        
        self.plc_timer  = 4 #[s];

#------------------------------------------------------------------------
# ping_         
#------------------------------------------------------------------------
    def pingSys(self, host, port):
        parameter = '-n' if platform.system().lower()=='windows' else '-c';
        command = ['ping', parameter, '1', host];
        response = call(command);
        if response == 0:    
            self.logger.debug("pingSys: %s %d ok..." %(host, port));
            self.is_ping = True;
            return self.is_ping;
        else:
            self.logger.debug("pingSys: %s %d is not responding..." %(host, port));
            self.is_ping = False;
            return self.is_ping;
        
#------------------------------------------------------------------------
# pingSocket - nepotrebuje root prava...
#------------------------------------------------------------------------
    def pingSocket(self, host, port):

        self.is_ping = False;
        ping_cnt = 0;


        socket_ = None;
    # Loop while less than max count or until Ctrl-C caught
        while ping_cnt < 2:
            ping_cnt += 1;    
            try:
                # New Socket
                socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
                socket_.settimeout(1);
                # Try to Connect
                socket_.connect((host, int(port)));
                self.logger.debug("pingSocket: %s ok..." %(host));
                self.is_ping = True;
                return(self.is_ping);

            except socket.error as err:
                self.logger.debug("pingSocket: failed with error %s %s" %(host, err));
                self.is_ping = False;
                return(self.is_ping);

                # Connect TimeOut
            except socket.timeout as err:
                self.logger.debug("pingSocket: socket.timeout %s %s" %(host, err));
                self.is_ping = False;
                return(self.is_ping);
                
                # OS error
            except OSError as err:
                self.logger.debug("pingSocket: OSError %s %s" %(host, err));
                self.is_ping = False;
                return(self.is_ping);

                # Connect refused
            except ConnectionRefusedError as err:    
                self.logger.debug("pingSocket: ConnectionRefusedError %s %s" %(host, err));
                self.is_ping = False;
                return(self.is_ping);
                
                # Other error
            except:    
                self.logger.debug("pingSocket: OtherError  %s %s" %(host, err));
                self.is_ping = False;
                return(self.is_ping);

            finally:
                #socket_.shutdown(socket.SHUT_RD);
                socket_.close();

        # end while            

        return(self.is_ping);    
                                                                                                            
        
#------------------------------------------------------------------------
# opcCollectorBR_PLC - opc server BR
#------------------------------------------------------------------------
    def opcCollectorBR_PLC(self):
        
        plc_isRunning = True;
        # tabulka nodu v br plc         
        plc_br_table        = np.array([["temp_ch01",     "ns=6;s=::AsGlobalPV:teplota_ch01"],
                                        ["temp_lo01",     "ns=6;s=::AsGlobalPV:teplota_lo01"],
                                        ["temp_lo03",     "ns=6;s=::AsGlobalPV:teplota_lo03"],
                                        ["temp_po01",     "ns=6;s=::AsGlobalPV:teplota_po01"],
                                        ["temp_pr01",     "ns=6;s=::AsGlobalPV:teplota_pr01"],
                                        ["temp_pr02",     "ns=6;s=::AsGlobalPV:teplota_pr02"],
                                        ["temp_pr03",     "ns=6;s=::AsGlobalPV:teplota_pr03"],
                                        ["temp_sl01",     "ns=6;s=::AsGlobalPV:teplota_sl01"],
                                        ["temp_sl02",     "ns=6;s=::AsGlobalPV:teplota_sl02"],
                                        ["temp_sl03",     "ns=6;s=::AsGlobalPV:teplota_sl03"],
                                        ["temp_sl04",     "ns=6;s=::AsGlobalPV:teplota_sl04"],
                                        ["temp_st01",     "ns=6;s=::AsGlobalPV:teplota_st01"],
                                        ["temp_st02",     "ns=6;s=::AsGlobalPV:teplota_st02"],
                                        ["temp_st03",     "ns=6;s=::AsGlobalPV:teplota_st03"],
                                        ["temp_st04",     "ns=6;s=::AsGlobalPV:teplota_st04"],
                                        ["temp_st05",     "ns=6;s=::AsGlobalPV:teplota_st05"],
                                        ["temp_st06",     "ns=6;s=::AsGlobalPV:teplota_st06"],
                                        ["temp_st07",     "ns=6;s=::AsGlobalPV:teplota_st07"],
                                        ["temp_st08",     "ns=6;s=::AsGlobalPV:teplota_st08"],
                                        ["temp_vr01",     "ns=6;s=::AsGlobalPV:teplota_vr01"],
                                        ["temp_vr02",     "ns=6;s=::AsGlobalPV:teplota_vr02"],
                                        ["temp_vr03",     "ns=6;s=::AsGlobalPV:teplota_vr03"],
                                        ["temp_vr04",     "ns=6;s=::AsGlobalPV:teplota_vr04"],
                                        ["temp_vr05",     "ns=6;s=::AsGlobalPV:teplota_vr05"],
                                        ["temp_vr06",     "ns=6;s=::AsGlobalPV:teplota_vr06"],
                                        ["temp_vr07",     "ns=6;s=::AsGlobalPV:teplota_vr07"],
                                        ["temp_vz02",     "ns=6;s=::AsGlobalPV:teplota_vz02"],
                                        ["temp_vz03",     "ns=6;s=::AsGlobalPV:teplota_vz03"],
                                        ["light_ambient", "ns=6;s=::AsGlobalPV:vstup_osvit"],
                                        ["temp_ambient",  "ns=6;s=::AsGlobalPV:vstup_teplota"],
                                        ["humid_ambient", "ns=6;s=::AsGlobalPV:vstup_vlhkost"]]);

        if not self.pingSocket(self.host1, self.port1):
            plc_isRunning = False;
            return(plc_br_table, plc_isRunning);
   
        client = Client(self.uri1)
        try:        
            client.connect();
            self.logger.debug("Client: " + self.uri1 + " connect.....")
            plc_br_table = np.c_[plc_br_table, np.zeros(len(plc_br_table))];
            
            for i in range(len(plc_br_table)):
                node = client.get_node(str(plc_br_table[i, 1]));
                typ  = type(node.get_value());
                val = float(self.myFloatFormat(node.get_value())) if typ is float else node.get_value();
                #val = node.get_value() if typ is float else node.get_value();
                plc_br_table[i, 2] = val;

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect();
 
        return(plc_br_table, plc_isRunning);


#------------------------------------------------------------------------
# opcCollectorHH_PLC - opc server PLC HEIDENHAIN
#------------------------------------------------------------------------
    def opcCollectorHH_PLC(self):
        
        plc_isRunning = True;
        #tabulka nodu v plc heidenhain
        plc_hh_table        = np.array([["datetime",     ""],
                                        ["tool",         "ns=2;s=Technology data.ACTUAL_TOOL_T"],
                                        ["state",        "ns=2;s=Technology data.PROGRAM_STATE"],
                                        ["program",      "ns=2;s=Technology data.PIECE_PROGRAM"],
                                        ["load_s1",      "ns=2;s=S1.Load"],
                                        ["mcs_s1",       "ns=2;s=S1.MCS"],
                                        ["speed_s1",     "ns=2;s=S1.Motor_speed"],
                                        ["temp_s1",      "ns=2;s=S1.Temperature"],
                                        ["load_x",       "ns=2;s=X.Load"],
                                        ["mcs_x",        "ns=2;s=X.MCS"],
                                        ["speed_x",      "ns=2;s=X.Motor_speed"],
                                        ["temp_x",       "ns=2;s=X.Temperature"],
                                        ["load_y",       "ns=2;s=Y.Load"],
                                        ["mcs_y",        "ns=2;s=Y.MCS"],
                                        ["speed_y",      "ns=2;s=Y.Motor_speed"],
                                        ["temp_y",       "ns=2;s=Y.Temperature"],
                                        ["load_z",       "ns=2;s=Z.Load"],
                                        ["mcs_z",        "ns=2;s=Z.MCS"],
                                        ["speed_z",      "ns=2;s=Z.Motor_speed"],
                                        ["temp_z",       "ns=2;s=Z.Temperature"],
                                        ["dev_datetime1","ns=2;s=Machine data.M0111"],
                                        ["dev_x1",       "ns=2;s=Machine data.M0112"],
                                        ["dev_y1",       "ns=2;s=Machine data.M0113"],
                                        ["dev_z1",       "ns=2;s=Machine data.M0114"],
                                        ["dev_datetime2","ns=2;s=Machine data.M0211"],
                                        ["dev_x2",       "ns=2;s=Machine data.M0212"],
                                        ["dev_y2",       "ns=2;s=Machine data.M0213"],
                                        ["dev_z2",       "ns=2;s=Machine data.M0214"],
                                        ["dev_datetime3","ns=2;s=Machine data.M0311"],
                                        ["dev_x3",       "ns=2;s=Machine data.M0312"],
                                        ["dev_y3",       "ns=2;s=Machine data.M0313"],
                                        ["dev_z3",       "ns=2;s=Machine data.M0314"],
                                        ["dev_datetime4","ns=2;s=Machine data.M0411"],
                                        ["dev_x4",       "ns=2;s=Machine data.M0412"],
                                        ["dev_y4",       "ns=2;s=Machine data.M0413"],
                                        ["dev_z4",       "ns=2;s=Machine data.M0414"],
                                        ["dev_datetime5",""],
                                        ["dev_x5",       ""],
                                        ["dev_y5",       ""],
                                        ["dev_z5",       ""]]);
                                        
        if not self.pingSocket(self.host2, self.port2):
            plc_isRunning = False;
            return(plc_hh_table, plc_isRunning);
                                                    
        client = Client(self.uri2);
        try:        
            client.connect();
            self.logger.debug("Client: " + self.uri2+ " connect.....")
            
            plc_hh_table = np.c_[plc_hh_table, np.zeros(len(plc_hh_table))];
            
            for i in range(len(plc_hh_table)):
                if "datetime" in plc_hh_table[i, 0]:
                    plc_hh_table[i, 2] = datetime.now().strftime("%Y-%m-%d %H:%M:%S");
                else:
                    if plc_hh_table[i, 1]:
                        node = client.get_node(str(plc_hh_table[i, 1]));
                        typ  = type(node.get_value());
                        val = float(self.myFloatFormat(node.get_value())) if typ is float else node.get_value();
                        #val = node.get_value() if typ is float else node.get_value();
                        plc_hh_table[i, 2] = val;
            
        except OSError as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect(); 


        return(plc_hh_table, plc_isRunning);


#------------------------------------------------------------------------
# opcCollectorSendToPLC - zapis kompenzacni parametry do PLC HEIDENHAIN
#------------------------------------------------------------------------
# OPC Strom: TM-AI
#              +---Compensation
#                   +---CompX         
#                   +---CompY         
#                   +---CompZ         
#                   +---setCompX         
#                   +---setCompY         
#                   +---setCompZ         
#                   +---write_comp_val_TM_AI         
#------------------------------------------------------------------------
    def opcCollectorSendToPLC(self, df_plc):

        if self.debug_mode is True:
            self.logger.error("POZOR Nezapisujeme do PLC, jsme v DEBUG modu !!!");
            return;
        
        plc_isRunning = True;
        
        uri = self.prefix+self.host2+":"+self.port2;
        plcData = self.PLCData;

        if not self.isPing():          
            plc_isRunning = False;
            return plc_isRunning;
        
        client = Client(self.uri2);
        try:        
            client.connect();
                
            root = client.get_root_node();
            # Nacti aktualni hodnoty kompenzace CompX, CompY, CompZ
            # get: CompX                
            node = client.get_node("ns=2;s=Machine data.CompX");
            plcData.CompX = node.get_value();
            
            # get: CompY                
            node = client.get_node("ns=2;s=Machine data.CompY");
            plcData.CompY = node.get_value();
            
            # get: CompZ                
            node = client.get_node("ns=2;s=Machine data.CompZ");
            plcData.CompZ = node.get_value();
            
            # Zapis aktualni hodnoty kompenzace CompX, CompY, CompZ
            plcData.setCompX = int(df_plc[df_plc.columns[1]][0]);
            plcData.setCompY = int(df_plc[df_plc.columns[2]][0]);
            plcData.setCompZ = int(df_plc[df_plc.columns[3]][0]);
            
            
            node_x = client.get_node("ns=2;s=Machine data.setCompX");
            node_x.set_value(ua.DataValue(ua.Variant(plcData.setCompX, ua.VariantType.Int32)));
            
            node_y = client.get_node("ns=2;s=Machine data.setCompY");
            node_y.set_value(ua.DataValue(ua.Variant(plcData.setCompY, ua.VariantType.Int32)));
            
            node_z = client.get_node("ns=2;s=Machine data.setCompZ");
            node_z.set_value(ua.DataValue(ua.Variant(plcData.setCompZ, ua.VariantType.Int32)));
                                
            # Aktualizuj hodnoty v PLC - ns=2;s=Machine data.write_comp_val_TM_AI
            parent = client.get_node("ns=2;s=Machine data")
            method = client.get_node("ns=2;s=Machine data.write_comp_val_TM_AI");
            parent.call_method(method); 
            
            
            
            # Nacti aktualni hodnoty kompenzace CompX, CompY, CompZ
            # get: CompX                
            node = client.get_node("ns=2;s=Machine data.CompX");
            plcData.CompX = node.get_value();
            
            # get: CompY                
            node = client.get_node("ns=2;s=Machine data.CompY");
            plcData.CompY = node.get_value();
            
            # get: CompZ                
            node = client.get_node("ns=2;s=Machine data.CompZ");
            plcData.CompZ = node.get_value();
            return plc_isRunning;
            
        
        except OSError as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            return plc_isRunning;
            
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            plc_isRunning = False;
            return plc_isRunning;
        
        finally:
            client.disconnect();
                
    
#------------------------------------------------------------------------
# prepJitter,  setJitter
#------------------------------------------------------------------------
#    v pripade ze se v prubehu cteciho cykluz OPC zadna data nemeni 
#    pridame jim umele trochu sumu, ktery nezhorsi presnost ale umozni
#    neuronove siti presnejsi predikci. Sit se prilis nedokaze vyrovnat 
#    s konstantnim prubehem datove sady.
#------------------------------------------------------------------------
    def prepJitter(self, df_size):
        
        jitter = np.random.normal(0, .0005, df_size);
        for i in range(len(jitter)):
            jitter[i] = self.myFloatFormat(jitter[i]);
        return jitter;    
        #return(pd.DataFrame(jitter, columns=["jitter"]));

#------------------------------------------------------------------------
# prepJitter,  setJitter
#------------------------------------------------------------------------
    def setJitter(self, df, df_parms, jitter_=False):
        
        if not jitter_:
            return(df);
        
        df_size = len(df);
        
        for col in df.head():
            if  col in df_parms and ptypes.is_numeric_dtype(df[col]):
                jitter = self.prepJitter(df_size);
                df[col].apply(lambda x: np.asarray(x) + np.asarray(jitter));
        return(df);
    
    

#------------------------------------------------------------------------
# opcCollectorGetPLCdata
#------------------------------------------------------------------------
#    opcCollectorGetPLCdata - nacti PLC HEIDENHAIN + PLC BR
#    zapis nactena data do br_data - rozsireni treninkove mnoziny 
#    o data z minulosti
#------------------------------------------------------------------------
    def opcCollectorGetPlcData(self, df_parms):

        plc_isRunning = True;

        if not self.isPing():
            plc_isRunning = False;
            return(None);

        # zapis dat z OPC pro rozsireni treninkove mnoziny        
        current_day =  datetime.now().strftime("%Y-%m-%d");
        path_to_df = "./br_data/getplc/tm-ai_"+current_day+".csv";
        
        
        self.logger.info("Nacitam "+str(self.batch)+" vz.: " +str(self.plc_timer)+ "[s]");
        for i in range(self.batch):
            self.logger.debug("Nacitam vzorek c. %d" %(i));
            br_plc, plc_isRunning = self.opcCollectorBR_PLC();
            if not plc_isRunning:
                self.logger.info("GetPLCData - BR_PLC off -return");
                return(None);
            
            hh_plc, plc_isRunning = self.opcCollectorHH_PLC();
            if not plc_isRunning:
                self.logger.info("GetPLCData - HH_PLC off -return");
                return(None);
        
            hh_plc = np.concatenate((hh_plc, br_plc)).T;
            cols = np.array(hh_plc[0]);
            data = list((hh_plc[2]));
            
            if i == 0:
                df_predict = pd.DataFrame(columns = cols);
                
            df_predict.loc[len(df_predict)] = data;
            time.sleep(self.plc_timer);
            
        # add jitter 
        df_predict = self.setJitter(df_predict, df_parms, True);
        
        # zapis nova treninkova data
        self.logger.debug("Zapis do: "+str(path_to_df));
        path_to = Path(path_to_df);
        
        if exists(path_to):
            self.logger.info("Nacteno "+str(self.batch)+" vz. zapis-> "+path_to_df);
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="a", index=False, header=False);
        else:    
            self.logger.info("Nacteno "+str(self.batch)+" vz. zapis-> "+path_to_df);
            df_predict.to_csv(path_to_df, sep=";", float_format="%.6f", encoding="utf-8", mode="w", index=False, header=True);
            
        return(df_predict);
    
#------------------------------------------------------------------------
# opcCollectorGetDebugData
#------------------------------------------------------------------------
#    opcCollectorGetDebugData - totez co opcCollectorGetPLCdata ovsem
#    data se nectou z OPC ale  z CSV souboru. Toto slouzi jen pro ladeni
#    abychom nebyli zavisli na aktivite OPC serveruuuuu.
#    v pripade ladeni se nezapisuji treninkova data....
#------------------------------------------------------------------------
    def opcCollectorGetDebugData(self, df_parms):

        global df_debug_count;
        global df_debug_header;

        plc_isRunning = True;
        df_predict    = None;
        df_predict    = pd.DataFrame();
        current_day   = datetime.now().strftime("%Y-%m-%d");
        csv_file      = Path("./br_data/predict-debug.csv");
        self.logger.debug("Nacitam %s pro debug rezim... " %(csv_file));

        file_exists = exists(csv_file)

        if not file_exists:
            self.logger.info("Soubor ./br_data/predict-debug.csv nenalezen, exit(1)...");
            sys.exit(1);

        try:
            df_predict = pd.read_csv(csv_file,
                                     sep=",|;", 
                                     engine='python',  
                                     header=0, 
                                     encoding="utf-8",
                                     skiprows=df_debug_count,
                                     nrows=self.batch
                        );
            
        except  EmptyDataError as ex:
            self.logger.debug("Soubor predict-debug EmptyDataError %s " %(ex));
            return None;
        #df_predict = self.df_debug[self.df_debug_count : self.df_debug_count + self.batch];
        
        df_len = int(len(df_predict));
        if df_len <= 0:
            return None;

        if df_debug_count == 0:
            df_debug_header = df_predict.columns.tolist();
        else:
            df_predict.columns = df_debug_header;    
                
        df_debug_count += self.batch;
        
        # add jitter
        df_predict = self.setJitter(df_predict, df_parms, False);
        #df_predict.to_csv("./result/temp"+current_date+".csv");
        self.logger.debug("Nacteno "+str(self.batch)+" vzorku dat pro predict");
        return(df_predict);
    
#------------------------------------------------------------------------
# myFloatFormat         
#------------------------------------------------------------------------
    def myFloatFormat(self, x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

#------------------------------------------------------------------------
# myIntFormat         
#------------------------------------------------------------------------
    def myIntFormat(self, x):
        return ('%.f' % x).rstrip('.');


#------------------------------------------------------------------------
# isPing ????         
#------------------------------------------------------------------------
    def isPing(self):
        plc_isRunning = True;


        if not self.pingSocket(self.host1, self.port1):
            self.logger.debug("PING: False....: %s " %(self.host1));
            plc_isRunning = False;
            return(plc_isRunning);

        if not self.pingSocket(self.host2, self.port2):
            self.logger.debug("PING: False....: %s " %(self.host2));
            plc_isRunning = False;
            return(plc_isRunning);
        
        return (plc_isRunning);
    


#------------------------------------------------------------------------
# DataFactory
#------------------------------------------------------------------------
class DataFactory():

    df_parmX_predict = [];
    
    @dataclass
    class DataTrain:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z


    @dataclass
    class DataTrainDim:
          DataTrain: object;
          
    @dataclass
    class DataResult:
        # return self.DataResult(x_test, y_test, y_result, mse, mae)
        x_test:    object           #testovaci mnozina v ose x
        y_test:    object           #testovaci mnozina v ose y
        y_result:  object           #vysledna mnozina
        axis:      string           #osa stroje [X,Y, nebo Z]

    @dataclass
    class DataResultDim:
          DataResultX: object;

    def __init__(self, path_to_result, debug_mode, batch, current_date):

        self.logger    = logging.getLogger("ai");

        
    #Vystupni list parametru - co budeme chtit po siti predikovat
        self.df_parmx = [];
                       # 'temp_S1',
                       # 'temp_vr01',
                       # 'temp_vr02',
                       # 'temp_vr03',
                       # 'temp_vr04',
                       # 'temp_vr05',
                       # 'temp_vr06',
                       # 'temp_vr07'];
        
    #Tenzor predlozeny k uceni site
        self.df_parmX = [];
                       # 'dev_x4',
                       # 'dev_y4',
                       # 'dev_z4', 
                       # 'temp_S1',
                       # 'temp_vr01',
                       # 'temp_vr02',
                       # 'temp_vr03',
                       # 'temp_vr04',
                       # 'temp_vr05',
                       # 'temp_vr06',
                       # 'temp_vr07'];
        
        self.path_to_result = path_to_result;
        self.df_multiplier  = 1;   
        self.train          = pd.DataFrame();
        self.valid          = pd.DataFrame();
        self.predict        = pd.DataFrame();
        self.debug_mode     = debug_mode;
        self.batch          = batch;
        self.current_date   = current_date;
        
        
        self.parms  = [];
        self.header =["typ", 
                      "model", 
                      "epochs", 
                      "units",
                      "layesrs",
                      "batch", 
                      "actf", 
                      "shuffling", 
                      "txdat1", 
                      "txdat2",
                      "curr_txdat"];

        self.df_recc = 0;


        #parametry z parm file - nacte parametry z ./parms/parms.txt
        self.getParmsFromFile();
        # new OPCAgent()
        self.opc = OPCAgent(batch=self.batch, debug_mode=self.debug_mode);


#------------------------------------------------------------------------
# interpolateDF
#------------------------------------------------------------------------
#    interpoluje data splinem - vyhlazeni schodu na merenych artefaktech
#    u parametru typu 'dev' - odchylky [mikrometr]
#------------------------------------------------------------------------
    def interpolateDF(self, df, s_factor=0.001, ip_yesno=True, part="train"):

        test = False;
        s_factor = 0.00001;
        
        if df is None:
            return None;

        if not ip_yesno:
            self.logger.debug("interpolace artefaktu ip = False");
            return df;
        else:
            self.logger.debug("interpolace artefaktu ip = True");

            
        col_names = list(self.df_parmX);
        x = np.arange(0, df.shape[0]);

        for i in range(len(col_names)):
           if "dev" in col_names[i]:
               
        #       #interpolace univariantnim splinem
                spl =  UnivariateSpline(x, df[col_names[i]], s=s_factor);
        #       #linearni interpolace
                #spl =  interp1d(x, df[col_names[i]]);
                df[col_names[i]] = spl(x);

                
        if test:
            filename = "./temp/"+part+"_temp.csv" 
            path = Path(filename)
            if path.is_file():
                df.to_csv(filename, mode = "a", index=False, header=False, float_format='%.5f');
            else:
                df.to_csv(filename, mode = "w", index=False, header=True, float_format='%.5f');
            
        return df;

#------------------------------------------------------------------------
# setDataX(self, df,  size_train, size_valid, size_test)
#------------------------------------------------------------------------
    def setDataX(self, df, df_test,  size_train, size_valid, size_test, txdt_b=False, shuffling=False):
        #OSA XYZ
        try:
            
            DataTrain_x = self.DataTrain;
            DataTrain_x.train = pd.DataFrame(df[0 : size_train][self.df_parmX]);
            DataTrain_x.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmX]);
            
            DataTrain_x.test  = df_test;
            DataTrain_x.df_parm_x = self.df_parmx;  # data na ose x, pro rovinu X
            DataTrain_x.df_parm_y = self.df_parmX;  # data na ose y, pro rovinu Y
            DataTrain_x.axis = "OSA_XYZ";
            
            self.train = DataTrain_x.train;
            self.valid = DataTrain_x.valid;
            self.predict = DataTrain_x.test;
            
            return(DataTrain_x);
    
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());

    
#-----------------------------------------------------------------------
# getData
#-----------------------------------------------------------------------
#    Hlavni metoda pro manipulaci s daty
#    V DEBUG modu nacita data pro predikci ze souboru predict-debug.csv
#    V NODEBUG modu nacita data pro predikci z PLC
#    data pro trenink a validaci nacita ze souboruuuu
#    ./br-data/tm-ai_*.csv - provadi merge a plni hlavni datovy
#    ramec.
#-----------------------------------------------------------------------
    def getData(self,
                shuffling       = False, 
                timestamp_start = '2022-01-01 00:00:00', 
                timestamp_stop  = '2099-12-31 23:59:59', 
                type            = "predict",
                ilcnt           = 1,
                ip_yesno        = True):
        
        txdt_b      = False;
        df          = pd.DataFrame(columns = self.df_parmX);
        df_test     = pd.DataFrame(columns = self.df_parmx);
        
        size_train  = 0;
        size_valid  = 0;
        size_test   = 0;
        size        = 0;
                            #  interpolacni faktor
        s_factor    = 0.5   #  0.5 pro periodu 60[s]
                            #  0.01 pro periodu 1[s]
        
        try:
           
            #self.DataTrainDim.DataTrain = None;
            
#----------------------------------------------------------------------------- 
# Data pro trenink - if type == "train"
#----------------------------------------------------------------------------- 
            if "train" in type:
                if os.name == "nt":
                    files = os.path.join(Path("./br_data"), "tm-ai_*.csv");
                else:
                    files = os.path.join("./br_data", "tm-ai_*.csv");
                # list souboru pro join
                joined_list = glob.glob(files);

                len_list = len(joined_list);

                if len_list == 0:
                    self.logger.info("Data pro trenink nejsou k dispozici, exit()...");
                    sys.exit(1);
            
                # sort souboru pro join
                joined_list.sort(key=None, reverse=False);
                usecols = ["datetime"];
                for col in self.df_parmX:
                    usecols.append(col);

                df = pd.concat([pd.read_csv(csv_file,
                                         sep=",|;", 
                                         engine='python',  
                                         header=0, 
                                         encoding="utf-8",
                                         usecols = usecols 
                                       )
                                    for csv_file in joined_list],
                                    axis=0, 
                                    ignore_index=True
                    );
                # Odfiltruj data kdy stroj byl vypnut
                #df = df[(df["dev_y4"] != 0) & (df["dev_z4"] != 0)];
                # bordel pri domluve nazvoslovi...            
                df.columns = df.columns.str.lower();
                # interpoluj celou mnozinu data  
                df = self.interpolateDF(df=df, ip_yesno=ip_yesno, part="train");
            
                # vyber dat dle timestampu
                df["timestamp"] = pd.to_datetime(df["datetime"].str.slice(0, 18));
                
                # treninkova a validacni mnozina    
                df = df[(df["timestamp"] > timestamp_start) & (df["timestamp"] <= timestamp_stop)];
                
                if len(df) <= 1:
                    self.logger.error("Data pro trenink maji nulovou velikost - exit(1)");
                    sys.exit(1);

                #nactena kazda ilcnt-ta veta - zmenseni mnoziny dat pro uceni.
                #if self.debug_mode is True and ilcnt > 1:
                if ilcnt > 1:
                    df = df.iloc[::ilcnt, :];
            
                df["index"] = pd.Index(range(0, len(df), 1));
                df.set_index("index", inplace=True);

                #dopruceny pomer train/valid
                #   60/40 -:- 70/30 pro mensi objemy dat
                #   80/20 -:- 90/10 pro velke dat

                size = len(df);
                size_train = math.floor(size * 8 / 12);
                size_valid = math.floor(size * 4 / 12);
                size_test  = math.floor(size * 0 / 12);
                self.df_recc = size;
                
                if size > 0:
                    self.logger.info("Data pro trenink nactena, pocet vet: %d, ilcnt: %d " %(size, ilcnt));

#----------------------------------------------------------------------------- 
# Data pro predict - if type == "train" || type == "predict"
#----------------------------------------------------------------------------- 
            self.logger.debug("Data pro predikci....");
            if self.debug_mode is True:
                df_test = self.opc.opcCollectorGetDebugData(self.df_parmX);
            else:
                self.logger.debug("Data pro predikci getPlcData....");
                df_test = self.opc.opcCollectorGetPlcData(self.df_parmX);
                
            if df_test is None:
                self.logger.error("Nebyla nactena zadna data pro predikci");
            elif len(df_test) == 0:
                self.logger.error("Patrne nebezi nektery OPC server  ");
            else:    
                df_test = self.interpolateDF(df=df_test, ip_yesno=ip_yesno, part="predict");
                
            if self.df_parmx is None or self.df_parmX is None:
                pass;
            else:
                self.DataTrainDim.DataTrain = self.setDataX( df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b,
                                                             shuffling=shuffling 
                                                        );
            

            return self.DataTrainDim(self.DataTrainDim.DataTrain);

        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            
#-----------------------------------------------------------------------
# saveDataToPLC  - result
#-----------------------------------------------------------------------
#     index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
#
#     return True - nacti dalsi porci dat pro predict
#     return False - nenacitej dalsi porci dat pro predict.
#     POZOR !!! do PLC zapisuje jen v NODEBUG modu
#-----------------------------------------------------------------------
    def saveDataToPLC(self,
                      threads_result,
                      timestamp_start,
                      thread_name,
                      typ,
                      model,
                      units,
                      epochs,
                      ip_yesno=False):
        
        saveresult = True;
        
        thread_name = thread_name[0 : len(thread_name) - 1].replace("_","");
        
        col_names_y = list(self.DataTrain.df_parm_y);
        filename = "./result/plc_archiv/plc_"+thread_name+"_"+str(self.current_date)[0:10]+".csv"
        
        df_result = pd.DataFrame(columns = col_names_y);
        df_append = pd.DataFrame();
        
        i = 0
        frames = [];
        #precti vysledky vsech threadu...
        for result in threads_result:
            i += 1;
            if result[0] is None:
                return False;   # nenacitej novou porci dat, thready neukoncily cinnost....
            else:
                df = result[0];
                if df is not None:
                    df["index"] =  df.reset_index().index;
                    frames.append(df);

        df_result = pd.concat(frames); 
        df_result = df_result.groupby("index").mean();
        df_plc = self.formatToPLC(df_result, col_names_y);
                                      
        path = Path(filename)
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            self.logger.debug(f'Soubor {filename} existuje - append');
            df_plc.to_csv(filename, mode = "a", index=False, header=False, float_format='%.5f');
        else:
            self.logger.debug(f'Soubor {filename} neexistuje - insert');
            df_plc.to_csv(filename, mode = "w", index=False, header=True, float_format='%.5f');

        # data do PLC - debug mode -> disable...
        if self.debug_mode is False:
            result_opc = self.opc.opcCollectorSendToPLC(df_plc=df_plc );
            if result_opc:
                self.logger.debug("Data do PLC byla zapsana.");
            else:    
                self.logger.error("Data do PLC nebyla zapsana.");
        
        # data ke zkoumani zapisujeme v pripade behu typu "train" a zaroven v debug modu
        if  self.debug_mode is True:
            saveresult=True;
        else:
            saveresult=False;
            
        if saveresult:
            self.logger.debug(filename + " vznikne.");

                                   
            self.saveDataResult(timestamp_start,
                                df_result,
                                model,
                                typ,
                                units,
                                epochs,
                                saveresult=True,
                                ip_yesno=False);                    

        else:
            self.logger.debug(filename + " nevznikne :" +str(saveresult));
        
        #vynuluj 
        for result in threads_result:
            result[0] = None;
        
        return True; #nacti novou porci dat    
        
     
#-----------------------------------------------------------------------
# formatToPLC  - result
#-----------------------------------------------------------------------
    def formatToPLC(self, df_result, col_names_y):
        #curent timestamp UTC
        current_time = time.time()
        utc_timestamp = datetime.utcfromtimestamp(current_time);

        l_plc = [];        
        l_plc_col = [];        
        
        l_plc.append( str(utc_timestamp)[0:19]);
        l_plc_col.append("utc");

        for col in col_names_y:
            if "dev" in col:
                mmean = self.myIntFormat(df_result[col].mean() *10000);   #prevod pro PLC (viz dokument Teplotni Kompenzace AI)
                l_plc.append(mmean);                                  #  10 = 0.001 atd...
                l_plc_col.append(col+"mean");
                
        return (pd.DataFrame([l_plc], columns=[l_plc_col]));
        
        
#-----------------------------------------------------------------------
# saveDataResult  - result
#-----------------------------------------------------------------------
#    index = pd.RangeIndex(start=10, stop=30, step=2, name="data")
#    POZOR !!! volano jen v DEBUG modu
#-----------------------------------------------------------------------
    def saveDataResult(self,
                       timestamp_start,
                       df_result,
                       model,
                       typ,
                       units,
                       epochs,
                       saveresult        = True,
                       ip_yesno          = False
                ):

        filename="./result/res_"+str(self.current_date)[0:10]+"_"+model+ "_"+str(units)+ "_"+ str(epochs)+".csv";
        try:
            col_names_y = list(self.DataTrain.df_parm_y);
            col_names_x = list(self.DataTrain.df_parm_x);
            
            col_names_predict = list("");
            col_names_drop    = list("");
            col_names_drop2   = list("");
            col_names_train   = list({"datetime"});
            col_names_dev     = list("");
            
            for col in col_names_y:
                col_names_train.append(col);
            
            for i in range(len(col_names_y)):
                if "dev" in col_names_y[i]:
                    col_names_predict.append(col_names_y[i]+"_predict");
                    col_names_dev.append(col_names_y[i]);
                else:    
                    col_names_drop.append(col_names_y[i]);

            for col in self.DataTrain.test.columns:
                if col in col_names_train:
                    a = 0;
                else:    
                    col_names_drop2.append(col);
                    
            
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());

            
        try:
            self.DataTrain.test.reset_index(drop=True, inplace=True)
            
            df_result.drop(col_names_drop, inplace=True, axis=1);
            df_result  = pd.DataFrame(np.array(df_result), columns =col_names_predict);

            df_result2 = pd.DataFrame();
            df_result2 = pd.DataFrame(self.DataTrain.test);
            df_result2 = df_result2.iloc[ : df_result.shape[0], :];

            #merge1 - left inner join
            df_result  = pd.concat([df_result.reset_index(drop=True),
                                    df_result2.reset_index(drop=True)],
                                    axis=1);
            # Absolute Error
            for col in col_names_dev:
                ae = df_result[col].astype(float) - df_result[col+"_predict"].astype(float);
                df_result[col+"_ae"] = ae;
                
            list_cols     = list({"idx"}); 
            list_cols_mean = list({"idx"}); 
            
            for col in col_names_dev:
                if "dev" in col:
                    list_cols.append(col+"_predict");
                    list_cols_mean.append(col+"_predict_avg");

            
            # AE absolute error
            for col in col_names_dev:
                ae = (df_result[col].astype(float) - df_result[col+"_predict"].astype(float));
                df_result[col+"_ae"] = ae;

            path = Path(filename)
            if path.is_file():
                append = True;
            else:
                append = False;

            if append:             
                self.logger.debug(f"Soubor {filename} existuje - append: "
                                  + str(len(df_result))
                                  +" vet...");
                df_result.to_csv(filename,
                                 mode = "a",
                                 index=True,
                                 header=False,
                                 float_format='%.5f');
            else:
                self.logger.debug(f"Soubor {filename} neexistuje - insert: "
                                  + str(len(df_result))
                                  + " vet...");
                df_result.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');
                
            self.saveParmsMAE(df_result, model, typ, units, epochs)    

        except Exception as ex:
            traceback.print_exc();
        
        return;    

#-----------------------------------------------------------------------
# saveParmsMAE - zapise hodnoty MAE v zavislosti na pouzitych parametrech
#-----------------------------------------------------------------------
#    Zapisuje vysledky mereni a predikce
#    POZOR !!! volano jen v DEBUG modu
#-----------------------------------------------------------------------
    def saveParmsMAE(self, df,  model, typ, units, epochs):

        filename = "./result/mae_" + str(self.current_date)[0:10] + "_" + model + "_" + str(units) + "_" + str(epochs) + ".csv"
        
        local_parms = [];   
        local_header = [];     
        #pridej maximalni hodnotu AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    local_header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    local_parms.append(float(res));        
                else:
                    local_header.append(col+"_max");
                    res = self.myFloatFormat(df[col].abs().max())
                    local_parms.append(float(res));        
        
        #pridej mean AE
        for col in df.columns:
            if "_ae" in col:
                if "_avg" in col:
                    local_header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    local_parms.append(float(res));        
                else:
                    local_header.append(col+"_avg");
                    res = self.myFloatFormat(df[col].abs().mean())
                    local_parms.append(float(res));

        local_header.append("df_recc");
        local_parms.append(int(self.df_recc));

        local_parms = self.parms + local_parms;
        local_header = self.header + local_header;

        df_ae = pd.DataFrame(data=[local_parms], columns=local_header);
        
        path = Path(filename)
        if path.is_file():
            append = True;
        else:
            append = False;
        
        if append:             
            self.logger.debug(f'Soubor {filename} existuje - append');
            df_ae.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
        else:
            self.logger.debug(f'Soubor {filename} neexistuje - insert');
            df_ae.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');
            
        return;    
        
#-----------------------------------------------------------------------
# getParmsFromFile - nacte parametry z ./cfg/ai-parms.cfg
#-----------------------------------------------------------------------
    def getParmsFromFile(self):
        # otevreno v read mode
        
        parmfile = "./cfg/ai-parms.cfg";
        try:
            file = open(Path(parmfile), "r")
            lines = file.readlines();
            
            count = 0
            for line in lines:
                line = line.replace("","").replace("'","").replace(" ","");
                line = line.strip();
                #if not line:
                x = line.startswith("df_parmx=")
                if x:
                    line = line.replace("df_parmx=", "").lower();
                    self.df_parmx = line.split(",");
                    if "null" in line:
                        self.df_parmx = None;
                        
                X = line.startswith("df_parmX=");
                if X:
                    line = line.replace("df_parmX=", "").lower();
                    self.df_parmX = line.split(",");
                    if "null" in line:
                        self.df_parmX = None;
            
                
            file.close();
            self.logger.info("parametry nacteny z "+ parmfile);       
                
        except:
            self.logger.error("Soubor parametru "+ parmfile + " nenalezen!");                
            self.logger.info("Parametry pro trenink site budou nastaveny implicitne v programu");                 
        
        return();  
    
#------------------------------------------------------------------------
# DataFactory getter metody
#------------------------------------------------------------------------
    def getDf_parmx(self):
        return self.df_parmx;
    
    def getDf_parmX(self):
        return self.df_parmX;

    def getDfTrainData(self):
        return (self.train);
    
    def getDfValidData(self):
        return (self.valid);
    
    def getDfPredictData(self):
        return (self.predict);
    
    def setParms(self, parms):
        self.parms = parms;
    
    def getParms(self):
        return self.parms;
    
    def setHeader(self, header):
        self.header = header;
    
    def getHeader(self):
        return self.header;

    def getCurrentDate(self):
        return(self.current_date);
    
    def setCurrentDate(self, val):
        self.current_date = val;
        
#------------------------------------------------------------------------
# isPing         
#------------------------------------------------------------------------
    def isPing(self):
        is_ping = self.opc.isPing();
        return(is_ping);
    
#------------------------------------------------------------------------
# myFloatFormat         
#------------------------------------------------------------------------
    def myFloatFormat(self,x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

#------------------------------------------------------------------------
# myIntFormat         
#------------------------------------------------------------------------
    def myIntFormat(self, x):
        return ('%.f' % x).rstrip('.');

#------------------------------------------------------------------------
# Neuronova Vrstava -obecna neuronova vrstva
#------------------------------------------------------------------------
class NeuronLayerLSTM():
    #definice datoveho ramce
    
    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, 
                 path_to_result, 
                 typ, 
                 model_1,
                 model_2,
                 epochs,
                 batch, 
                 txdat1, 
                 txdat2, 
                 units_1, 
                 units_2, 
                 layers_1,
                 layers_2,
                 shuffling, 
                 actf,
                 debug_mode,
                 current_date = "",
                 thread_name = "",
                 ilcnt = 1,
                 ip_yesno = False,
                 lrn_rate = 0.0005
    ):
        
        self.logger         = logging.getLogger("ai");
        self.path_to_result = path_to_result; 
        self.typ            = typ;
        self.epochs         = epochs;  
        self.batch          = batch;
        self.txdat1         = txdat1;
        self.txdat2         = txdat2;
        self.ilcnt          = ilcnt;
        
        self.df             = pd.DataFrame()
        self.df_out         = pd.DataFrame()
        self.graph          = None;
        self.data           = None;
        
        # parametry hidden vrstvy
        self.model_1        = model_1;        # model hidden vrstvy
        self.model_2        = model_2;
        self.units_1        = units_1;        # pocet neuronu v hidden
        self.units_2        = units_2;
        self.layers_1       = layers_1;       # pocet vrstev v hidden
        self.layers_2       = layers_2;

        self.shuffling      = shuffling;
        self.actf           = actf;
        self.debug_mode     = debug_mode;
        self.current_date   = current_date,
        self.data           = None;
        self.x_train_scaler = None;
        self.y_train_scaler = None;
        self.x_valid_scaler = None;
        self.y_valid_scaler = None;
        self.neural_model   = None;
        self.ip_yesno       = ip_yesno;
        self.lrn_rate       = lrn_rate;

        
        #parametry window-size
        self.windowX        = 24;  #min 1, max 64
        #parametry window-size
        self.windowY        = 1;  # musi byt 1
        #parametry time-observe...
        self.n_in           = 3;
        self.n_out          = 3;

#------------------------------------------------------------------------
# toTensorLSTM(self, dataset, window = 16):
#------------------------------------------------------------------------
# Pracujeme - li s rekurentnimi sitemi (LSTM GRU...), pak 
# musíme vygenerovat dataset ve specifickém formátu.
# Vystupem je 3D tenzor ve forme 'window' casovych kroku.
#  
# Jakmile jsou data vytvořena ve formě 'window' časových kroků, 
# jsou nasledne prevedena do pole NumPy a reshapovana na 
# pole 3D X_dataset.
#
# Funkce take vyrobi pole y_dataset, ktere muze byt pouzito pro 
# simulaci modelu vstupnich dat, pokud tato data nejsou k dispozici.  
# y_dataset predstavuje "window" časových rámců krat prvni prvek casoveho 
# ramce pole X_dataset
#
# funkce vraci: X_dataset - 3D tenzor dat pro uceni site
#               y_dataset - vektor vstupnich dat (model)
#               dataset_cols - pocet sloupcu v datove sade. 
#
# poznamka: na konec tenzoru se pripoji libovolne 'okno' aby se velikost
#           o toto okno zvetsila - vyresi se tim chybejici okno pri predikci
#           
#------------------------------------------------------------------------
    
    def toTensorLSTM(self, dataset, window):

        X_dataset = []  #data pro tf.fit(x - data pro uceni
        y_dataset = []  #data pro tf.fit(y - vstupni data 
                            #jen v pripade ze vst. data nejsou definovana

        values = dataset[0 : window, ];
        dataset = np.append(dataset, values, axis=0) #pridej delku okna
        dataset_rows, dataset_cols = dataset.shape;

        if window == 0:
            window = 1;
        
        if window >= dataset_rows:
            self.logger.error("prilis maly  vektor dat k uceni!!! \nparametr window je vetsi nez delka vst. vektoru ");
            return(None);
        
        for i in range(window, dataset_rows):
            X_dataset.append(dataset[i - window : i, ]);
            y_dataset.append(dataset[i, ]);
        
        #doplnek pro append chybejicich window vzorku pri predikci
        X_dataset.append(dataset[0 : window, ]);
            
        X_dataset = np.array(X_dataset);
        y_dataset = np.array(y_dataset);
        
        X_dataset = np.reshape(X_dataset, (X_dataset.shape[0], X_dataset.shape[1], dataset_cols));
        
        return NeuronLayerLSTM.DataSet(X_dataset, y_dataset, dataset_cols);

#------------------------------------------------------------------------
# fromTensorLSTMMean(self, dataset, window = 64):
#------------------------------------------------------------------------
# Udelej mean nad window
# funkce vraci: y_result - 2D array vysledku predikce
#
#------------------------------------------------------------------------
    def fromTensorLSTMMean(self, dataset, dropNaN=True):
        df = pd.DataFrame(np.mean(dataset, axis=1));
        df = df.iloc[ : -self.windowX, :];
        if len(df) == 0:
            return(None);
        return(df);
    
#------------------------------------------------------------------------    
# toTimeSeries
#------------------------------------------------------------------------    
#    vstup....: casova rada hodnot [list nebo NumPy array]
#    n_in.....:pocet kroku casoveho zpozdeni pro vstupni hodnoty
#    n_out....: pocet kroku
#    dropNaN..: drop NaN ?
#------------------------------------------------------------------------    

    def toTimeSeries (self, df, n_in=3, n_out=3, dropNaN=True):

        n = n_in + n_out;
        if n == 0:
            return df;
        
        head   = list(df.head());
        cols   = list(); 
        names  = list();
        n_vars = df.shape [1];
        
    # input sequence (t-n, ... t-1)
        for i in range(n_in , 0, -1):
            cols.append(df.shift(i));
            names += [(head[j]+"(t-%d)" % (j+1)) for j in range(n_vars)];
        
    # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i));
            if i == 0:
                names += [(head[j]+"(t)" ) for j in range(n_vars)];
            else:
                names += [(head[j]+"(t+%d)" % (j+1)) for j in range(n_vars)];

    # concat
        result = concat(cols , axis=1);
        result.columns = names;
    
    # drop rows with NaN
        if dropNaN:
            result.dropna(inplace=True);

        return result;


#------------------------------------------------------------------------    
# fromTimeSeries
#------------------------------------------------------------------------    
#    vstup....: casova rada hodnot [list nebo NumPy array]
#    n_in.....: pocet kroku casoveho zpozdeni pro vstupni hodnoty
#    n_out....: pocet kroku
#    dropNaN..: drop NaN ?
#    df.groupby(np.arange(len(df.columns))//3, axis=1).mean()
#------------------------------------------------------------------------
    def fromTimeSeriesMean (self, array, colnames, n_in=3, n_out=3, dropNaN=True):

        n = n_in + n_out;
        if n <= 1:
            df = pd.DataFrame(array);
            df.columns = colnames;
            return df;
        
        df = pd.DataFrame(array);
        df_mean = pd.DataFrame();
        icol = int(df.shape[1]/n);
        
        icnt = 0;
        for col in colnames:
            df_mean[col] =  df.iloc[:,icnt::icol].mean(axis=1);
            icnt += 1;

        # zvetsi o ztracene radky na velikost self.batch

        #df_len = df_mean.shape[0];
        #loss   = self.batch - df_len;
        #df_loss = df.iloc[(df_len - loss)  : df_len, :];
        #frames = [df_mean, df_loss];
        #result = pd.concat(frames);
        
        return(df_mean);

#------------------------------------------------------------------------
# smoothGraph - trochu vyhlad graf
#------------------------------------------------------------------------
    def smoothGraph(self, points, factor=0.9):
        smoothed_points = [];
        
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1];
                smoothed_points.append(previous * factor + point * (1 - factor));
            else:
                smoothed_points.append(point);
                
        return smoothed_points;        


#------------------------------------------------------------------------    
# printTrainGraph - tisk optimalizacnich grafu (jen v debug modu)
#------------------------------------------------------------------------    
    def printTrainGraph (self, history):
        
        #kolik se vynecha o zacatku v grafu?
        start_point = 0;
        
        loss_train = history.history['loss'];
        loss_train = self.smoothGraph(points = loss_train[start_point:], factor = 0.9)
        loss_val = history.history['val_loss'];
        loss_val = self.smoothGraph(points = loss_val[start_point:], factor =  0.9)
        epochs = range(0,len(loss_train));
        
        # disable GUI: pokud bezi ve vlaknu bez GUI - vyvola vyjimku
        plt.switch_backend('agg')
        plt.clf();
        plt.plot(epochs, loss_train, label='LOSS treninku');
        plt.plot(epochs, loss_val,   label='LOSS validace');
        plt.title('LOSS treninku');
        plt.xlabel('Pocet epoch');
        plt.ylabel('LOSS');
        plt.legend();
        plt.savefig(self.path_to_result+'/graf_loss.pdf', format='pdf');
        plt.clf();
        
        acc_train =  history.history['acc'];
        acc_train = self.smoothGraph(points=acc_train[start_point:], factor = 0.9)
        acc_val = history.history['val_acc'];
        acc_val = self.smoothGraph(points=acc_val[start_point:], factor = 0.9)
        epochs = range(0,len(acc_train));
        # disable GUI: pokud bezi ve vlaknu bez GUI - vyvola vyjimku
        plt.switch_backend('agg')
        plt.clf();
        plt.plot(epochs, acc_train, label='ACC treninku');
        plt.plot(epochs, acc_val, label='ACC validace');
        plt.title('ACC treninku');
        plt.xlabel('Pocet epoch');
        plt.ylabel('ACC');
        plt.legend();
        plt.savefig(self.path_to_result+'/graf_acc.pdf', format='pdf');
        plt.clf();



#------------------------------------------------------------------------
# Definice neuronove vrstvy
# -  definice DropOut filtru - konstanta rate
# -  definice prvni skryte vrstvy
# -  definice druhe skryte vrstvy, je li layers_count_2 == 0, druha vrstva
#    se preskakuje
#
#    Dve skryte vrstvy jsou definovany proto, aby bylo mozno kombinovat
#    ruzne typy neuronovych siti (DENSE, CONV1D, LSTM, GRU)
#------------------------------------------------------------------------
    def neuralNetworkLSTMtrain(self, DataTrain, thread_name):

        n_in         = self.n_in;
        n_out        = self.n_out;

        window_X     = self.windowX;
        window_Y     = self.windowY;

        model_1      = self.model_1;
        model_2      = self.model_2;
        units_1      = self.units_1;
        units_2      = self.units_2;

        # ladici parametry
        layers_count_1 =  self.layers_1;       # pocet vrstev v hidden
        layers_count_2 =  self.layers_2;
        
        dropout_filter = True;                # Dropout
        rate           =  0.15;               # a jeho rate....
        
        try:
            y_train_data = np.array(self.toTimeSeries(DataTrain.train[DataTrain.df_parm_y], n_in, n_out));
            x_train_data = np.array(self.toTimeSeries(DataTrain.train[DataTrain.df_parm_x], n_in, n_out));
            y_valid_data = np.array(self.toTimeSeries(DataTrain.valid[DataTrain.df_parm_y], n_in, n_out));
            x_valid_data = np.array(self.toTimeSeries(DataTrain.valid[DataTrain.df_parm_x], n_in, n_out));
            
            if (x_train_data.size == 0 or y_train_data.size == 0):
                return();
            

            inp_size   = x_train_data[0].size;
            out_size   = y_train_data[0].size;
            v_inp_size = x_train_data[0].shape[0];
            v_out_size = y_train_data[0].shape[0];

            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_train_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train_data = self.x_train_scaler.fit_transform(x_train_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_train_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_train_scaler = MinMaxScaler(feature_range=(0, 1))
            y_train_data = self.y_train_scaler.fit_transform(y_train_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_train_scaler"+thread_name+".pkl", 'wb'))
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            self.x_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            x_valid_data = self.x_valid_scaler.fit_transform(x_valid_data)
            pickle.dump(self.x_train_scaler, open("./temp/x_valid_scaler"+thread_name+".pkl", 'wb'))
            
            self.y_valid_scaler = MinMaxScaler(feature_range=(0, 1))
            y_valid_data = self.y_valid_scaler.fit_transform(y_valid_data)
            pickle.dump(self.y_train_scaler, open("./temp/y_valid_scaler"+thread_name+".pkl", 'wb'))
        
        #data pro trenink -3D tenzor
            X_train =  self.toTensorLSTM(x_train_data, window=window_X);
        #vstupni data train 
            Y_train = self.toTensorLSTM(y_train_data, window=window_Y);
            Y_train.X_dataset = Y_train.X_dataset[0 : X_train.X_dataset.shape[0]];
        #data pro validaci -3D tenzor
            X_valid = self.toTensorLSTM(x_valid_data, window=window_X);
        #vxsystupni data pro trenink -3D tenzor
            Y_valid = self.toTensorLSTM(y_valid_data, window=window_Y);
            Y_valid.X_dataset = Y_valid.X_dataset[0 : X_valid.X_dataset.shape[0]];
            
#------------------------------------------------------------------------    
# Input layer    
#------------------------------------------------------------------------    
            neural_model = Sequential();
            initializer  = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            neural_model.add(Input(shape=(X_train.X_dataset.shape[1], X_train.cols,)));
            #pridana vstupni vrstva Dense -> lepsi vysledky RMSE
            neural_model.add(layers.Dense(units= int(X_train.X_dataset.shape[1])));

#------------------------------------------------------------------------    
# Hidden layer - begin DENSE, LSTM, GRU, CONV1D
# Vrstva 1
#------------------------------------------------------------------------
            for i in range(layers_count_1):
                    
                if "DENSE" in model_1:
                    self.logger.debug(model_1);
                    neural_model.add(layers.Dense(units= int(units_1),
                                              activation=self.actf,
                                              kernel_initializer=initializer,
                                              bias_initializer="zeros"
                    )
                );

                if "LSTM" in model_1:
                    self.logger.debug(model_1);
                    neural_model.add(layers.LSTM(units = int(units_1), # self.units / 4
                                             activation=self.actf,
                                             recurrent_activation="sigmoid",
                                             use_bias=True,
                                             # kernel_initializer=initializer,
                                             kernel_initializer="glorot_uniform",
                                             recurrent_initializer="orthogonal",
                                             bias_initializer="zeros",
                                             unit_forget_bias=True,
                                             return_sequences=True
                    )
                );

                if "GRU" in model_1:
                    self.logger.debug(model_1);
                    neural_model.add(layers.GRU(units = int(units_1), # self.units / 4
                                            activation=self.actf,
                                            recurrent_activation="sigmoid",
                                            use_bias=True,
                                            # kernel_initializer=initializer,
                                            kernel_initializer="glorot_uniform",
                                            recurrent_initializer="orthogonal",
                                            bias_initializer='zeros',
                                            return_sequences=True
                    )
                );
                    
                if "CONV1D" in model_1:
                    self.logger.debug(model_1);
                    neural_model.add(layers.Conv1D(filters=16,       # filters=32 
                                               kernel_size=2,    # kernel_size=4
                                               padding="same",
                                               activation="relu",# relu
                                               use_bias=True,
                                               kernel_initializer="glorot_uniform",
                                               bias_initializer="zeros"
                    )
                );

                if dropout_filter:    
                    neural_model.add(Dropout(rate=rate, noise_shape=None, seed=None));

            #end-for

#------------------------------------------------------------------------    
# Hidden layer - begin DENSE, LSTM, GRU, CONV1D
# Vrstva 2
#------------------------------------------------------------------------    
            for i in range(layers_count_2):
                    
                if "DENSE" in model_2:
                    self.logger.debug(model_2);
                    neural_model.add(layers.Dense(units= int(units_2),
                                              activation=self.actf,
                                              kernel_initializer=initializer,
                                              bias_initializer="zeros"
                    )
                );

                if "LSTM" in model_2:
                    self.logger.debug(model_2);
                    neural_model.add(layers.LSTM(units = int(units_2), # self.units / 4
                                             activation=self.actf,
                                             recurrent_activation="sigmoid",
                                             use_bias=True,
                                             # kernel_initializer=initializer,
                                             kernel_initializer="glorot_uniform",
                                             recurrent_initializer="orthogonal",
                                             bias_initializer="zeros",
                                             unit_forget_bias=True,
                                             return_sequences=True
                    )
                );

                if "GRU" in model_2:
                    self.logger.debug(model_2);
                    neural_model.add(layers.GRU(units = int(units_2), # self.units / 4
                                            activation=self.actf,
                                            recurrent_activation="sigmoid",
                                            use_bias=True,
                                            # kernel_initializer=initializer,
                                            kernel_initializer="glorot_uniform",
                                            recurrent_initializer="orthogonal",
                                            bias_initializer='zeros',
                                            return_sequences=True
                    )
                );
                    
                if "CONV1D" in model_2:
                    self.logger.debug(model_2);
                    neural_model.add(layers.Conv1D(filters=16,       # filters=32 
                                               kernel_size=2,    # kernel_size=4
                                               padding="same",
                                               activation="relu",# relu
                                               use_bias=True,
                                               kernel_initializer="glorot_uniform",
                                               bias_initializer="zeros"
                    )
                );

                if dropout_filter:    
                    neural_model.add(Dropout(rate=rate, noise_shape=None, seed=None));

            #end-for

#------------------------------------------------------------------------
# Hidden layer - end
#------------------------------------------------------------------------    
                    
#------------------------------------------------------------------------    
# Output layer
#------------------------------------------------------------------------    
        
            neural_model.add(layers.Dense(Y_train.cols, activation="relu"));

            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0003,
                                                  beta_1       = 0.9,
                                                  beta_2       = 0.999,
                                                  epsilon      = 1e-07
                                            );


        # definice ztratove funkce a optimalizacniho algoritmu
            neural_model.compile(loss='mse', optimizer=optimizer, metrics=["mse", "acc"]);
        # natrenuj neural_model na vstupni dataset
            history = neural_model.fit(X_train.X_dataset, 
                                       Y_train.X_dataset, 
                                       epochs         =self.epochs, 
                                       batch_size     =self.batch, 
                                       verbose        =2,
                                       shuffle        =self.shuffling,
                                       validation_data=(X_valid.X_dataset, Y_valid.X_dataset)
                            );

        # zapis neural_modelu    
            neural_model.save("./models/model_"+thread_name+"_"+DataTrain.axis, overwrite=True, include_optimizer=True);

            if self.debug_mode is True:
                neural_model.summary();
                self.printTrainGraph(history);
                
            self.neural_model = neural_model;
            return ();
        
        except Exception as ex:
            self.logger.error(traceback.print_exc());
        
        
#------------------------------------------------------------------------
# Neuronova Vrstava - predict 
#------------------------------------------------------------------------
    def neuralNetworkLSTMpredict(self, DataTrain, thread_name):
        
        n_in  = self.n_in;
        n_out = self.n_out;
        
        try:
            axis     = DataTrain.axis;  
            #x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            #y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            x_test   = np.array(self.toTimeSeries(DataTrain.test[DataTrain.df_parm_x], n_in, n_out));
            y_test   = np.array(self.toTimeSeries(DataTrain.test[DataTrain.df_parm_y], n_in, n_out));

            if x_test.shape[0] == 0:
                return(None);

            if y_test.shape[0] == 0:
                return(None);
            
            self.x_train_scaler =  pickle.load(open("./temp/x_valid_scaler"+thread_name+".pkl", 'rb'))
            self.y_train_scaler =  pickle.load(open("./temp/y_valid_scaler"+thread_name+".pkl", 'rb'))
            
            x_test        = self.x_train_scaler.transform(x_test);
            
            x_object      = self.toTensorLSTM(x_test, window=self.windowX);
            dataset_rows, dataset_cols = x_test.shape;
        # predict
            y_result      = self.neural_model.predict(x_object.X_dataset);
        
        # reshape 3d na 2d  
            z_result      = self.fromTensorLSTMMean(y_result);
            
        # if z_result == NOne -> len(df) == 0;    
            if z_result is None:
                return DataFactory.DataResult(x_test, y_test, y_result, axis)
            
            y_result      = self.y_train_scaler.inverse_transform(z_result);

            x_test   = self.fromTimeSeriesMean(x_test, DataTrain.df_parm_x, n_in, n_out);
            y_test   = self.fromTimeSeriesMean(y_test, DataTrain.df_parm_y, n_in, n_out);
            y_result = self.fromTimeSeriesMean(y_result, DataTrain.df_parm_y, n_in, n_out);
        # plot grafu compare...
            #model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            self.logger.error("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem");
            self.logger.error("          zkuste nejdrive --typ == train !!!");
            self.logger.error(traceback.print_exc());

#------------------------------------------------------------------------
# neuralNetworkLSTMexec - exec 
#------------------------------------------------------------------------
    def neuralNetworkLSTMexec(self, threads_result, threads_cnt):

        thread_name = "thread01";
        try:
            
            thread_name = threads_result[threads_cnt][1];
            startTime = int(time.time_ns());

            #nacti pouze predikcni data 
            self.logger.debug("Nacitam data pro predikci, v modu: "+ str(self.typ));
            self.data.Data = self.data.getData(shuffling       = self.shuffling, 
                                               timestamp_start = self.txdat1, 
                                               timestamp_stop  = self.txdat2,
                                               type            = self.typ,
                                               ilcnt           = self.ilcnt,
                                               ip_yesno        = self.ip_yesno);


            if self.typ == 'train':
                self.logger.debug("Start threadu: %s v TRAIN modu " %(thread_name));

                self.neuralNetworkLSTMtrain(self.data.DataTrainDim.DataTrain, thread_name);
            else:    
                self.logger.debug("Start threadu: %s v PREDICT modu " %(thread_name));
            
            if self.data.DataTrainDim.DataTrain.test is None or len(self.data.DataTrainDim.DataTrain.test) == 0: 
                self.logger.info("Data : "+ thread_name+ " pro predikci nejsou k dispozici....");
                
                if self.debug_mode is True:
                    self.logger.info("Exit...");
                    sys.exit(1);
                else:    
                    return();
           
            
            self.data.DataResultDim.DataResultX = self.neuralNetworkLSTMpredict(self.data.DataTrainDim.DataTrain, thread_name);

            if self.data.DataResultDim.DataResultX is None:
                return();
            
            col_names_y = list(self.data.DataTrain.df_parm_y);
            threads_result[threads_cnt][0]  =  pd.DataFrame(self.data.DataResultDim.DataResultX.y_result, columns = col_names_y);
            
            if self.data.saveDataToPLC(threads_result,
                                       self.txdat1,
                                       thread_name,
                                       self.typ,
                                       self.model_1 + self.model_2,
                                       self.units_1 + self.units_2,
                                       self.epochs,
                                       self.ip_yesno):
                pass;

            
            stopTime = int(time.time_ns());
            self.logger.info ("Thread- %s, predict: %d [ms] " %(thread_name, int((stopTime - startTime)/1000000)));

            return();

        except FileNotFoundError as e:
            self.logger.error(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!");    
        except Exception as ex:
            self.logger.error(traceback.print_exc());
            traceback.print_exc();
            
#------------------------------------------------------------------------
# setter - getter
#------------------------------------------------------------------------
    def getModel(self):
        return self.model_;
    
    def getData(self):
        return self.data;

    def setData(self, data):
        self.data = data;

    def getTyp(self):
        return self.typ;

    def setTyp(self, typ):
        self.typ = typ;

    def getTxdat1(self):
        return(self.txdat1);
    
    def setTxdat1(self, val):
        self.txdat1 = val;
        
    def getTxdat2(self):
        return(self.txdat2);
    
    def setTxdat2(self, val):
        self.txdat2 = val;

    def getCurrentDate(self):
        return(self.current_date);
    
    def setCurrentDate(self, val):
        self.current_date = val;
        
#------------------------------------------------------------------------
# isPing ????
#------------------------------------------------------------------------
    def isPing(self):
        return self.data.isPing();


#------------------------------------------------------------------------
# Definice tridy NeuroDaemon
#------------------------------------------------------------------------
class NeuroDaemon():
    
    def __init__(self, 
                 pidf, 
                 path_to_result, 
                 model_1,
                 model_2,
                 epochs, 
                 batch, 
                 units_1,
                 units_2,
                 layers_1,
                 layers_2,
                 shuffling, 
                 txdat1, 
                 txdat2, 
                 actf, 
                 debug_mode,
                 current_date,
                 max_threads,
                 ilcnt,
                 ip_yesno,
                 lrn_rate
            ):

        self.logger         = logging.getLogger('ai-daemon');
        
        self.pidf           = pidf; 
        self.path_to_result = path_to_result;
        self.model_1        = model_1;
        self.model_2        = model_2;
        self.epochs         = epochs;
        self.batch          = batch;
        self.units_1        = units_1;
        self.units_2        = units_2;
        self.layers_1       = layers_1;
        self.layers_2       = layers_2;
        self.shuffling      = shuffling;
        self.txdat1         = txdat1; 
        self.txdat2         = txdat2;
        self.actf           = actf;
        self.debug_mode     = debug_mode;
        self.current_date   = current_date;
        self.typ            = "train";
        self.ilcnt          = ilcnt;
        
        self.train_counter  = 0;
        self.data           = None;
        self.threads_result = [];
        self.max_threads    = max_threads;
        self.ip_yesno       = ip_yesno;
        self.lrn_rate       = lrn_rate;

#------------------------------------------------------------------------
# start daemon pro parametr DENSE
# tovarna pro beh demona
#------------------------------------------------------------------------
    def printParms(self, debug_mode):

        time.sleep(1);
        self.logger.info("-----------------------------------");
        self.logger.info("epochs...........: %s" %(str(self.epochs)));
        self.logger.info("batch............: %s" %(str(self.batch)));
        self.logger.info("-----------------------------------");
        self.logger.info("model_1..........: %s" %(self.model_1));
        self.logger.info("model_2..........: %s" %(self.model_2));
        self.logger.info("units_1..........: %s" %(str(self.units_1)));
        self.logger.info("units_2..........: %s" %(str(self.units_2)));
        self.logger.info("layers_1.........: %s" %(str(self.layers_1)));
        self.logger.info("layers_2.........: %s" %(str(self.layers_2)));
        self.logger.info("-----------------------------------");
        self.logger.info("act.func.........: %s" %(self.actf));
        self.logger.info("txdat1...........: %s" %(self.txdat1));
        self.logger.info("txdat2...........: %s" %(self.txdat2));
        self.logger.info("ilcnt............: %s" %(str(self.ilcnt)));
        self.logger.info("shuffling........: %s" %(str(self.shuffling)));
        self.logger.info("debug mode.......: %s" %(self.debug_mode));
        self.logger.info("interpolate......: %s" %(str(self.ip_yesno)));
        self.logger.info("lrn_rate.........: %s" %(self.lrn_rate));
        self.logger.info("-----------------------------------");
        time.sleep(2);
        return;
                     
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def getLogFileHandlers(self, logger):

        i = 0;
        handlers = [];

        if logger is not None:
            for i in range(len(logger.handlers)):
                handler = logger.handlers[i];
                i += 1;
                if handler is not None:
                    handlers.append(handler.stream.fileno())
                else:
                    break;
            
            if self.logger.parent:
                handlers += self.getLogFileHandlers(logger.parent)
        else:
            pass;
                
        return (handlers);

#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def setLogHandler(self):

        progname = os.path.basename(__file__);

        logging.basicConfig(filename="./log/"+progname+".log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S');

        log_handler = logging.StreamHandler();
        logger = logging.getLogger("parent");
        logger.addHandler(log_handler)
        return(log_handler);
        

#------------------------------------------------------------------------
# mailto....
#------------------------------------------------------------------------
    def sendtoMSG(self, current_time, recipient, txdat1, txdat2, plc_isRunning):
        
        
        subject = "ai-daemon";
        if plc_isRunning:
            msg0 = "ai-daemon v rezimu run (stroj je zapnut)...\n";
            msg1 = "start v modu: train, cas:"+str(current_time)+"\n";
            msg2 = "treninkova mnozina, timestamp start :"+txdat1+"\n";
            msg3 = "                    timestamp stop  :"+txdat2+"\n";
        else:    
            msg0 = "ai-daemon v rezimu sleep (stroj je vypnut)...\n";
            msg1 = "start v modu: train, cas:"+str(current_time)+"\n";
            msg2 = "treninkova mnozina, timestamp start :"+txdat1+"\n";
            msg3 = "                    timestamp stop  :"+txdat2+"\n";

        msg = msg0+msg1+msg2+msg3 ;
        self.logger.info(msg);
        #webbrowser.open("mailto:?to="+ recipient + "&subject=" + subject + "&body=" + msg, new=1);
        
#------------------------------------------------------------------------
# start daemon pro parametr LSTM, GRU, DENSE a CONV1D
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemonLSTM(self, threads_result, threads_cnt):

        current_day  = datetime.today().strftime('%A');   #den v tydnu
        # new LSTM layer
        thread_name = threads_result[threads_cnt][1];
        epochs  = self.epochs;
        units_1 = self.units_1;
        units_2 = self.units_2;
        timestamp_format = "%Y-%m-%d %H:%M:%S";
        plc_isRunning = True;
        sleep_interval = 0;   #[s]
        self.printParms(self.debug_mode);
        
        if threads_cnt > 0:
            epochs   = epochs  + 1;
            units_1  = units_1 + 1;
            units_2  = units_2 - 1;

        neural = NeuronLayerLSTM(path_to_result = path_to_result, 
                                 typ            = "train", 
                                 model_1        = self.model_1, 
                                 model_2        = self.model_2, 
                                 epochs         = epochs, 
                                 batch          = self.batch,
                                 txdat1         = self.txdat1,
                                 txdat2         = self.txdat2,
                                 units_1        = units_1,
                                 units_2        = units_2,
                                 layers_1       = self.layers_1,
                                 layers_2       = self.layers_2,
                                 shuffling      = self.shuffling,
                                 actf           = self.actf, 
                                 debug_mode     = self.debug_mode,
                                 current_date   = self.current_date,
                                 thread_name    = thread_name,
                                 ilcnt          = self.ilcnt,
                                 ip_yesno       = self.ip_yesno,
                                 lrn_rate       = self.lrn_rate
                                  
                            );
                            
        neural.setData(self.data);
        plc_isRunning = self.data.isPing();

        if plc_isRunning is True:
            sleep_interval =   0;         #  0 [s]
        else:
            sleep_interval = 600;         #600 [s]
            
        if self.debug_mode is True:
            sleep_interval =   0;         #  0 [s]   

        current_date =  datetime.now().strftime(timestamp_format);
        current_day  = datetime.today().strftime('%A');   #den v tydnu
        given_time = datetime.strptime(current_date, timestamp_format);
        
        final_time_day1 = given_time - timedelta(days=1);
        final_time_day1_str = final_time_day1.strftime(timestamp_format);

        final_time_day30 = given_time - timedelta(days=30);
        final_time_day30_str = final_time_day30.strftime(timestamp_format);

        #jsou nabity timestampy pro vyber mnoziny trenikovych dat?
        if neural.getTxdat1() == "":
            neural.setTxdat1(final_time_day30_str);
            self.txdat2 = final_time_day30_str;
            
        if neural.getTxdat2() == "":
            neural.setTxdat2(final_time_day1_str);
            self.txdat2 = final_time_day1_str;

        self.current_date = current_date;
        neural.setCurrentDate(current_date);
        self.data.setCurrentDate(current_date);
            
        txdat1 = neural.getTxdat1();
        txdat2 = neural.getTxdat2();

        self.sendtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);


        #predikcni beh
        while True:
            
            current_date =  datetime.now().strftime(timestamp_format);
            plc_isRunning = self.data.isPing();
            
            #prechod na novy den - pretrenuj sit...
            if (datetime.today().strftime('%A') not in current_day):
                self.logger.info("start train:"+ current_date+"");
                neural.setTyp("train");
                
                current_day = datetime.today().strftime('%A');
                given_time = datetime.strptime(current_date, timestamp_format);
                final_time = given_time - timedelta(days=1);
                final_time_str = final_time.strftime(timestamp_format);
                txdat1 = neural.getTxdat1();
                txdat2 = neural.getTxdat2();
                
                self.txdat2 = final_time_str;
                neural.setTxdat2(final_time_str);
                self.current_date = current_date;
                neural.setCurrentDate(current_date);
                self.data.setCurrentDate(current_date);

                self.sendtoMSG(current_date, "plukasik@tajmac-zps.cz", txdat1, txdat2, plc_isRunning);

            if plc_isRunning and not self.debug_mode:
                if neural.getTyp() == "train":
                    self.logger.info("PLC ON:"+ current_date + " mode train, thread cnt: " + str(threads_cnt + 1)+"");
                else:
                    self.logger.info("PLC ON:"+ current_date + " mode predict, thread cnt: " + str(threads_cnt +1)+"");
                    
                neural.neuralNetworkLSTMexec(threads_result, threads_cnt);
                neural.setTyp("predict");
                sleep_interval =  0;         #  0 [s]

            if not plc_isRunning and not self.debug_mode:
                self.logger.info("PLC OFF:"+ current_date+"");
                neural.setTyp("train");
                sleep_interval = 600;        #600 [s]
                
            if self.debug_mode:
                neural.neuralNetworkLSTMexec(threads_result, threads_cnt);
                neural.setTyp("predict");
                sleep_interval =   0;        #  0 [s]

            time.sleep(sleep_interval);

            
#------------------------------------------------------------------------
# factory to daemon - getPid
#------------------------------------------------------------------------
    def getPid(self, pidf_path):
        
        
        pid = None
        pidfile = None

        if os.name == 'nt':
            pidf_path = Path(pidf_path)    #kvuli windowsum
        
        try:
            pidfile = open(pidf_path, 'r');
            
        except IOError as ex:
            if ex.errno == errno.ENOENT:
                pass;
            else:
                raise;

        if pidfile:
        
            line = pidfile.readline().strip()
            try:
                pid = int(line)
            except ValueError:
                raise PIDFileParseError(u"PID file %(pidfile_path)r contents invalid" % vars())
            
            pidfile.close()

        return pid;
            
            
            
#------------------------------------------------------------------------
# Factory to daemon - setPid
#------------------------------------------------------------------------
    def setPid(self, pidf_path):
        
        #Open stream to pidfile
        try:
            open_flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY);
            open_mode = (((os.R_OK | os.W_OK) << 6) | ((os.R_OK) << 3) | ((os.R_OK)));
            pidfile_fd = os.open(pidf_path, open_flags, open_mode);
            pidfile = os.fdopen(pidfile_fd, 'w');
        
        #Get pid from sys
            pid = os.getpid();
            line = u"%(pid)d\n" % vars();
            pidfile.write(line);
            pidfile.close();
            
            return True;
            
        except OSError as ex:
            return False;
        
            
#------------------------------------------------------------------------
# Factory to daemon - removePid
#------------------------------------------------------------------------
    def removePid(self, pidfile_path):
        try:
            os.remove(pidfile_path);
            
        except OSError as ex:
            if ex.errno == errno.ENOENT:
                pass
            else:
                raise
        
#------------------------------------------------------------------------
# Factory to daemon - run daemon, parms = DENSE || LSTM 
#------------------------------------------------------------------------
    def runDaemon(self, threads_max_cnt):

        threads_cnt = 0;
        threads_result = [];
        data = None;

        try:
            self.data = DataFactory(path_to_result=self.path_to_result, 
                                    batch=self.batch,
                                    debug_mode=self.debug_mode,
                                    current_date=self.current_date);
        
                
            parms = ["train",      #self.typ <train, predict>
                     self.model_1 + "_" + self.model_2,
                     self.epochs,
                     self.units_1 + self.units_2,
                     self.layers_1 + self.layers_2,
                     self.batch,
                     self.actf,
                     str(self.shuffling), 
                     self.txdat1, 
                     self.txdat2,
                     str(self.current_date)];
            
            self.data.setParms(parms);
            
        except Exception as ex:
            traceback.print_exc();
            self.logger.error(traceback.print_exc());
            
        
        self.logger.info("Thread executor start, thread cnt:("+str(threads_max_cnt)+") model: "
                         + self.model_1
                         +"/"
                         + self.model_2);
        try:
            for i in range(threads_max_cnt):
                #prepare result array
                arr = [None,"_"+self.model_1+self.model_2+"_"+str(i)];
                threads_result.append(arr);
                
            executor = ThreadPoolExecutor(max_workers = threads_max_cnt);
            futures = [executor.submit(self.runDaemonLSTM,
                                       threads_result,
                                       threads_cnt
                                       )
                       for threads_cnt in range(threads_max_cnt)];
            
        except Exception as exc:
                self.logger.error(traceback.print_exc());
                
#------------------------------------------------------------------------
# info daemon
#------------------------------------------------------------------------
    def info(self):
        self.logger.info("daemon pro sledovani a kompenzaci teplotnich zmen stroje");
        return;

#------------------------------------------------------------------------
# daemonize...
#    do the UNIX double-fork magic, see Stevens' "Advanced
#    Programming in the UNIX Environment" for details (ISBN 0201563177)
#    http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
#
#------------------------------------------------------------------------
    def daemonize(self, pidf):
        
        self.logger.debug("daemonize.....");
        handler = self.setLogHandler();

        current_dir = os.getcwd()
        self.logger.debug("current directory: %s, pidf directory: %s " %(os.getcwd(), pidf));
        
        try:
            handlers = self.getLogFileHandlers(self.logger);
            pidfile_ = daemon.pidfile.TimeoutPIDLockFile(pidf);
            
            context  = daemon.DaemonContext(working_directory   = current_dir,
                                            umask               = 0o002,
                                            pidfile             = pidfile_,
                                            detach_process      = True,                  
                                            files_preserve      = handlers);
                                            
            context.signal_map = {
                signal.SIGHUP:  'terminate',
        }

        except (Exception, getopt.GetoptError)  as ex:
            traceback.print_exc();
            self.logger.error("Start daemon : %s sys.exit(1).... " %ex);
            sys.exit(1);

        
        return(context);
            
#------------------------------------------------------------------------
# start daemon
#------------------------------------------------------------------------
    def start(self):
        # Kontrola existence pid - daemon run...
        self.logger.debug("ai-daemon, start...");
        
        pid = self.getPid(self.pidf);
        if pid is None:
            try:
                context = self.daemonize(self.pidf);
                context.open();
                with context:
                    self.runDaemon(threads_max_cnt = self.max_threads);
                    
            except (Exception, getopt.GetoptError)  as ex:
                traceback.print_exc();
                self.logger.error("Start daemon : %s sys.exit(1).... " %(ex));
                sys.exit(1);
                
        else:
            messge = "pid procesu %d existuje!!!. Daemon patrne bezi - exit(1)";
            self.logger.error(message %(pid));
            sys.exit(1);                                                                                                    

#------------------------------------------------------------------------
# start daemon
#------------------------------------------------------------------------
    def run(self):
        # Kontrola existence pid - daemon run...
        self.logger.debug("ai-daemon, run...");
        
        pid = self.getPid(self.pidf);
        if pid is None:
            self.runDaemon(threads_max_cnt = self.max_threads);
        else:        
            message = "pid procesu %d existuje!!!. ai-daemon patrne bezi jako demon!!! - exit(1)";
            self.logger.error(message % pid);                                                                       
            sys.exit(1);                                                                                                    

#------------------------------------------------------------------------
# stop daemon
#------------------------------------------------------------------------
    def stop(self):
        self.logger.debug("ai-daemon, stop...");
        
        pid = self.getPid(self.pidf);
                                                                                                                            
        if pid is None:
            return;                                                                                                             
            message = "pid procesu neexistuje!!!. Daemon patrne nebezi - exit(1)";                                         
            self.logger.error(message);                                                                       
            sys.exit(1);
        else:
            self.removePid(self.pidf)                                                                                                        
            message = "pid procesu %d existuje!!!. Daemon %d stop....";                                         
            self.logger.info(message % pid);                                                                       
            sys.exit(0);

#------------------------------------------------------------------------
# stop daemon
#------------------------------------------------------------------------
    def restart(self):
        self.stop();
        self.start();

#------------------------------------------------------------------------
# getter, setter metody
#------------------------------------------------------------------------
    def getDebug(self):
        return self.debug_mode;
    
    def setDebug(self, debug_mode):
        self.debug_mode = debug_mode;

        
#------------------------------------------------------------------------
# MAIN CLASS
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# definice loggeru - parametry nacteny z ./cfg/log.cfg
#------------------------------------------------------------------------
def setLogger(logf):

    logging.config.fileConfig(fname=Path("./cfg/log.cfg"));
    logger = logging.getLogger("ai");
    return(logger);

#------------------------------------------------------------------------
# saveModelToArchiv - zaloha modelu, spusteno jen pri parametru train
#------------------------------------------------------------------------

def saveModelToArchiv(model, dest_path, data):

    axes = np.array([data.DataTrainDim.DataTrain.axis]);
    src_dir  = "./models/model_"+model+"_";
    dest_dir = "/models/model_"+model+"_";
    try:
        if data.DataTrainDim.DataTrain is None:
            pass;
        else:    
            src_dir_  = src_dir + axes[0];
            dest_dir_ = dest_path + dest_dir + axes[0];
            files = os.listdir(Path(src_dir_));
            shutil.copytree(Path(src_dir_), Path(dest_dir_));
            
        return(0);    
   
    except Exception as ex:
        traceback.print_exc();
        self.logger.error(traceback.print_exc());
 


#------------------------------------------------------------------------
# setEnv
#------------------------------------------------------------------------
def setEnv(path, model_1, model_2, type):

        progname = os.path.basename(__file__);
        current_date =  datetime.now().strftime("%Y-%m-%d_%H.%M.%S");

        path_1 = path+model_1+model_2;
        path_2 = path_1+"/"+current_date+"_"+type;
        
        try: 
            os.mkdir(Path("./log"));
        except OSError as error: 
            pass;

        try: 
            os.mkdir(Path("./cfg"));
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir(Path("./pid"));
        except OSError as error: 
            pass; 
 
        try: 
            os.mkdir(Path("./result"));
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir(Path("./result/plc_archiv"));
        except OSError as error: 
            pass; 
 
        try: 
            os.mkdir(Path("./temp"));
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir(Path(path_1));
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir(Path(path_2));
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir(Path(path_2+"/src"));
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir(Path("./models"));
        except OSError as error: 
            pass;

        try: 
            os.mkdir(Path("./br_data"));
        except OSError as error: 
            pass;

        try: 
            os.mkdir(Path("./br_data/getplc"));
        except OSError as error: 
            pass;

        try:
            shutil.copy("./py-src/"+progname, Path(path_2+"/src"));
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii zdrojoveho kodu.", error)
        except:
            print("Chyba pri kopii zdrojoveho kodu.")

        try:
            shutil.copy("./cfg/ai-parms.cfg", Path(path_2+"/src"));
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii ai-parms.cfg.", error)
        except:
            print("Chyba pri kopii ai-parms.cfg.")
            
        try:
            shutil.copy("ai-daemon.sh", Path(path_2+"/src"));
            shutil.copy("./py-src/ai-daemon.py", Path(path_2+"/src"));
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii ai-parms.txt.", error)
        except:
            print("Chyba pri kopii ai-parms.txt.")
            
         
        return path_2, current_date;    

#------------------------------------------------------------------------
# Exception handler
#------------------------------------------------------------------------
def exception_handler(exctype, value, tb):
    logger.error(exctype)
    logger.error(value)
    logger.error(traceback.extract_tb(tb))
    
#------------------------------------------------------------------------
# Signal handler
#------------------------------------------------------------------------
def signal_handler(self, signal, frame):
    #Catch Ctrl-C and Exit
    sys.exit(1);

#------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------
def help (activations):
    print("HELP:");
    print("------------------------------------------------------------------------------------------------------ ");
    print("pouziti: <nazev_programu> <arg-1> <arg-2> <arg-3>,..., <arg-n>");
    print(" ");
    print("        --help            list help ")
    print(" ");
    print(" ");
    print("        --model_1         model prvni skryte vrstvy neuronove site")
    print("                                'DENSE'  - zakladni model site - nejmene narocny na system")
    print("                                'GRU'    - Narocny model rekurentni site s feedback vazbami")
    print("                                'LSTM'   - Narocny model rekurentni site s feedback vazbami")
    print("                                'CONV1D' - konvolucni filtr")
    print(" ");
    print(" ");
    print("        --model_2         model druhe skryte vrstvy neuronove site")
    print("                                'DENSE'  - zakladni model site - nejmene narocny na system")
    print("                                'GRU'    - Narocny model rekurentni site s feedback vazbami")
    print("                                'LSTM'   - Narocny model rekurentni site s feedback vazbami")
    print("                                'CONV1D' - konvolucni filtr")
    print("                                ' '      - druha skryta vrstva disable...")
    print(" ");
    print("        --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
    print("                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
    print("                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
    print("                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
    print("                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
    print(" ");
    print("        --batch           pocet vzorku do predikce - cislo v intervalu <16,256> pro rezim nodebug ")
    print("                                                   - cislo v intervalu <16,32768> pro rezim nodebug ")
    print("                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
    print("                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
    print("                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
    print("                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
    print(" ");
    print("        --units_1         pocet neuronu v prvni sekci skryte vrstvy <32, 1024>");
    print(" ");
    print("        --units_2         pocet neuronu v druhe sekci skryte vrstvy <0, 1024>");
    print("                          --units_2 = 0 druha skryta vrstva disable...");
    print(" ");
    print("        --layers_1        pocet vrstev prvni sekci skryte vrstvy <0, 6>")
    print(" ");
    print("        --layers_2        pocet vrstev druhe sekci skryte vrstvy <0, 6>")
    print(" ");
    print(" ");
    print("        --actf            Aktivacni funkce - jen pro parametr DENSE")
    print("                                 U LSTM, GRU a BIDI se neuplatnuje.")
    print("                                 Pokud actf neni uveden, je implicitne nastaven na 'elu'."); 
    print("                                 U site GRU, LSTM a BIDI je pouzito defaultni nastaveni  ");
    print(" ");
    print(" ");
    print("        --txdat1          timestamp zacatku datove mnoziny pro predict, napr '2022-04-09 08:00:00' ")
    print(" ");
    print("        --txdat2          timestamp konce   datove mnoziny pro predict, napr '2022-04-09 12:00:00' ")
    print(" ");
    print("                                 parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere");
    print("                                 se v uvahu cela mnozina dat k trenovani, to znamena:");
    print("                                 od pocatku mereni: 2022-01-01 00:00:00 ");
    print("                                 do konce   mereni: current timestamp() - 1 [den] ");
    print(" ");
    print("        --ilcnt           vynechavani vet v treninkove mnozine - test s jakym poctem vet lze");
    print("                                 jeste solidne predikovat:  1 = nacteny vsechny vety");
    print("                                                            2 = nactena kazda druha");
    print("                                                            3 = nactena kazda treti...");
    print(" ");
    print("        --shuffle         nahodne promichani dat ");
    print(" ");
    print("                                 shuffle=True ");
    print("                                 shuffle=False");
    print(" ");
    print("        --interpolate     interpolace treninkovych i predikcnich dat ");
    print("                                 splinem univariateSpline ");
    print(" ");
    print("                                 interpolate = True");
    print("                                 interpolate = False");
    print(" ");
    print(" ");
    print(" ");
    print("Parametry treninkove a predikcni mnoziny jsou v ./cfg/ai-parms.cfg.");
    print("Pozor!!! Parametry v treninkovem tenzoru (df_parm_X) se musi(!) ");
    print("v casti teplot, shodovat s parametry k predikci (df_parm_x).  ");
    print(" ");
    print("Syntaxe v ai-parms.cfg je nasledujici: ");
    print("#--------------------------------------------------------------------------------------------");
    print("#Tenzor predlozeny k predikci                                                                ");
    print("#--------------------------------------------------------------------------------------------");
    print("df_parmx = temp_lo03, temp_st02, temp_st06, temp_st07, temp_S1, temp_vr05,...");
    print("#--------------------------------------------------------------------------------------------");
    print("#Tenzor predlozeny k treninku                                                                ");
    print("#--------------------------------------------------------------------------------------------");
    print("df_parmX = dev_y4, dev_z4, temp_lo03, temp_st02, temp_st06, temp_st07, temp_S1, temp_vr05,...");
    print("POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy vstupniho datoveho CSV souboru");
    print(" ");
    print("(C) GNU General Public License, autor Petr Lukasik , 2022 ");
    print(" ");
    print("Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)");
    print("              miniconda3,");
    print("              python 3.9, tensorflow 2.8, mathplotlib,  ");
    print("              tensorflow 2.8,");
    print("              mathplotlib,  ");
    print("              scikit-learn-intelex,  ");
    print("              pandas,  ");
    print("              numpy,  ");
    print("              keras   ");
    print(" ");
    print(" ");
    print("Povolene aktivacni funkce: ");
    print(tabulate(activations, headers=['Akt. funkce', 'Popis']));

    return();

#------------------------------------------------------------------------
# kontrola zda byla zadana platna aktivacni funkce 
# ze seznamu activations...
#------------------------------------------------------------------------
def checkActf(actf, activations):
    

    for i in activations:
        if i[0] in actf:
            return(True);

    return(False);


#------------------------------------------------------------------------
# Test GPU 
#------------------------------------------------------------------------
def testGPU(logger):

    gpu_devices = tf.config.list_physical_devices("GPU");
    
    if gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True);

            tf.config.set_visible_devices(gpu_devices[0], "GPU");
            logical_gpus = tf.config.list_logical_devices("GPU");
            
            tf.debugging.set_log_device_placement(True)
            
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
            c = tf.matmul(a, b);
            
            print("----------------------------------------------------------------");
            print("Verze TensorFlow :", tf.__version__);
            print("Pocet fyzickych GPU: ", len(gpu_devices)," Pocet logickych GPU: " , len(logical_gpus));
            print("----------------------------------------------------------------");
                
            return(True);        

        except:
            print("Neplatne zarizeni :", gpu_devices);
            pass;
    
    print("----------------------------------------------------------------");
    print("Verze TensorFlow :", tf.__version__);
    print("Neni k dispozici  zadna GPU... ");
    print("----------------------------------------------------------------");
    return(False);        



#------------------------------------------------------------------------
# main
# preddefinovane hyperparametry neuronove site
# ["typ=", "model=", "epochs=", "batch=", "units=", "shuffle=","actf=",
#  "txdat1=","txdat2=", "help="])
#   parm0  = sys.argv[0];   - nazev programu
#   typ    = "train";       - typ behu site <train, predict>
#   model  = "DENSE";       - model site
#   epochs = 128;           - pocet treninkovych metod
#   batch  = 128;           - velikost davky
#   units  = 512;           - pocet neuronu
#   txdat1 = "";            - timestamp start -
#                               vyber dat treninkove a validacni mnoziny
#   txdat2 = "";            - timestamp stop  -
#                               vyber dat treninkove a validacni mnoziny
#   shuffling = True;       - promichat nahodne data <True, False>
#   actf = "tanh";          - aktivacni funkce
#   pid = 0;
#------------------------------------------------------------------------
def main(argv):
    
    global path_to_result;
    global current_date;
    global g_window;
    g_window = 16;
    
    path_to_result = "./result";
    current_date ="";
    pidf = "./pid/ai-daemon.pid"
    logf = "./log/ai-daemon.log"
    
    #registrace signal handleru
    signal.signal(signal.SIGINT, signal_handler);
    
    '''
    debug_mode - parametr pro ladeni programu, pokud nejedou OPC servery, 
    data se nacitaji ze souboru, csv_file = "./br_data/predict-debug.csv", 
    ktery ma stejny format jako data z OPC serveruuu.
    
    debug_mode se zaroven posila do DataFactory.getData, kde se rozhoduje,
    zda se budou data cist z OPC a nebo z predict-debug souboru. 
    V predict-debug souboru jsou data nasbirana z OPC serveru 
    v intervalu<2022-07-19 08:06:01, 2022-07-19 11:56:08> v sekundovych
    vzorcich.
    '''
    
#-----------------------------------------------------------------------------
# implicitni  parametry - plati pokud nejsou prebity parametry ze sys.argv[1:]
#-----------------------------------------------------------------------------
    parm0          = sys.argv[0];        
    model_1        = "DENSE";
    model_2        = "";
    epochs         = 64;
    batch          = 256;
    units_1        = 71;
    units_2        = 0;
    txdat1         = "2022-01-01 00:00:01";
    txdat2         = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S");
    shuffling      = True;
    actf           = "elu";
    pid            = 0;
    status         = "";
    startTime      = datetime.now();
    type           = "train";
    debug_mode     = False;
    max_threads    = 1;
    ilcnt          = 1; # reccnt
    layers_1       = 2;
    layers_2       = 0;
    ip_yesno       = False;
    lrn_rate       = 0.0005;     #<0.0002 - 0.002>
    opts           = "";
    
        
#-----------------------------------------------------------------------------
# seznam povolenych aktivacnich funkci
#-----------------------------------------------------------------------------
    activations = [["deserialize", "Returns activation function given a string identifier"],
                   ["elu", "Exponential Linear Unit"],
                   ["exponential", "Exponential activation function"],
                   ["gelu", "Gaussian error linear unit (GELU) activation function"],
                   ["get", "Returns function"],
                   ["hard_sigmoid", "Hard sigmoid activation function"],
                   ["linear", "Linear activation function (pass-through)"],
                   ["relu", "Rectified linear unit activation function"],
                   ["selu","Scaled Exponential Linear Unit"],
                   ["serialize","Returns the string identifier of an activation function"],
                   ["sigmoid","Sigmoid activation function: sigmoid(x) = 1 / (1 + exp(-x))"],
                   ["softmax","Softmax converts a vector of values to a probability distribution"],
                   ["softplus","Softplus activation function: softplus(x) = log(exp(x) + 1)"],
                   ["softsign","Softsign activation function: softsign(x) = x / (abs(x) + 1)"],
                   ["swish","Swish activation function: swish(x) = x * sigmoid(x)"],
                   ["tanh","Hyperbolic tangent activation function"],
                   ["None","pro GRU a LSTM site"]];

    models_1    = ["DENSE","LSTM","GRU","CONV1D"];
    models_2    = ["DENSE","LSTM","GRU","CONV1D",""];


    logger = setLogger(logf);
    
#-----------------------------------------------------------------------------
# init objektu daemona
#-----------------------------------------------------------------------------
    try:
        logger.info("ai-daemon start...");

#-----------------------------------------------------------------------------
# kontrola platne aktivacni funkce
#-----------------------------------------------------------------------------
        
        if not checkActf(actf, activations):
            print("Chybna aktivacni funkce - viz help...");
            help(activations)
            sys.exit(1);

#-----------------------------------------------------------------------------
# kontrola aktivni GPU?
#-----------------------------------------------------------------------------
            
        testGPU(logger);            
        txdat_format = "%Y-%m-%d %h:%m:%s"

#-----------------------------------------------------------------------------
# zpracovani argumentu
#-----------------------------------------------------------------------------
        try:
            
            opts, args = getopt.getopt(sys.argv[1:],
                                       "hs:db:p:l:m1:m2:e:b:u1:u2:l1:l2:a:t1:t2:il:sh:ip:lr:h:x",
                                      ["status=",
                                       "dbmode=", 
                                       "pidfile=", 
                                       "logfile=", 
                                       "model_1=", 
                                       "model_2=", 
                                       "epochs=", 
                                       "batch=", 
                                       "units_1=", 
                                       "units_2=", 
                                       "layers_1=", 
                                       "layers_2=", 
                                       "actf=", 
                                       "txdat1=", 
                                       "txdat2=", 
                                       "ilcnt=", 
                                       "shuffle=", 
                                       "interpolate=", 
                                       "lrnrate=", 
                                       "help="]
                                    );
            
        except getopt.GetoptError:
            print("Err: parsing parms :", opts);
            sys.exit(1);

        for opt, arg in opts:
            
            if opt in ["-s","--status"]:
                status = arg;
                
                if ("start"   in status or
                   "stop"     in status or
                   "restart"  in status or
                   "status"   in status or
                   "run"      in status):
                    pass;
                else:
                    print("Err: <start, stop, restart, run, status>");
                    help(activations);
                    sys.exit(1);    
                    
            elif opt in ["-db","--dbmode"]:    #debug nodebug....
                
                if "true" in arg.lower() or "debug" in arg.lower():
                    debug_mode = True;
                    txdat2 = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d %H:%M:%S");
                else:
                    debug_mode = False;

            
            elif opt in ["-p","--pidfile"]:
                pidf = arg;
                
            elif opt in ["-l","--logfile"]:
                logf = arg;
                
        
            elif opt in ("-m1", "--model_1"):
                model_1 = arg.upper();
                if model_1 not in models_1:
                    print("Err: 'model_1', in ", models);
                    help(activations);
                    sys.exit(1);    

            elif opt in ("-m2", "--model_2"):
                model_2 = arg.upper();
                if model_2 not in models_2:
                    print("Err: 'model_2', in ", models);
                    help(activations);
                    sys.exit(1);    
                
            elif opt in ("-e", "--epochs"):
                try:
                    r = range(16-1, 256+1);
                    epochs = int(arg);
                    if epochs not in r:
                        print("Err: 'epochs' in <16, 256>");
                        help(activations);
                        sys.exit(1)    
                        
                except:
                    print("Chyba: parametr 'epochs' musi byt integer v rozsahu <16, 128>");
                    help(activations);
                    sys.exit(1);
                        
            elif opt in ("-u1", "--units_1"):
                try:
                    r = range(8-1, 1024+1);
                    units_1 =  int(arg);
                    if units_1 not in r:
                        print("Err: 'units_1' in <0, 1024>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Err: 'units_1' in <0, 1024>");
                    help(activations);
                    sys.exit(1);

            elif opt in ("-u2", "--units_2"):
                try:
                    r = range(-1, 1024+1);
                    units_2 =  int(arg);
                    if units_2 not in r:
                        print("Err: 'units_2' in <0, 1024>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Err: 'units_2' in <0, 1024>");
                    help(activations);
                    sys.exit(1);

                    
            elif opt in ("-l1", "--layers_1"):
                try:
                    r = range(-1, 5+1);
                    layers_1 = int(arg);
                    if layers_1 not in r:
                        print("Err: 'layers_1' in <0, 5>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Err: 'layers_1' in <0, 5>");
                    help(activations);
                    sys.exit(1);
                    
            elif opt in ("-l2", "--layers_2"):
                try:
                    r = range(-1, 5+1);
                    layers_2 = int(arg);
                    if layers_2 not in r:
                        print("Err: 'layers_2' in <0, 5>");
                        help(activations);
                        sys.exit(1);    
                except:    
                    print("Err: 'layers_2' in <0, 5>");
                    help(activations);
                    sys.exit(1);
                    

            elif opt in ["-af","--actf"]:
                actf = arg.lower();
                if not checkActf(actf, activations):
                    print("Err: aktivacni funkce - viz help...");
                    help(activations)
                    sys.exit(1);
                        
            elif opt in ("-b", "--batch"):
                try:
                    r = range(16-2, 32768+2);
                    batch = int(arg);
                    if batch not in r:
                        if debug_mode:
                            print("Err: 'batch' in <16, 32768>");
                        else:    
                            print("Err: 'batch' in <16, 256>");
                        help(activations);
                        sys.exit(1)    
                except:    
                    if debug_mode:
                        print("Err: 'batch' in <16, 32768>");
                    else:    
                        print("Err: 'batch' in <16, 256>");
                    help(activations);
                    sys.exit(1)
                    

            elif opt in ["-t1","--txdat1"]:
                txdat1 = arg.replace("_", " ");
                if txdat1:
                    try:
                        res = bool(parser.parse(txdat1));
                    except ValueError:
                        print("Err: format txdat1, YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    

            elif opt in ["-t2","--txdat2"]:
                txdat2 = arg.replace("_", " ");
                if txdat2:
                    try:
                        res = bool(parser.parse(txdat2));
                    except ValueError:
                        print("Err: format txdat2, YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    

            elif opt in ["-il","--ilcnt"]:
                try:
                    r = range(-1, 16+1);
                    ilcnt  = int(arg);
                    if ilcnt not in r:
                        print("Err: parametr 'ilcnt' in <1, 16>");
                        help(activations);
                        sys.exit(1)    
                except:    
                    print("Err: parametr 'ilcnt' in <1, 16>");
                    help(activations);
                    sys.exit(1)

            elif opt in ("-sh", "--shuffle"):
                sh = arg.upper();
                if "TRUE" in sh:
                    shuffling=True;
                else:
                    shuffling=False;

            elif opt in ("-ip", "--interpolate"):
                sh = arg.upper();
                if "TRUE" in sh:
                    ip_yesno = True;
                else:
                    ip_yesno = False;
         
            elif opt in ["-lr","--lrnrate"]:
                try:
                    r = range(1, 21);
                    lrn_rate  = float(arg);
                    lrn_ratemult = int(lrn_rate*10000);
        
                    if lrn_ratemult not in r:
                        print("Err: parametr 'lrn-rate' in <0.0002, 0.002>");
                        help(activations);
                        sys.exit(1)    
                except:    
                    print("Err: parametr 'lrn-rate' in <0.0002, 0.002>");
                    help(activations);
                    sys.exit(1)
                    
            elif opt in ["-h","--help"]:
                help(activations);
                sys.exit(0);

#-----------------------------------------------------------------------------
# reset layer 2 if model_2 is blank or units_2 = 0 or layers_2 = 0
#-----------------------------------------------------------------------------
        if model_2 == "" or units_2 == 0 or layers_2 == 0:
            model_2  = "";
            units_2  = 0;
            layers_2 = 0;

#-----------------------------------------------------------------------------
# chyba - parametry
#-----------------------------------------------------------------------------
        if len(sys.argv) < 2:
            help(activations);
            sys.exit(1);
            
#-----------------------------------------------------------------------------
# set environment
#-----------------------------------------------------------------------------
        path_to_result, current_date = setEnv(path=path_to_result,
                                              model_1=model_1,
                                              model_2=model_2,
                                              type=type);
    
#-----------------------------------------------------------------------------
# Start status - run as daemon 
#-----------------------------------------------------------------------------

        if "start" in status:
            try:
                
                # new NeuroDaemon
                daemon_ = NeuroDaemon(pidf           = pidf,
                                      path_to_result = path_to_result,
                                      model_1        = model_1,
                                      model_2        = model_2,
                                      epochs         = epochs,
                                      batch          = batch,
                                      units_1        = units_1,
                                      units_2        = units_2,
                                      layers_1       = layers_1,
                                      layers_2       = layers_2,
                                      shuffling      = shuffling,
                                      txdat1         = txdat1, 
                                      txdat2         = txdat2,
                                      actf           = actf,
                                      debug_mode     = debug_mode,
                                      current_date   = current_date,
                                      max_threads    = max_threads,
                                      ilcnt          = ilcnt,
                                      ip_yesno       = ip_yesno,
                                      lrn_rate       = lrn_rate
                                );
       
                daemon_.info();
                
                if debug_mode:
                    logger.info("ai-daemon run v debug mode...");
                    #daemon_.runDaemon(threads_max_cnt = max_threads);
                    daemon_.start();
                else:    
                    logger.info("ai-daemon start...");
                    #daemon_.runDaemon(threads_max_cnt = max_threads);
                    daemon_.start();
                    
            except:
                traceback.print_exc();
                logger.info(str(traceback.print_exc()));
                logger.info("ai-daemon start exception...");
                pass

#-----------------------------------------------------------------------------
# Run status - run as prog 
#-----------------------------------------------------------------------------
        elif "run" in status:
            try:
                
                # new NeuroDaemon
                daemon_ = NeuroDaemon(pidf           = pidf,
                                      path_to_result = path_to_result,
                                      model_1        = model_1,
                                      model_2        = model_2,
                                      epochs         = epochs,
                                      batch          = batch,
                                      units_1        = units_1,
                                      units_2        = units_2,
                                      layers_1       = layers_1,
                                      layers_2       = layers_2,
                                      shuffling      = shuffling,
                                      txdat1         = txdat1, 
                                      txdat2         = txdat2,
                                      actf           = actf,
                                      debug_mode     = debug_mode,
                                      current_date   = current_date,
                                      max_threads    = max_threads,
                                      ilcnt          = ilcnt,
                                      ip_yesno       = ip_yesno,
                                      lrn_rate       = lrn_rate
                                );
       
                daemon_.info();
                
                if debug_mode:
                    logger.info("ai-daemon run v debug mode...");
                    daemon_.run();
                else:    
                    logger.info("ai-daemon run...");
                    daemon_.run();
                    
            except:
                traceback.print_exc();
                logger.info(str(traceback.print_exc()));
                logger.info("ai-daemon start exception...");
                pass

#-----------------------------------------------------------------------------
# Stop status - exit  
#-----------------------------------------------------------------------------
        elif "stop" in status:
            logger.info("ai-daemon stop...");
            daemon_.stop();
            stopTime = datetime.now();
            logger.info("cas vypoctu: " + str(stopTime - startTime) +" [h:m:s.ms]" );
            logger.info("stop obsluzneho programu pro demona - ai-daemon...");
            sys.exit(0);
            
        elif "restart" in status:
            logger.info("ai-daemon restart...");
            daemon_.restart()
            
        elif "status" in status:
            try:
                pf = file(pidf,'r');
                pid = int(pf.read().strip())
                pf.close();
            except IOError:
                pid = None;
            except SystemExit:
                pid = None;
            if pid:
                logger.info("Daemon ai-daemon je ve stavu run...");
            else:
                logger.info("Daemon ai-daemon je ve stavu stop....");
        else:
            logger.info("Neznamy parametr:<"+status+">");
            sys.exit(0)
        
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        #logger.error(traceback.print_exc());
        help(activations);

    finally:    
        stopTime = datetime.now();
        logger.info("cas vypoctu: " + str(stopTime - startTime) +" [h:m:s]" );
        logger.info("stop obsluzneho programu pro demona - ai-daemon...");
        sys.exit(0);



#-----------------------------------------------------------------------------
# main entry point
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async";
    print("getenv :", os.getenv("TF_GPU_ALLOCATOR"));
        
    main(sys.argv[1:]);
    

