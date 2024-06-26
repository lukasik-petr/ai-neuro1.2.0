----------------------------------------------------------------------
   Windows
----------------------------------------------------------------------
ai-daemon.bat - start demona v rezimu "nodebug". Ve windowsech nelze
              spustit program jako daemon, proto je spusten jako normalni
	      program, s nekonecnou smyckou. Pozor vsak, pri restartu
	      systemu se tento program sam nespusti, je nutno zajistit
	      spusteni jinymi prostredky.

	      Jinak plati vice-mene totez co pro Linux.

----------------------------------------------------------------------
   Linux
----------------------------------------------------------------------
1. RUTINNI PROVOZ

1.1. Skript: ai-start.sh - start demona v rezimu "nodebug",-> spusten
                           jako  daemon v rutinnim provozu

rezim <mode>=nodebug=rutinni provoz
nodebug - parametr, ktery prepne rezim demona do aktivniho provozu.
          V aktivnim provozu:
	                    1. data pro predict mod se nacitaji z PLC
	                    2. vysledky predikce se zapisuji do PLC
			    3. v pripade rezimu mereni artefaktu
			       se rozsituje mnozina dat urcenych
			       k uceni.
			    4. pri prechodu na novy den se startuje
			       cyklus TRAIN, po ukonceni tohoto
			       cyklu se obnovi cyklus PREDICT
			    5. Data k doucovani jsou zapisovana do
			       adresare ./br_data/tm-ai_YYYY_MM_DD.csv
			    6. Data pro trenink a validaci se ctou
			       z adresare ./br-data/tm-ai_<YYYY_MM_DD>.csv
			       Treninkova mnozina vznika jako JOIN
			       vsech suboru vyhovujicich masce
			       tm-ai_<YYYY_MM_DD>.csv

          V pripade, ze je nektery z PLC zdroju vypnut, prechazi demon
	  do rezimu sleep, v nemz je kazdych 600 [s] testovana aktivita
	  PLC. V pripade znovu spusteni PLC prechazi demon do aktivniho
	  rezimu. Sleep interval je nastaven v programu ai-daemon.py v
	  objektu NeuroDaemon, metode runDaemonLSTM a promenne
	  sleep_interval = 600[s];

	  V aktivnim rezimu se ctou data z PLC v intervalu 10 [s]. Tento
	  interval je nastaven v programu ai-daemon.py v promenne
	  self.plc_timer (objekt OPCAgent)

	  agenta ai-daemon, lze spoustet v rezimu daemon
	  (vstupni parametr=start) nebo v rezimu normalniho programu


2. TESTOVACI PROVOZ

2.1 Skript: ai-test.sh
=========================================================================
ai-test.sh - start demona v rezimu "debug",-> spusten jako program.
Rezim debug spousti ai-demona v ladicim rezimu, ktery je urcen pro
optimalizaci hyperparametru neuronove site.

rezim DEBUG_MODE=debug=ladici provoz 
	  V neaktivnim provozu (parametr "DEBUG_MODE=debug") je program spusten v
	  normalnim run modu. Tento rezim je urcen pro ladeni a optimalizaci
	  hyparametru neuronove site.
	  
	  V neaktivnim rezimu je prerusena nekonecna smycka predikcniho
	  modu. Program v tomto rezimu udela 1x TRAIN a 1x PREDICT. Vys-
	  ledky predikce zapise do ./result/<nazev-testu>. Tyto vysledky
	  je nasledne mozno prevest do grafu <nazev-testu>.pdf, s pomoci
	  programu ./py-src/ai-graf01.py a ./py-src/ai-graf02.py.

	  V neaktivnim rezimu:
	                    1. Je prerusena nekonecna smycka PREDICT
			    2. Data pro trenink a validaci se ctou
			       z adresare ./br-data/tm-ai_<YYYY_MM_DD>.csv
			       Treninkova mnozina vznika jako JOIN
			       vsech suboru vyhovujicich masce
			       tm-ai_<YYYY_MM_DD>.csv
	                    3. Data pro predict mod se nacitaji ze
			       souboru ./br-data/predict-debug.csv, ktery
			       ma stejny format jako soubory pro trenink
			       tm-ai_<YYYY_MM_DD>.csv
			       POZOR !!! obsah souboru predict-debug.csv
			       se nesmi shodovat s obsahem souboruuu
			       tm-ai_<YYYY_MM_DD>.csv.Tedy, podminkou
			       pro predict-debug.csv je vyber dat z jineho
			       casoveho intervalu, ktery neni obsazen
			       v tm-ai_<YYYY_MM_DD>.csv
			    4. Vysledky predikce jsou zapisovany do adre-
			       sare ./result/<nazev-testu>.

ai-test.sh spusti programy v testovacim (debug) modu. Parametry
programu lze zadat editaci tohoto skriptu na radku:

PRIKLAD:
#--LIST TESTU  <jmeno-testu><neuron><typ><epoch><vrstev>
./ai-part01.sh test11_yz 76 DENSE 29 1
./ai-part01.sh test12_yz 86 DENSE 39 2
./ai-part01.sh test13_yz 96 DENSE 49 3

pricemz:
<jmeno-testu> - jmeno testu pod kterym budou zaznamenany vysledky v
                adresari ./result/<jmeno-testu>
<neuron>      - pocet neuronu ve vrstve
<typ>         - typ site, - povoleno [DENSE, GRU, LSTM, CONV1D]
<epoch>       - pocet treninkovych cyklu (pocet ucebnich iteraci)
<vrstev>      - pocet skrytych vrstev neuronove site
		
Techto radku je mozno pridat vice, uloha se nasledne prokouse vsemi
zadanymi testy

2.2 Skript ./ai-part01.sh
=========================================================================
je volan skriptem ./ai-test.sh. Spousti vlastni ulohu (./ai-daemon.sh)
a provadi nektere operace nad vyslednymi daty. Napriklad generuje grafy z
vyslednych souboru.

Pro testovaci rezim jsou pripravena ladici data z obdobi 31.10.2022
az 15.11.2022. Tato data jsou rozdelena do ctyr uloh.

v1_src - treninkova data od 2022-11-01 do 2022-11.05 vcetne
       - predikcni data  od 2022-11-14 do 2022-11-15 vcetne

v2_src - treninkova data od 2022-11-01 do 2022-11-05 vcetne
                         od 2022-11-08 do 2022-11-11 vcetne
       - predikcni data  od 2022-11-14 do 2022-11-15 vcetne

v3_src - treninkova data od 2022-09-12 do 2022-09-12 vcetne
                         od 2022-11-01 do 2022-11-05 vcetne
                         od 2022-11-08 do 2022-11-11 vcetne
                         od 2022-11-14 do 2022-11-15 vcetne
       - predikcni data  od 2022-09-10 do 2022-09-10 vcetne

v4_src - treninkova data od 2022-09-10 do 2022-09-10 vcetne
                         od 2022-11-01 do 2022-11-05 vcetne
                         od 2022-11-08 do 2022-11-11 vcetne
                         od 2022-11-14 do 2022-11-15 vcetne
       - predikcni data  od 2022-09-12 do 2022-09-12 vcetne

Skript ai-part.sh probehne vsechny treniknove ulohy v1_src az v4_src
a zaznamena vysledky v adesari ./result.

3. HLAVNI SKRIPT
3.1 Skript ./ai-daemon.sh
=========================================================================
hlavni skript pro spusteni ai-daemona. Lze nastavit vsechny
potrebne hyperparametry, (viz ./ai-daemon.sh --help)

POZOR ve windowsech nelze pouzit parametr <--mode>
			       

nodebug - parametr, ktery prepne rezim demona do aktivniho provozu.
          V aktivnim provozu:
	                    1. data pro predict mod se nacitaji z PLC
			    
	                    2. vysledky predikce se zapisuji do PLC
			    
			    3. v pripade rezimu mereni artefaktu
			       se rozsituje mnozina dat urcenych
			       k uceni.
			       
			    4. pri prechodu na novy den se startuje
			       cyklus TRAIN, po ukonceni tohoto
			       cyklu se obnovi cyklus PREDICT.
			       
			    5. Data k doucovani jsou zapisovana do
			       adresare ./br_data/getplc/tm-ai_YYYY_MM_DD.csv
			       
			    6. Data pro trenink a validaci se ctou
			       z adresare ./br-data/tm-ai_<YYYY_MM_DD>.csv
			       Treninkova mnozina vznika jako JOIN
			       vsech suboru vyhovujicich masce
			       tm-ai_<YYYY_MM_DD>.csv

          V pripade, ze je nektery z PLC zdroju vypnut, prechazi demon
	  do rezimu sleep, v nemz je kazdych 600 [s] testovana aktivita
	  PLC. V pripade znovu spusteni PLC prechazi demon do aktivniho
	  rezimu. Sleep interval je nastaven v programu ai-daemon.py v
	  objektu NeuroDaemon, metode runDaemonLSTM a promenne
	  sleep_interval = 600[s];

	  V aktivnim rezimu se ctou data z PLC v intervalu 10 [s]. Tento
	  interval je nastaven v programu ai-daemon.py v promenne
	  self.plc_timer (objekt OPCAgent)

	  agenta ai-daemon, lze spoustet v rezimu daemon
	  (vstupni parametr=start) nebo v rezimu normalniho programu

debug   - parametr, ktery prepne rezim demona do aktivniho provozu.
	  V neaktivnim rezimu:
	  (parametr "debug") je program spusten v normalnim run modu.
	  Tento rezim je urcen pro ladeni a optimalizaci
	  hyparametru neuronove site.
	  
	  V neaktivnim rezimu je prerusena nekonecna smycka predikcniho
	  modu. Program v tomto rezimu udela 1x TRAIN a 1x PREDICT. Vys-
	  ledky predikce zapise do ./result/<nazev-testu>. Tyto vysledky
	  je nasledne mozno prevest do grafu <nazev-testu>.pdf, s pomoci
	  programu ./py-src/ai-graf01.py a ./py-src/ai-graf02.py.

	  V neaktivnim rezimu:
	                    1. Je prerusena nekonecna smycka PREDICT
			    
			    2. Data pro trenink a validaci se ctou
			       z adresare ./br-data/tm-ai_<YYYY_MM_DD>.csv
			       Treninkova mnozina vznika jako JOIN
			       vsech suboru vyhovujicich masce
			       tm-ai_<YYYY_MM_DD>.csv
			       
	                    3. Data pro predict mod se nacitaji ze
			       souboru ./br-data/predict-debug.csv, ktery
			       ma stejny format jako soubory pro trenink
			       tm-ai_<YYYY_MM_DD>.csv
			       POZOR !!! obsah souboru predict-debug.csv
			       se nesmi shodovat s obsahem souboruuu
			       tm-ai_<YYYY_MM_DD>.csv.Tedy, podminkou
			       pro predict-debug.csv je vyber dat z jineho
			       casoveho intervalu, ktery neni obsazen
			       v tm-ai_<YYYY_MM_DD>.csv
			       
			    4. Vysledky predikce jsou zapisovany do adre-
			       sare ./result/<nazev-testu>.

popis parametru skriptu ai-demon.sh 
=========================================================================

pouziti: <nazev_programu> <arg-1> <arg-2> <arg-3>,..., <arg-n>
 
        --help            list help ")
 
 
        --model           model neuronove site 'DENSE', 'LSTM', 'GRU', 'BIDI'")
                                 DENSE - zakladni model site - nejmene narocny na system")
                                 LSTM - Narocny model rekurentni site s feedback vazbami")
 
        --epochs          pocet ucebnich epoch - cislo v intervalu <1,256>")
                                 pocet epoch urcuje miru uceni. POZOR!!! i zde plati vseho s mirou")
                                 Pri malych cislech se muze stat, ze sit bude nedoucena ")
                                 a pri velkych cislech preucena - coz je totez jako nedoucena.")
                                 Jedna se tedy o podstatny parametr v procesu uceni site.")
 
        --units           pocet vypocetnich jednotek <32,1024>")
                                 Pocet vypocetnich jednotek urcuje pocet neuronu zapojenych do vypoctu.")
                                 Mějte prosím na paměti, že velikost units ovlivňuje 
                                 dobu tréninku, chybu, které dosáhnete, posuny gradientu atd. 
                                 Neexistuje obecné pravidlo, jak urcit optimalni velikost parametru units.
                                 Obecne plati, ze maly pocet neuronu vede k nepresnym vysledkum a naopak
                                 velky pocet units muze zpusobit preuceni site - tedy stejny efekt jako pri
                                 nedostatecnem poctu units. Pamatujte, ze pocet units vyrazne ovlivnuje alokaci
                                 pameti. pro 1024 units je treba minimalne 32GiB u siti typu LSTM, GRU nebo BIDI.
                                 Plati umera: cim vetsi units tim vetsi naroky na pamet.
                                              cim vetsi units tim pomalejsi zpracovani.
 
        --layers          pocet vrstev v hidden <0,12>")
                                 Pocet vrstev v hidden casti - pozor!!! vice vrstev muze")
                                 mit vyrazne negativni vliv na uceni site 
 
 
        --actf            Aktivacni funkce - jen pro parametr DENSE")
                                 U LSTM, GRU a BIDI se neuplatnuje.")
                                 Pokud actf neni uvedan, je implicitne nastaven na 'tanh'. 
                                 U site GRU, LSTM a BIDI je implicitne nastavena na 'tanh' 
 
 
        --txdat1          timestamp zacatku datove mnoziny pro predict, napr '2022-04-09 08:00:00' ")
 
        --txdat2          timestamp konce   datove mnoziny pro predict, napr '2022-04-09 12:00:00' ")
 
                                 parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere
                                 se v uvahu cela mnozina dat k trenovani, to znamena:
                                 od pocatku mereni: 2022-02-15 00:00:00 
                                 do konce   mereni: current timestamp() - 1 [den]
        --ilcnt           vynechavani vet v treninkove mnozine - test s jakum poctem vet lze
                                 jeste solidne predikovat
				 1 = vsechny vety
                                 2 = nactena kazda druha
                                 3 = nactena kazda treti...
				 .
				 .
				 .
        --shuffle         nahodne promichani dat 
                                 shuffle=TRUE - implicitni hodnota - treninkova data
                                                se promichaji                       
                                 shuffle=FALSE- treninkova data se nepromichaji
				 parametr shuffle=TRUE pozitivne ovlivnuje presnost
				 predikce.
				 
 
POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM, GRU nebo BIDI!!!
       pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise 
       natrenovany model site..
       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.
       spusti normalni beh site z jiz natrenovaneho modelu.
       Takze: budte trpelivi...
 
Pokud pozadujete zmenu parametu je mozno primo v programu poeditovat tyto promenne:
  df_parmx a df_parmX. Pozor!!! df_parmX musi obsahovat totoznou mnozinu parametru
  pro predikci.
 
a nebo vyrobit soubor ./cfg/ai-parms.cfg s touto syntaxi: (lepe nezli editovat program)

#Vystupni list parametru - co budeme chtit po siti predikovat
#--------------------------------------------------------------------------------------------
#Tenzor predlozeny k predikci
#--------------------------------------------------------------------------------------------
df_parmx = temp_vr02, temp_vr05, temp_vr06, temp_vr07, temp_st02, temp_st05, temp_st07, temp_S1

#--------------------------------------------------------------------------------------------
#Tenzor predlozeny k treninku
#--------------------------------------------------------------------------------------------
df_parmX = dev_x4, dev_y4, dev_z4, dev_x5, dev_y5, dev_z5,
           temp_vr02, temp_vr05, temp_vr06, temp_vr07, temp_st02, temp_st05, temp_st07, temp_S1
  
POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy
vstupniho datoveho CSV souboru (nebo souboruuu) a muzou tam byt i uvozovky:
priklad: 'machinedata_m0112','machinedata_m0212'


(C) GNU General Public License, autor Petr Lukasik , 2022 
 
Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)
              miniconda3,
              python 3.9, tensorflow 2.8, mathplotlib,  
              tensorflow 2.8,
              mathplotlib,  
              scikit-learn-intelex,  
              pandas,  
              numpy,  
              keras   
 
 
Povolene aktivacni funkce: 
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
		   
Podporovane typy siti
    models_1    = ["DENSE","LSTM","GRU","CONV1D"];
    models_2    = ["DENSE","LSTM","GRU","CONV1D",""];


CONDA
-----------------------------------------------------------------------
prostredi pro spusteni skriptu - tf.yaml

conda env create -f tf.yaml



