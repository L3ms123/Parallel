
README

Aquest arxiu conté informació sobre:
  -Com configurar les variables globals
  -La estructura d'arxius
  -El funcionament del programa
  -Més coses


---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------
CONFIGURACIO:

configuration/configfile.txt es l'arxiu que ens permet configurar totes les variables
globals del programa per no tenir variables "hard coded" i poder configurar tot en un
sol arxiu per poder mantenir coherencia

Per afegir noves variables globals que volguem configurar desde l'arxiu de configuraci� cal
seguir la semàntica utilitzada en l'arxiu; Una variable per linea amb el forma var_name=x
on var_name -> nom de la variable i x -> valor que volem asignar-li

A continuació a la funció void readConfiguration(char *configfile) en configuration/config.c
hem d'afegir la lectura de la variable. 
MOLT IMPORTANT: La lectura de variables ha de ser en el mateix ordre que en l'arxiu de configuració
Per llegir una variable fem servir fscanf() de la següent forma:

ok = fscanf(file, "alpha=%f\n", &alpha); -> alpha= ens indica en el ficher quina variable llegim 
                                         -> %f indica el valor a llegir i que en aquest cas es un float (%d per ints, %s per strings, etc)
                                         -> \n s'ha de posar al final per saltar a la següent linea per llegir-la més endavant

per poder fer servir una variable global en altres arxius cal fer servir la directiva "extern"
com es pot veure varies ocasions ja al codi
extern permet "indicar" que una variable global ja esta declarada en un altre arxiu


LLISTA ACTUAL DE VARIABLES:

num_layers=5 // Number of layers and memory allocation
layer=1024 // Number of neurons layer0 (in layer or img_size (dim_x * dim_y))
layer=500 // Number neurons layer1 (hidden layer)
layer=210 // Number neurons layer2 (hidden layer)
layer=135 // Number neurons layer3 (hidden layer)
layer=10 // Number neurons layer4 (out layer)
num_training_patterns=1934 // Number of training examples 
num_test_patterns=946 // Number of patterns to recognize 
img_dim_x=32 // images size x
img_dim_y=32 //images size y
dataset_path=./datasets/optdigits/optdigits.tra // Training set path -> all training is in datasets/
dataset_test_path=./datasets/optdigits/optdigits.cv // Dataset path for test after training
num_epochs=10   // number of training iterations
seed=50     // randomizer seed
alpha=0.015 // Learning rate (indicates what portion of gradient should be used)
batch_size=50 // Number of patrons per batch 
debug=0 // debug level (0: no debug (shows loss for each epoch and the final number of hits), 
                        1: shows the activation values for each layer and each pattern, 
                        2: shows the output for each test pattern)




---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

DATASETS:

Datasets description:

1) hands:
Subset of the fingers dataset, composed by images of open and closed left hands.
    3000 128x128 train patters
     600 128x128 test images
    
Dataset values for configfile.txt:
    
num_layers=3
layer=16384
layer=135
layer=2
num_training_patterns=3000
num_test_patterns=600
img_dim_x=128
img_dim_y=128
dataset_training_path=datasets/hands/hands.tra
dataset_test_path=datasets/hands/hands.cv
    
2) fingers:
Set composed of images of left and right hands showing form 0 to 5 fingers. 
Obtained from https://www.kaggle.com/koryakinp/fingers and processed using 
https://imagemagick.org/script/index.php in order to produce two value pixels
('0' background, '1' hand)
    18000 128x128 train images
     3600 128x128 test images
    
Dataset values for configfile.txt:
    
num_layers=3
layer=16384
layer=135
layer=12
num_training_patterns=18000
num_test_patterns=3600
img_dim_x=128
img_dim_y=128
dataset_training_path=datasets/fingers/fingers.tra
dataset_test_path=datasets/fingers/fingers.cv
    
3) optdigits:
Set composed of images of handwritten digits (0-9).
Can be obtained at https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits and
it's already in the textual format indicated before ('0' background, '1' number)
    1934 32x32 train images
     946 32x32 test images
    
Dataset values for configfile.txt:
    
num_layers=5
layer=1024
layer=500
layer=210
layer=135
layer=10
num_training_patterns=1934
num_test_patterns=946
img_dim_x=32
img_dim_y=32
dataset_training_path=datasets/optdigits.tra
dataset_test_path=datasets/optdigits.cv
    
3) mnist:
Set composet of images of handwritten digits (0-9).
It looks to be obtained from the original Yann Lecun MNIST dataset 
https://www.kaggle.com/datasets/hojjatk/mnist-dataset and
it's already in the textual format indicated before ('0' background, '1' number)
    55000 32x32 train images
    5000 32x32 test images
    
Dataset values for configfile.txt:
    
num_layers=5
layer=1024
layer=500
layer=210
layer=135
layer=10
num_training_patterns=55000
num_test_patterns=5000
img_dim_x=32
img_dim_y=32
dataset_training_path=datasets/mnist_format.tra
dataset_test_path=datasets/mnist_format.cv
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

DEBUGGING:

Com que estem treballant amb C si volem fer debugging per veure on falla el nostre codi pot ser molt complicat, sobretot
perque les eines que existeixen com valgrind i altres no son molt trivials d'utilitzar

Per aixó hem fet un mètode senzill de debugging basat en una variable en l'arxiu configfile.txt

debug == 0 -> No esta activat el mode debug
debug == 1 -> Es mostra el valor d'activació de les neurones de cada capa per cada patró
debug == 2 -> Es mostra per cada patró del conjunt de prova quina és la sortida (valor d'activació)
              de cada neurona de la capa de sortida

Quan el mode debug esta activat simplement es fan printf() abans de certes crides a funcions importants o
funcions delicades on solen tenir lloc error de Segmentation fault.

Es poden afegir més punts de debug obviament. Simplement cal afegir un if() amb l'opció de debug

if(debug == 1)          // o 2
  printf("Print Something");
  
Es recomana seguir aquesta sintaxis per mantenir el codi consistent i poder distingir per a que es cada printf()
També permet desactivar-los en qualsevol moment canviant el valor de debug!!!

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------
MAKEFILE:

Cal gcc (al cluster: module add gcc/13.2.1). Per perfilar, gprof i/o perf.

Configs:
  configuration/configfile.txt       -> 10 epochs (temps, accuracy, speedup)
  configuration/config_short.txt     -> 2 epochs, mateixa xarxa i datasets (apartat 2.3)

Targets:
  make / make seq     -> exec        sequential baseline (-Ofast -g, sense -fopenmp)
  make omp            -> exec_omp    OpenMP (-Ofast -g -fopenmp)
  make gprof          -> exec_gprof  sequential amb -O2 -pg -g
  make prof-gprof     -> gprof_report.txt   executa exec_gprof amb config_short + gprof
  make prof-perf      -> perf.data, perf_report.txt   perf record -g sobre exec + config_short
  make short          -> executa exec amb config_short.txt
  make clean          -> esborra binaris, gmon.out, perf.data i reports

Perfil seqüencial (apartat 2.3):
  make prof-gprof
  make prof-perf

OpenMP vs seqüencial (10 epochs):
  make seq
  ./exec configuration/configfile.txt

  make omp
  OMP_NUM_THREADS=1 ./exec_omp configuration/configfile.txt
  OMP_NUM_THREADS=2 ./exec_omp configuration/configfile.txt
  OMP_NUM_THREADS=4 ./exec_omp configuration/configfile.txt
  OMP_NUM_THREADS=8 ./exec_omp configuration/configfile.txt

OMP_NUM_THREADS=1 ha de donar el mateix resultat que exec (seed=50).
exec es el baseline seqüencial; exec_omp es la versió paral·lela. No barrejar-los en les mesures.

Flags:
  -Ofast      timing / seq, omp, perf
  -O2 -pg     nomes gprof (no usar per speedup)
  -g          simbols per perf / debug
  -fopenmp    nomes exec_omp
  -lm         llibreria matematica

---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------