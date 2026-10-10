
/*
 *  initialize.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: fall 21 (curs 21-22)
 *  Last modified: fall 24 (curs 24-25)
 *  Author: Blanca Llauradó,
 *  Modified: Christian Germer
 *
 *  Descripció:
 *  Constants i capçaleres de les funcions auxiliars per la creació de l'arquitectura de la xarxa neuronal d'acord amb els paràmetres
 *  indicats en l'arxiu de configuració.
 *
 */

#ifndef INITIALIZE_H
#define INITIALIZE_H

#define SUCCESS_INIT 0
#define ERR_INIT 1

#define SUCCESS_DINIT 0
#define ERR_DINIT 1

#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1

#define SUCCESS_UPDATE_WEIGHTS 0

#define SUCCESS_CREATE_ARCHITECTURE 0
#define ERR_CREATE_ARCHITECTURE 1

#include "../common/common.h"
#include "../layer/layer.h"
#include "../randomizer/randomizer.h"

extern char dataset_training_path[256];
extern int num_layers;
extern int* num_neurons;
extern layer* lay;
extern char** input;
extern int debug;

int init();
int dinit();
int create_architecture();
int initialize_weights();
void get_data();

#endif