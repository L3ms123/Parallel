/*
 *  main.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Capçaleres de la funció que entrena la xarxa neuronal definida + funció que
 * fa el test del model entrenat. Declaració de variables globals.
 *
 */

#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>

#include "common/common.h"
#include "configuration/config.h"
#include "initialize/initialize.h"
#include "layer/layer.h"
#include "randomizer/randomizer.h"
#include "training/training.h"

layer* lay = NULL;
char** input;
int total = 0;

extern int num_layers;
extern int* num_neurons;
extern int batch_size;
extern int num_epochs;
extern float alpha;
extern int num_training_patterns;
extern int seed;
extern char dataset_test_path[256];
extern int num_test_patterns;
extern int num_out_layer;
extern int* Validation;
extern int debug;

void printRecognized(int p, layer Output);
void train_neural_net();
void test_nn();

#endif