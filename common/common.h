/*
 *  common-2018.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 13 gen. 2018
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar
 *  Modified: Blanca Llauradó, Christian Germer
*/

#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <sys/time.h>

extern int batch_size;  // Patrons per batch

extern float** desired_outputs;
extern int* Validation;

/*Resta de variables globals que s'utilitzen en aquest
 *arxiu
 */
extern int num_training_patterns;
extern int num_test_patterns;
extern int img_dim_x;
extern int img_dim_y;
extern int num_out_layer;
extern int debug;

/* Capçaleres de les funcions implementades a common.c */
char** loadPatternSet(int nf, char* fname, int trainS);

char* readImg(FILE* fd);
void printImg(char* Img, int x);
double elapsedT(struct timeval begin);

#endif
