/*
 *  layer.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: fall 21 (curs 21-22)
 *  Last modified: fall 24 (curs 24-25)
 *  Author: Blanca Llauradó,
 *  Modified: Christian Germer
 *
 *  Descripció:
 *  Constants, estructures de dades i capçaleres de les funcions auxiliars per la creació de cada capa (layer) de la xarxa neuronal.
 *
 */

#ifndef LAYER_H
#define LAYER_H

typedef struct layer_t {
    int num_neu;
    float* actv;
    float* bias;
    float* z;
    float* dactv;
    float* dbias;
    float* dz;
    float* out_weights;
    float* dw;
} layer;

extern int debug;

layer create_layer(int num_neurons, int number_of_neurons_next_layer);
void free_layer(layer lay);

#endif
