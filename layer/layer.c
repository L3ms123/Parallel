/*
 *  layer.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: fall 21 (curs 21-22)
 *  Last modified: fall 24 (curs 24-25)
 *  Author: Blanca Llauradó,
 *  Modified: Christian Germer
 * 			  Eduardo i Ania (Afegir funció per alliberar la memòria ocupada per la capa. 28-01-25)
 *
 *  Descripció:
 *  Implementació de les funcions auxiliars per la creació de cada capa (layer) de la xarxa neuronal.
 *
 */

#include "layer.h"

#include <stdio.h>
#include <stdlib.h>

layer create_layer(int number_of_neurons, int number_of_neurons_next_layer) {
    layer lay;
    lay.num_neu = number_of_neurons;

    lay.actv = malloc(number_of_neurons * sizeof(float));
    lay.bias = malloc(number_of_neurons * sizeof(float));
    lay.z = malloc(number_of_neurons * sizeof(float));
    lay.dactv = malloc(number_of_neurons * sizeof(float));
    lay.dbias = malloc(number_of_neurons * sizeof(float));
    lay.dz = malloc(number_of_neurons * sizeof(float));

    lay.out_weights = malloc(number_of_neurons * number_of_neurons_next_layer *
                             sizeof(float));
    lay.dw = malloc(number_of_neurons * number_of_neurons_next_layer *
                    sizeof(float));

    return lay;
}

void free_layer(layer lay) {
    free(lay.actv);
    free(lay.bias);
    free(lay.z);
    free(lay.dactv);
    free(lay.dbias);
    free(lay.dz);
    free(lay.out_weights);
    free(lay.dw);
}
