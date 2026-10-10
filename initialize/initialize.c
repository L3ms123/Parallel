/*
 *  initialize.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: fall 21 (curs 21-22)
 *  Last modified: fall 24 (curs 24-25)
 *  Author: Blanca Llauradó,
 *  Modified: Christian Germer, Eduardo i Ania (programació de dinit 30-01-25)
 *
 *  Descripció:
 *  Funcions auxiliars per la creació de l'arquitectura de la xarxa neuronal d'acord amb els paràmetres
 *  indicats en l'arxiu de configuració.
 *
 *
 */

#include "initialize.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------INIT------------
int init() {
    if (create_architecture() != SUCCESS_CREATE_ARCHITECTURE) {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    return SUCCESS_INIT;
}

//-----------DINIT------------
int dinit() {
    for (int i = 0; i < num_layers; i++)
        free_layer(lay[i]);
    free(lay);
    for (int i = 0; i < num_training_patterns; i++)
        free(desired_outputs[i]);
    free(desired_outputs);
    free(Validation);
    free(num_neurons);
    return SUCCESS_DINIT;
}

//-----------CREATE NEURAL NETWORK ARCHITECTURE------------
int create_architecture() {
    lay = (layer *)malloc(num_layers * sizeof(layer));

    for (int i = 0; i < num_layers; i++)
        lay[i] = create_layer(num_neurons[i],
                              (i < (num_layers - 1)) ? num_neurons[i + 1] : 0);

    // Initialize the weights
    if (initialize_weights() != SUCCESS_INIT_WEIGHTS) {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

//-----------INITIALIZE WEIGHTS------------
int initialize_weights() {
    if (lay == NULL) {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < num_neurons[i]; j++) {
            for (int k = 0; k < num_neurons[i + 1]; k++) {
                // Initialize Output Weights for each neuron
                lay[i].out_weights[k * num_neurons[i] + j] =
                    random_between_two(-sqrt((float)2 / (float)num_neurons[i]),
                                       sqrt((float)2 / (float)num_neurons[i]));
                lay[i].dw[k * num_neurons[i] + j] = 0.0;
            }

            if (i > 0)
                lay[i].bias[j] = random_between_two(
                    -sqrt((float)2 / (float)num_neurons[i - 1]),
                    sqrt((float)2 / (float)num_neurons[i - 1]));
        }
    }

    for (int j = 0; j < num_neurons[num_layers - 1]; j++)
        lay[num_layers - 1].bias[j] = random_between_two(
            -sqrt((float)2 / (float)num_neurons[num_layers - 2]),
            sqrt((float)2 / (float)num_neurons[num_layers - 2]));

    return SUCCESS_INIT_WEIGHTS;
}