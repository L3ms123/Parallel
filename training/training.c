/*
 *  training.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 31 gen. 2019
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions per entrenar la xarxa neuronal.
 *
 * TODO: Comentar bé aquestes funcions perquè són les que han de paral·lelitzar!!
 */

#include "training.h"

#include <math.h>
#include <stdio.h>


/**
 * @brief Inicialitza la capa incial de la xarxa (input layer) amb l'entrada
 *que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];
}

/**
 * @brief Inicialitza les matrius de variacions de pesos (dw) i els vectors de
 * variació de bias (dbias) de totes les capes.
 *
 * @param Això cal fer-ho per cada batch, així es poden acumular les variacions
 * per tots els patrons del batch i aplicar-les a l'acabar de processar tot el 
 * batch.
 **/
void init_deltas() {
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < num_neurons[i]; j++) 
            lay[i].dbias[j] = 0.0;
        for (int j = 0; j < num_neurons[i + 1] * num_neurons[i]; j++) 
            lay[i].dw[j] = 0.0;
    }
}

/**
 * @brief Propagació dels valors de les neurones de l'entrada (valors a la input
 * layer) a la resta de capes de la xarxa fins a obtenir una predicció (sortida)
 *
 * @details La capa d'entrada (input layer = capa 0) ja ha estat inicialitzada
 * amb els valors de l'entrada que volem reconeixer. Així, el for més extern
 * (sobre i) recorre totes les capes de la xarxa a partir de la primera capa
 * hidden (capa 1). El for intern (sobre j) recorre les neurones de la capa i
 * calculant el seu valor d'activació [lay[i].actv[j]]. El valor d'activació de
 * cada neurona depén de l'exitació de la neurona calculada en el for més intern
 * (sobre k) [lay[i].z[j]]. El valor d'exitació s'inicialitza amb el biax de la
 * neurona corresponent [j] (lay[i].bias[j]) i es calcula multiplicant el valor
 *          d'activació de les neurones de la capa anterior (i-1) pels pesos de
 * les connexions (out_weights) entre les dues capes. Finalment, el valor
 * d'activació de la neurona (j) es calcula fent servir la funció RELU
 * (REctified Linear Unit) si la capa (j) és una capa oculta (hidden) o la
 * funció Sigmoid si es tracte de la capa de sortida.
 *
 */
void forward_prop() {
    for (int i = 1; i < num_layers; i++) {
        if (debug == 1) printf("Valors activació capa %d \n", i);
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].z[j] = lay[i].bias[j];
            for (int k = 0; k < num_neurons[i - 1]; k++)
                lay[i].z[j] +=
                    ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) *
                     (lay[i - 1].actv[k]));
            if (i < num_layers - 1)  // Relu Activation Function for Hidden Layers
                lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
            else  // Sigmoid Activation Function for Output Layer
                lay[i].actv[j] = 1 / (1 + exp(-lay[i].z[j]));
            if (debug == 1) printf("%f %f ", lay[i].actv[j], lay[i].z[j]);
        }
        if (debug == 1) printf("\n\n");
    }
}

/** 
 * @brief Calcula la funció de perdua per cada patró i l'acumulat fins el punt actual 
 * d'entrenament
 * 
 * @details La funció de pèrdua (loss function) en l'entrenament 
 * d'una xarxa neuronal multivivell és essencialment una mesura de com de malament li 
 * va a la xarxa per realitzar la tasca per a la qual està sent entrenada, donades les 
 * seves sortides actuals en comparació amb les sortides desitjades (ground truth).
 * Fem servir la funció de loss Entropia Creuada Binària (Binary Cross-Entropy o Log Loss)
 * que s'utilitza per a problemes de classificació quan la capa de sortida fa servir una 
 * funció d'activació sigmoide.
*/
float compute_loss(int i) {
    double tmpcost = 0, p;

    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        p = lay[num_layers - 1].actv[j];
        if (lay[num_layers - 1].actv[j] < EPSILON) p = EPSILON;
        else if(lay[num_layers - 1].actv[j] > 1 - EPSILON) p = 1 - EPSILON;

        tmpcost += desired_outputs[i][j] * log(p) + (1 - desired_outputs[i][j]) * log(1.0 - p);
    }
    return -tmpcost/num_neurons[num_layers - 1];
}

/**
 * @brief Calcula el gradient que es necessari aplicar als pesos de les
 * connexions entre neurones per corregir els errors de predicció
 *
 * @details Calcula dos vectors de correcció per cada capa de la xarxa, un per
 * corregir els pesos de les connexions de la neurona (j) amb la capa anterior
 *          (lay[i-1].dw[j]) i un segon per corregir el biax de cada neurona de
 * la capa actual (lay[i].bias[j]). Hi ha un tractament diferent per la capa de
 * sortida (num_layesr -1) perquè aquest és l'única cas en el que l'error es
 * conegut (lay[num_layers-1].actv[j] - desired_outputs[p][j]). Això es pot
 * veure en els dos primers fors. Per totes les capes ocultes (hidden layers) no
 * es pot saber el valor d'activació esperat per a cada neurona i per tant es fa
 * una estimació. Aquest càlcul es fa en el doble for que recorre totes les
 * capes ocultes (sobre i) neurona a neurona (sobre j). Es pot veure com en cada
 * cas es fa una estimació de quines hauríen de ser les activacions de les
 * neurones de la capa anterior (lay[i-1].dactv[k] = lay[i-1].out_weights[j*
 * num_neurons[i-1] + k] * lay[i].dz[j];), excepte pel cas de la capa d'entrada
 * (input layer) que és coneguda (imatge d'entrada).
 *
 */
void back_prop(int p) {
    // Output Layer
    for (int k = 0; k < num_neurons[num_layers - 2]; k++) 
        lay[num_layers - 2].dactv[k] = 0;

    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        lay[num_layers - 1].dz[j] =
            (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
            (lay[num_layers - 1].actv[j]) * (1 - lay[num_layers - 1].actv[j]);
        lay[num_layers - 1].dbias[j] += lay[num_layers - 1].dz[j];
    }

    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] +=
                (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
            lay[num_layers - 2].dactv[k] +=          
                lay[num_layers - 2].out_weights[j * num_neurons[num_layers - 2] + k] *
                lay[num_layers - 1].dz[j];
        }
    }

    // Hidden Layers
    for (int i = num_layers - 2; i > 0; i--) {
        if ( i > 1 )
            for (int k = 0; k < num_neurons[i - 1]; k++)
                lay[i - 1].dactv[k] = 0;

        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0;

            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i - 1].dw[j * num_neurons[i - 1] + k] +=
                    lay[i].dz[j] * lay[i - 1].actv[k];

                if (i > 1)
                    lay[i - 1].dactv[k] +=
                        lay[i - 1].out_weights[j * num_neurons[i - 1] + k] *
                        lay[i].dz[j];
            }
            lay[i].dbias[j] += lay[i].dz[j];
        }
    }
}

/**
 * @brief Actualitza els vectors de pesos (out_weights) i de biax (bias) de cada
 * etapa d'acord amb els càlculs fet a la funció de back_prop i el factor
 * d'aprenentatge alpha
 *
 * @see back_prop
 */
void update_weights(void) {
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < num_neurons[i + 1]; j++)
            for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                lay[i].out_weights[j * num_neurons[i] + k] =
                    (lay[i].out_weights[j * num_neurons[i] + k]) -
                    (alpha * lay[i].dw[j * num_neurons[i] + k]);

        for (int j = 0; j < num_neurons[i]; j++)  // Update Bias
            lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);
    }
}
