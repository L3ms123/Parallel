/**
 *  main.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 * l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 * gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funció que entrena la xarxa neuronal definida + Funció que fa el test del
 * model entrenat + programa principal.
 *
 *  @todo: modificar funció d'entrenament per utilitzar epochs i batchs!!!
 *        Epochs, fet (23/01/25)
 *        Batchs implica canviar les funcions de back_propagation i
 * update_weights de training.c, de moment ho deixem com està Revisar
 * print_recognized
 */

#include "main.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//-----------FREE INPUT------------
void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized(int p, layer Output) {
    int imax = 0;

    for (int i = 1; i < num_out_layer; i++)
        if (Output.actv[i] > Output.actv[imax])
            imax = i;

    if (imax == Validation[p])
        total++;

    if (debug == 2) {
        printf("Patró %d sembla %c i és %d", p, '0' + imax,
               Validation[p]);
        for (int k = 0; k < num_out_layer; k++)
            printf("   %f", Output.actv[k]);
        printf("\n");
    }
}

/**
 * @brief Entrena la xarxa neuronal en base al conjunt d'entrenament
 *
 * @details Primer carrega tots els patrons d'entrenament (loadPatternSet)
 *          Després realitza num_epochs iteracions d'entrenament.
 *          Cada epoch fa:
 *              - Determina aleatòriament l'ordre en que es consideraran els
 * patrons (per evitar overfitting)
 *              - Per cada patró d'entrenament fa el forward_prop (reconeixament
 * del patró pel model actual) i el back_prop i update_weights (ajustament de
 * pesos i biaxos per provar de millorar la precisió del model)
 *
 * @see loadPatternSet, feed_input, forward_prop, back_prop, update_weights,
 * freeInput
 *
 */
void train_neural_net() {
    printf("\nTraining...\n");

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    int ranpat[num_training_patterns];

    for (int p = 0; p < num_training_patterns; p++)
        ranpat[p] = p;

    // Gradient Descent
    for (int it = 0; it < num_epochs; it++) {
        float loss = 0;
        // Train patterns randomly
        for (int p = 0; p < num_training_patterns; p++) {
            int x = rando();
            int np = (x * x) % num_training_patterns;
            int op = ranpat[p];
            ranpat[p] = ranpat[np];
            ranpat[np] = op;
        }

        for (int nb = 0; nb < num_training_patterns/batch_size; nb++) {
            init_deltas();
            for (int i = nb*batch_size; i < (nb+1)*batch_size; i++) {
                int p = ranpat[i];

                feed_input(p);
                forward_prop();
                loss += compute_loss(p);
                back_prop(p);
            }
            update_weights();
        }

        printf("Epoch %d Loss: %f\n", it, loss/num_training_patterns);
        if (debug == 1) getchar();
    }
    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    char** rSet;

    printf("\nTesting...\n");

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        exit(-1);
    }

    for (int i = 0; i < num_test_patterns; i++) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];

        forward_prop();
        printRecognized(i, lay[num_layers - 1]);
    }

    printf("\nTotal encerts = %d (/%d) \n", total, num_test_patterns);
    freeInput(num_test_patterns, rSet);
}

//-----------MAIN-----------//
int main(int argc, char** argv) {
    if (debug == 1)
        printf("argc = %d \n", argc);
    if (argc <= 1)
        readConfiguration("configuration/configfile.txt");
    else
        readConfiguration(argv[1]);

    if (debug == 1)
        printf("FINISH CONFIG \n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    // Start measuring time
    struct timeval begin;
    gettimeofday(&begin, 0);

    // Train
    train_neural_net();

    // Test
    test_nn();

    // Stop measuring time and calculate the elapsed time
    double elapsed = elapsedT(begin);

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    printf("\n\nGoodbye! (%f sec)\n\n", elapsed);

    return 0;
}