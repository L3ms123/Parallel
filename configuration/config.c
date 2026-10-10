/*
 *  config.c
 *
 *  Created on: March 23rd 2022
 *  Author: ecesar asikora
 *  Last modified: fall 24 (curs 24-25)
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Description:
 *  Functions for reading the program configuration file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXIT_FAILURE 1

int num_layers;
int* num_neurons;
float alpha;
int batch_size;
int num_epochs;
int num_training_patterns;
int num_test_patterns;
int num_out_layer;
char dataset_training_path[256];
char dataset_test_path[256];
int img_dim_x, img_dim_y;
int seed;
int debug;

float** desired_outputs;
int* Validation;

void checkError(int ok, char* msg, char* file) {
    if (!ok) {
        if (file == NULL)
            fprintf(stderr, "-- Error: %s\n", msg);
        else
            fprintf(stderr, "-- Error: %s File: %s\n", msg, file);
        exit(EXIT_FAILURE);
    }
}

void printConfiguration() {
    printf("num_layers=%d\n", num_layers);
    for (int i = 0; i < num_layers; i++)
        printf("layer=%d\n", num_neurons[i]);

    printf("alpha=%f\n", alpha);
    printf("batch_size=%d\n", batch_size);
    printf("num_epochs=%d\n", num_epochs);
    printf("num_training_patterns=%d\n", num_training_patterns);
    printf("num_test_patterns=%d\n", num_test_patterns);
    printf("num_out_layer=%d\n", num_out_layer);
    for (int i = 1; i < num_layers - 1; i++)
        printf("num_hidden_layer_%d=%d\n", i, num_neurons[i]);
    printf("img_dim_x=%d img_dim_y=%d\n", img_dim_x, img_dim_y);
    printf("dataset_training_path=%s\n", dataset_training_path);
    printf("dataset_test_path=%s\n", dataset_test_path);
    printf("seed=%d\n", seed);
    printf("debug=%d\n", debug);
}

void readConfiguration(char* configfile) {
    int ok;

    FILE* file = fopen(configfile, "r");
    checkError(file != NULL, "file not found", configfile);

    ok = fscanf(file, "num_layers=%d\n", &num_layers);
    checkError(ok != EOF, "reading num_layers", configfile);

    num_neurons = malloc(num_layers * sizeof(int));
    checkError(num_neurons != NULL, "allocating neurons_by_layer\n", NULL);

    for (int i = 0; i < num_layers; i++) {
        ok = fscanf(file, "layer=%d\n", &num_neurons[i]);
        checkError(ok != EOF, "reading num_layers", configfile);
    }

    num_out_layer = num_neurons[num_layers - 1];

    ok = fscanf(file, "num_training_patterns=%d\n", &num_training_patterns);
    checkError(ok != EOF, "reading num_training_ex", configfile);

    ok = fscanf(file, "num_test_patterns=%d\n", &num_test_patterns);
    checkError(ok != EOF, "reading num_rec_patterns", configfile);

    desired_outputs = (float**)malloc(num_training_patterns * sizeof(float*));
    for (int i = 0; i < num_training_patterns; i++)
        desired_outputs[i] = malloc(num_out_layer * sizeof(float));

    Validation = malloc(num_test_patterns * sizeof(int));

    ok = fscanf(file, "img_dim_x=%d\n", &img_dim_x);
    checkError(ok != EOF, "reading image dimensions", configfile);

    ok = fscanf(file, "img_dim_y=%d\n", &img_dim_y);
    checkError(ok != EOF, "reading image dimensions", configfile);

    ok = fscanf(file, "dataset_training_path=%s\n", dataset_training_path);
    checkError(ok != EOF, "reading dataset_path", configfile);

    ok = fscanf(file, "dataset_test_path=%s\n", dataset_test_path);
    checkError(ok != EOF, "reading dataset_test_path", configfile);

    ok = fscanf(file, "num_epochs=%d\n", &num_epochs);
    checkError(ok != EOF, "reading num_epochs\n", configfile);

    ok = fscanf(file, "seed=%d\n", &seed);
    checkError(ok != EOF, "reading seed", configfile);

    ok = fscanf(file, "alpha=%f\n", &alpha);
    checkError(ok != EOF, "reading alpha", configfile);

    ok = fscanf(file, "batch_size=%d\n", &batch_size);
    checkError(ok != EOF, "reading batch_size", configfile);

    ok = fscanf(file, "debug=%d\n", &debug);
    checkError(ok != EOF, "reading debug", configfile);

    if (debug == 1)
        printConfiguration();
}