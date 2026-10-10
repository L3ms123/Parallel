/*
 *  common-2018.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 31 gen. 2019
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions auxiliars per la lectura d'un arxiu de patrons format per un
 *  nombre indeterminat de xifres (0-9) escrites a mà representades en matrius
 *  de 32x32 posicions on cada posició pot ser 0 (no hi ha traç) o 1 (n'hi ha).
 *  Els arxius de patrons contenen un cert nombre d'imatges una rere l'altre,
 *  per cada imatge s'indiquen 32 línies amb 32 columnnes de caràcters 0s i 1s i una
 *  darrera línia amb la xifra representada (espai en blanc + xifra entre 0 i 9).
 *  Aquests arxius es corresponen amb conjunts reals que es fan servir en el mon
 *  del ML i sobre els quals podeu trobar més informació en l'enllaç:
 *  https://archive.ics.uci.edu/ml/index.php
 */
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* readImg(FILE* fd) {
    char* img;
    img = (char*)malloc(img_dim_x * img_dim_y);

    for (int i = 0; i < img_dim_x; i++) {
        if (fscanf(fd, "%s\n", &img[i * img_dim_x]) != 1) {
            free(img);
            fprintf(stderr, "Error reading image\n");
            exit(-1);
        }
        for (int j = 0; j < img_dim_y; j++)
            img[i * img_dim_y + j] -= '0';
    }
    return img;
}

char** loadPatternSet(int nf, char* fname, int trainS) {
    char** tset;
    FILE* fd;
    int c;

    if ((fd = fopen(fname, "rb")) == NULL)
        return NULL;

    tset = (char**)malloc(nf * sizeof(char*));
    int error = 0;

    for (int i = 0; i < nf; i++) {
        if ((tset[i] = readImg(fd)) == NULL) {
            error = 1;
            break;
        }
        memset(desired_outputs[i], 0, num_out_layer * sizeof(float));
        if (fscanf(fd, "%d\n", &c) != 1) {
            error = 1;
            break;
        }
        if (trainS)
            desired_outputs[i][c] = 1.0;
        else
            Validation[i] = c;
    }
    fclose(fd);
    if (error)
        return NULL;
    return tset;
}

void printImg(char* Img, int x) {
    printf("Pattern:\n");
    for (int i = 0; i < 1024; i++) {
        printf("%c", Img[i] + '0');
        if ((i != 0) && !(i % 32))
            printf("\n");
    }
    printf("\nTarget:\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", desired_outputs[x][i]);
    printf("\n");
}

double elapsedT(struct timeval begin) {
    struct timeval end;

    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    return seconds + microseconds * 1e-6;
}