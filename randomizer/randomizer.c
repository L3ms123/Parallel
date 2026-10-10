/*
 * randomizer.c
 *
 * Created: 2-11-2024
 * by: Christian Germer
 *
 *
 * funcions de aleatorietat de la xarxa Neuronal
 */

#include "randomizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//-----------RANDOM FUNCTIONS------------
int rando() {
    seed = (214013 * seed + 2531011);
    return seed >> 16;
}

float random_between_two(float min, float max) {
    return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}
