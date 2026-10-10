/*
 *
 * randomizer.c
 * funcions de aleatorietat de la xarxa Neuronal
 *
 * Created: 2-11-2024
 * by: Christian Germer
 *
 *
 */

#ifndef RANDOMIZER_H
#define RANDOMIZER_H

extern int seed;
extern int debug;
int rando();
float random_between_two(float min, float max);

#endif