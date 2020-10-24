#!/bin/bash

echo 'compiling fitlogistic.c';

gcc -g -o fitlogistic.bin \
    *.c \
    -lopenblas -lpthread  -Wall;

echo 'compiling finished';