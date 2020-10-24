#!/bin/bash

echo 'compiling fitlogistic.c';

gcc -o fitlogistic.bin \
    *.c \
    -lopenblas -lpthread  -Wall;

echo 'compiling finished';