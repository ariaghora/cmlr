#!/bin/bash

echo 'compiling fitlinreg.c';

gcc -O3 -g -o fitlinreg.bin \
    *.c \
    -lopenblas -Wall;

echo 'compiling finished';