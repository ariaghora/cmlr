#!/bin/bash

echo 'compiling fitlinreg.c';

gcc -g -o fitlinreg.bin \
    *.c \
    -lopenblas -lpthread  -Wall;

echo 'compiling finished';