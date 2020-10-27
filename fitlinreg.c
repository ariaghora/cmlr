#include <cblas.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mat.h"

/* training parameters */
long optimize_bias = 1;
long print_summary = 0;
long max_iter      = 1000;
long ds_row        = 0;
long ds_col        = 0;

long shape[2];
char *shape_str;

void show_help() {
    printf("fitlinreg version 0.1.0 - A Fast Multiple Linear Regression\n"
           "Copyright (C) 2020 Sorciencer \n"
           "Usage: fitlinreg [-f filename][-h]\n"
           "\n"
           "Options:\n"
           "  -f  set input filename\n"
           "  -h  show this help\n"
           "  -i  show max number of iterations (default=1000)\n"
           "  -s  determine the row and column, separated by a comma\n"
           "  -r  show training report\n"
           );
}

int main(int argc, char *argv[]) {
    char *filename = "";
    long label_idx = 8;
    double bias    = 0;
    double lr      = 0.1;
    double tol     = 1e-4;

    clock_t t;
    double time_taken;
    double mse;
    double dbias;
    
    char *endptr;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0) {
            if (i < argc - 1) filename = argv[i + 1];
        }

        if (strcmp(argv[i], "-r") == 0) {
            print_summary = 1;
        }

        if (strcmp(argv[i], "-s") == 0) {
            char *pch = strtok(argv[i + 1], ",");
            int i = 0;
            while (pch != NULL) {
                shape[i++] = strtol(pch, &endptr, 10);
                pch = strtok(NULL, ",");
            }
            ds_row = shape[0];
            ds_col = shape[1];
        }

        if (strcmp(argv[i], "-i") == 0) {
            max_iter = strtol(argv[i + 1], &endptr, 10);
        }
    }

    if (filename[0] == '\0') {
        show_help();
        return -1;
    }

    if ((ds_row == 0) || (ds_col == 0)) {
        printf("invalid number of columns or rows\n");
        return -1;
    }

    mat *dataset  = read_csv(filename, ds_row, ds_col, ",");
    long feat_row = dataset->r;
    long feat_col = dataset->c - 1;

    mat *x = allocatemat(dataset->r, feat_col - 1);
    matrange(dataset, 0, 0, dataset->r, feat_col - 1, x);

    mat *y = allocatemat(dataset->r, 1);
    matrange(dataset, 0, label_idx, dataset->r, 1, y);

    mat *theta = allocatemat(feat_col - 1, 1);
    for (int i = 0; i < feat_col - 1; i++) 
        theta->data[i] = 0.0;

    mat *losses  = allocatemat(feat_row, 1);
    mat *losses2 = allocatemat(feat_row, 1); 

    t = clock();

    /* gradient descent optimization */
    for (int i = 0; i < max_iter; i++) {
        mat *yhat;
        yhat    = allocatemat(feat_row, 1);

        mat *dtheta;
        dtheta  = allocatemat(feat_col - 1, 1);

        /* prediction */
        matmul(x, theta, 0, 0, yhat);
        addscalar(yhat, bias, yhat);
    
        /* compute losses */
        subtract(yhat, y, losses);

        /* Compute the gradient of theta, then update the theta */
        matmul(x, losses, 1, 0, dtheta);
        cblas_dscal(matsize(dtheta), lr * 1.0 / feat_row, dtheta->data, 1);
        subtract(theta, dtheta, theta);

        /* Compute the gradient of bias, then update the bias */
        if (optimize_bias) {
            dbias = lr * cblas_dasum(feat_row, losses->data, 1) / feat_row;
            bias -= dbias;
        }
        
        /* log the mean-squared-error (MSE) */
        mul(losses, losses, losses2); /* losses2 = losses^2 */
        mse = cblas_dasum(feat_row, losses2->data, 1) / feat_row;

        /* exit gradient descent when the loss is lower than a threshold */
        if (mse < tol) break;

        /* free dtheta and yhat for current iteration */
        freemat(dtheta);
        freemat(yhat);
    }

    t = clock() - t;

    if (print_summary) {
        time_taken = ((double) t) / CLOCKS_PER_SEC;
        printf("time taken: %f(s)\n", time_taken);

        mul(losses, losses, losses2);
        mse = cblas_dasum(feat_row, losses2->data, 1) / feat_row;
        printf("mse %f\n", mse);    

        printf("coefficients:\n");
        for (int i = 0; i < matsize(theta); i++) {
            printf("%f", theta->data[i]);
            if (i < matsize(theta) - 1) printf(", ");
        }
        printf("\nintercept: %f\n", bias);
    }

    freemat(dataset);
    freemat(x);
    freemat(y);
    freemat(theta);
    freemat(losses);
    freemat(losses2);
    
    return 0;
}