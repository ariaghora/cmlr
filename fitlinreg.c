#include <stdio.h>
#include <time.h>
#include "mat.h"

void show_help() {
    printf("fitlinreg version 0.1.0 - A Fast Multiple Linear Regression\n"
           "Copyright (C) 2020 Aria Ghora Prabono \n"
           "Usage: fitlinreg [-f filename][-h]\n"
           "\n"
           "Options:\n"
           "  -f   set input filename\n"
           "  -h   show this help\n"
           "  -s   show summary\n"
           );
}

int main(int argc, char *argv[]) {
    char *filename = "";
    long ds_row    = 768;
    long ds_col    = 10;
    long feat_row  = 768;
    long feat_col  = 8;
    long label_idx = 8;
    double bias    = 0;
    double lr      = 0.1;
    double tol     = 1e-4;

    // training params
    long optimize_bias = 1;
    long print_summary = 0;
    long max_iter      = 1000;

    clock_t t;
    double time_taken;
    double mse;
    double dbias;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0) {
            if (i < argc - 1) filename = argv[i + 1];
        }

        if (strcmp(argv[i], "-s") == 0) {
            print_summary = 1;
        }
    }

    if (filename[0] == '\0') {
        show_help();
        return -1;
    }

    mat *dataset = read_csv("energy_std.csv", ds_row, ds_col, ",");

    mat *x = allocatemat(dataset->r, feat_col);
    matrange(dataset, 0, 0, dataset->r, feat_col, x);

    mat *y = allocatemat(dataset->r, 1);
    matrange(dataset, 0, label_idx, dataset->r, 1, y);

    mat *theta = allocatemat(feat_col, 1);
    for (int i = 0; i < feat_col; i++) 
        theta->data[i] = .5;

    mat *losses  = allocatemat(feat_row, 1);
    mat *losses2 = allocatemat(feat_row, 1); 

    t = clock();

    // gradient descent    
    for (int i = 0; i < max_iter; i++) {
        mat *yhat;
        yhat    = allocatemat(feat_row, 1);

        mat *dtheta;
        dtheta  = allocatemat(feat_col, 1);

        // prediction 
        matmul(x, theta, 0, 0, yhat);
        addscalar(yhat, bias, yhat);
    
        // compute losses
        subtract(yhat, y, losses);

        // dtheta
        matmul(x, losses, 1, 0, dtheta);
        cblas_dscal(matsize(dtheta), lr * 1.0 / feat_row, dtheta->data, 1);
        subtract(theta, dtheta, theta);

        // dbias
        if (optimize_bias) {
            dbias = lr * cblas_dasum(feat_row, losses->data, 1) / feat_row;
            bias -= dbias;
        }
        
        // log the mse
        mul(losses, losses, losses2); // losses^2
        mse = cblas_dasum(feat_row, losses2->data, 1) / feat_row;
        if (mse < tol) {
            break;
        }

        freemat(dtheta);
        freemat(yhat);
    }

    if (print_summary) {
        t = clock() - t;
        time_taken = ((double) t) / CLOCKS_PER_SEC;
        printf("time taken: %f(s)\n", time_taken);

        mul(losses, losses, losses2);
        mse = cblas_dasum(feat_row, losses2->data, 1) / feat_row;
        printf("mse %f\n", mse);    

        printf("coefficients:\n");
        printmat(theta);
        printf("intercept: %f\n", bias);

    }

    

    freemat(dataset);
    freemat(x);
    freemat(y);
    freemat(theta);
    freemat(losses);
    freemat(losses2);
    
    return 0;
}