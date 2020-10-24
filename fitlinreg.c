#include <stdio.h>
#include <time.h>
#include "mat.h"

int main(int argc, char *argv[]) {
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
    long print_summary = 1;
    long max_iter      = 1000;

    clock_t t;
    double time_taken;
    double mse;
    double dbias;

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