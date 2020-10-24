#include <stdio.h>
#include <time.h>
#include "mat.h"

int main(int argc, char *argv[]) {
    long ds_row   = 768;
    long ds_col   = 10;
    long feat_row = 768;
    long feat_col = 8;

    clock_t t;

    mat *dataset = read_csv("energy.csv", ds_row, ds_col, ",");
    mat *x       = allocatemat(dataset->r, feat_col);
    mat *y       = allocatemat(dataset->r, 1);

    matrange(dataset, 0, 0, dataset->r, feat_col, x);
    matrange(dataset, 0, 8, dataset->r, 1, y);

    mat *theta = allocatemat(feat_col, 1);
    for (int i = 0; i < feat_col; i++) 
        theta->data[i] = .5;

    mat *yhat   = allocatemat(feat_row, theta->c);
    mat *losses = allocatemat(feat_row, 1);
    
    double bias = 0;   

    t = clock();

    // prediction
    matmul(x, theta, yhat);
    addscalar(yhat, bias, yhat);
 
    // compute losses
    subtract(y, yhat, losses);

    t = clock() - t;
    printf("%f", ((double) t) / CLOCKS_PER_SEC);

    // printmat(losses);
    // printmat(yhat);

    freemat(x);
    freemat(theta);
    freemat(yhat);
    
    return 0;
}