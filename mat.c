#include <cblas.h>
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mat.h"

mat *allocatemat(long r, long c) {
    mat *m = malloc(sizeof(mat));
    m->data = calloc(r * c, sizeof(double));
    m->r = r;
    m->c = c;
    return m;
}

mat *matones(long r, long c) {
    mat *m = allocatemat(r, c);
    for (int i = 0; i < r * c; i++) {
        m->data[i] = 1.0;
    }
    return m;
}

mat *read_csv(char *filename, long row, long col, char *delim) {
    FILE *fp   = fopen(filename, "r");
    char *line = NULL;
    size_t len = 0;
    int arr_sz = 0; 
    char *endptr;

    mat *outdata = allocatemat(row, col);

    int r_cnt = 0;
    while ((getline(&line, &len, fp) != -1) && (r_cnt < row)) {   
        char *pch = strtok(line, delim);

        for (int i = 0; i < col; i++) {
            outdata->data[arr_sz++] = strtod(pch, &endptr);
            pch = strtok(NULL, delim);
        }
        pch = NULL;

        r_cnt++;
    }

    free(line);
    fclose(fp);

    return outdata;
}

long matsize(mat *m) {
    return m->r * m->c;
}

void matmul(mat *a, mat *b, long tr_a, long tr_b, mat *out) {
    long a_trans = tr_a == 0 ? CblasNoTrans : CblasTrans;
    long b_trans = tr_b == 0 ? CblasNoTrans : CblasTrans;

    int A_height = tr_a ? a->c  : a->r;
    int A_width  = tr_a ? a->r : a->c;
    int B_width  = tr_b ? b->r : b->c;
    int m = A_height;
    int n = B_width;
    int k = A_width;

    int lda = tr_a ? m : k;
    int ldb = tr_b ? k : n;

    cblas_dgemm(CblasRowMajor, a_trans, b_trans,
        m, n, k,
        1,
        a->data, lda,
        b->data, ldb,
        1,
        out->data, n
    );
}

void mul(mat *a, mat *b, mat *out) {
    for (int i = 0; i < a->r * a->c; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void addscalar(mat *a, double x, mat *out) {
    for (int i = 0; i < a->r * a->c; i++) {
        out->data[i] = a->data[i] + x;
    }
}

void freemat(mat *m) {
    free(m->data);
    free(m);
}

void matrange(mat *m, long row, long col, long height, long width, mat*out) {
    long offset = 0;
    for (long r = row; r < row + height; r++) {
        for (long c = col; c < col + width; c++) {
            out->data[offset++] = m->data[m->c * r + c];
        }
    }
    out->r = height;
    out->c = width;
}

void printmat(mat *m) {
    int off = 0;
    for (long r = 0; r < m->r; r++) {
        for (long c = 0; c < m->c; c++) {
            printf("%.3f ", m->data[off++]);
        }
        printf("\n");
    }
}

void subtract(mat *a, mat *b, mat *out) {
    for (long i = 0; i < matsize(a); i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}