#include "mat.h"

mat *allocatemat(long r, long c) {
    mat *m = malloc(sizeof(mat));
    m->data = malloc(sizeof(double) * r * c);
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

    while (getline(&line, &len, fp) != -1) {   
        char *pch = strtok(line, delim);

        while (pch != NULL) {
            outdata->data[arr_sz++] = strtod(pch, &endptr);
            pch = strtok(NULL, delim);
        }
    }

    free(line);
    fclose(fp);

    return outdata;
}

long matsize(mat *m) {
    return m->r * m->c;
}

void matmul(mat *a, mat *b, mat *out) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        a->r, b->c, b->r,
        1,
        a->data, b->r,
        b->data, b->c,
        1,
        out->data, 1
    );
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
    for (long r = 0; r < m->r; r++) {
        for (long c = 0; c < m->c; c++) {
            printf("%.3f ", m->data[m->c * r + c]);
        }
        printf("\n");
    }
}

void subtract(mat *a, mat *b, mat *out) {
    for (long i = 0; i < matsize(a); i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}