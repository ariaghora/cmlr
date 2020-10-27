#ifndef MAT_H
#define MAT_H

typedef struct {
    double *data;
    long r;
    long c;
} mat;

long matsize(mat *m);
mat *allocatemat(long r, long c);
mat *matones(long r, long c);
mat *read_csv(char *filename, long row, long col, char *delim);
void matmul(mat *a, mat *b, long tr_a, long tr_b, mat *out);
void mul(mat *a, mat *b, mat *out);
void addscalar(mat *a, double x, mat *out);
void freemat(mat *m);
void matrange(mat *m, long row, long col, long height, long width, mat*out);
void printmat(mat *m);
void subtract(mat *a, mat *b, mat *out);

#endif