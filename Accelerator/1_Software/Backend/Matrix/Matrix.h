#pragma once


class Matrix {

public:

    int width;
    int height;
    int depth;
    float* elements_ref;
    float* elements;
    MatrixType MType;

    ~Matrix();
    Matrix();

    Matrix(int height, int width, int depth, float* elements, MatrixType MType);
    void Matrix_SetDimensions(int height, int width, int depth);

};