#pragma once


class Matrix {

public:

    int width;
    int height;
    int depth;
    const float* elements_ref;
    float* elements;
    MatrixType MType;

    ~Matrix();

    Matrix(int height, int width, int depth, const float* elements, MatrixType MType);
    void Matrix_SetDimensions(int height, int width, int depth);

};