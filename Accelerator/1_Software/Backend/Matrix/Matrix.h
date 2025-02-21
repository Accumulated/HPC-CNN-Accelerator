#pragma once


class Matrix {



public:

    int width;
    int height;
    int depth;
    float* elements;
    MatrixType MType;


    ~Matrix();

    Matrix(int depth, int height, int width, float* elements, MatrixType MType);


}