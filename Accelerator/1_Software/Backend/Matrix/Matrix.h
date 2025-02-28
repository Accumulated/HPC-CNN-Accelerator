#pragma once


class Matrix {

public:

    int width;
    int height;
    int depth;
    int density = 1;
    const float* elements_ref;
    float* elements;
    MatrixType MType;

    ~Matrix();

    Matrix(int height, int width, int depth, int density, const float* elements, MatrixType MType);
    Matrix(int height, int width, int depth, const float* elements_ref, MatrixType MType);
    void Matrix_SetDimensions(int height, int width, int depth);
    void Matrix_DumpDeviceMemory();

private:
    void show_me_enhanced(float* ptr, const char* Name);

};