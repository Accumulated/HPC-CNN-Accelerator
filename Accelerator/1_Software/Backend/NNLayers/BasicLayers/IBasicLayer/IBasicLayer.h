#pragma once

class IBasicLayer {

public:
    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~IBasicLayer() {}

    Matrix* operator()(Matrix* D_input) = 0;

};