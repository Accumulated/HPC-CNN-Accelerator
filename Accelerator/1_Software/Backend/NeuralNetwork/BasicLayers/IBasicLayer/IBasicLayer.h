#pragma once

class IBasicLayer {

public:
    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~IBasicLayer() {}

    virtual Matrix* operator()(Matrix* D_input) = 0;

};