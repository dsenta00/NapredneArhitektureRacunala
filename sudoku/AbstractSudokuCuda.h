#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "MatrixCuda.h"

class AbstractSudokuCuda {
public:
    explicit AbstractSudokuCuda(const char initialMatrix[LENGTH][LENGTH]);
    void print();
    void solve();
    virtual void solveUsing2Threads() = 0;
    virtual void solveUsing4Threads() = 0;
    virtual void solveUsing8Threads() = 0;
    virtual void solveUsing12Threads() = 0;
    void solveInRange(int from, int to);
    ~AbstractSudokuCuda();
protected:
    MatrixCuda *matrix;
};