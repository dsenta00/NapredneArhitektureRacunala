#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "Matrix.h"
#include "AbstractSudoku.h"

class SudokuCpp : public AbstractSudoku {
public:
    explicit SudokuCpp(const char initialMatrix1[LENGTH][LENGTH]);
    void solveUsing2Threads() override;
    void solveUsing4Threads() override;
    void solveUsing8Threads() override;
    void solveUsing12Threads() override;
};