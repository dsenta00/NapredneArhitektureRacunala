#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "SudokuMatrix.h"
#include "Sudoku.h"

using Matrix = std::shared_ptr<SudokuMatrix>;

class SudokuCpp : public Sudoku {
public:
    explicit SudokuCpp(const char initialMatrix1[LENGTH][LENGTH]);
    void solveUsing2Threads() override;
    void solveUsing4Threads() override;
    void solveUsing8Threads() override;
    void solveUsing12Threads() override;
};