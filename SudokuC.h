#pragma once

#include "Sudoku.h"

class SudokuC : public Sudoku {
public:
    explicit SudokuC(const char initialMatrix[LENGTH][LENGTH]);

    void solveUsing2Threads() override;
    void solveUsing4Threads() override;
    void solveUsing8Threads() override;
    void solveUsing12Threads() override;
};
