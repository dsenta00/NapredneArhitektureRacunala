#pragma once

#include "AbstractSudoku.h"

class SudokuC : public AbstractSudoku {
public:
    explicit SudokuC(const char initialMatrix[LENGTH][LENGTH]);

    void solveUsing2Threads() override;
    void solveUsing4Threads() override;
    void solveUsing8Threads() override;
    void solveUsing12Threads() override;
};
