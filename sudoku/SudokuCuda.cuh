#include "AbstractSudokuCuda.h"


class SudokuCuda : public AbstractSudokuCuda {
public:
    explicit SudokuCuda(const char initialMatrix1[LENGTH][LENGTH]);
    void solveUsing2Threads() override;
    void solveUsing4Threads() override;
    void solveUsing8Threads() override;
    void solveUsing12Threads() override;
};