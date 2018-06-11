#include "SudokuCuda.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>

SudokuCuda::SudokuCuda(const char initialMatrix[LENGTH][LENGTH]) : AbstractSudokuCuda(initialMatrix)
{
}

void
SudokuCuda::solveUsing2Threads()
{
    // Empty on purpose
}

void
SudokuCuda::solveUsing4Threads()
{
    // Empty on purpose
}

void
SudokuCuda::solveUsing8Threads()
{
    // Empty on purpose
}

void
SudokuCuda::solveUsing12Threads()
{
    // Empty on purpose
}