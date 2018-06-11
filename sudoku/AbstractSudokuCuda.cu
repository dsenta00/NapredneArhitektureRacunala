
#include <thread>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "AbstractSudokuCuda.h"

// hack
#define __globals__

AbstractSudokuCuda::AbstractSudokuCuda(const char initialMatrix[LENGTH][LENGTH])
{
    cudaMalloc((void **)&this->matrix, sizeof(MatrixCuda));
    *matrix = MatrixCuda(initialMatrix);
}

void
AbstractSudokuCuda::solve()
{
    while (!this->matrix->solved())
    {
        this->solveInRange(1, 9);
    }
}

void AbstractSudokuCuda::print()
{
    this->matrix->print();
}

__globals__ void solveInRangeCuda(MatrixCuda *matrix, 
                                 MatrixCuda *currentMatrix, 
                                 int number)
{
    matrix->findSolutionForANumber(number);
    currentMatrix->merge(*matrix);
}

void
AbstractSudokuCuda::solveInRange(int from, int to)
{
    int number = from;

    do
    {
        MatrixCuda *newMatrix = nullptr;
        cudaMalloc((void **)newMatrix, sizeof(MatrixCuda));
        *newMatrix = MatrixCuda(*matrix);
        solveInRangeCuda(newMatrix, matrix, number);
        number++;
    } while (number <= to);
}

AbstractSudokuCuda::~AbstractSudokuCuda()
{
    cudaFree(this->matrix);
}