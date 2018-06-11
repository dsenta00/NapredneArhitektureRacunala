
#include <thread>
#include "AbstractSudoku.h"

AbstractSudoku::AbstractSudoku(const char initialMatrix[LENGTH][LENGTH])
{
    this->matrix = MatrixPtr(new Matrix(initialMatrix));
}

void
AbstractSudoku::solve()
{
    while (!this->matrix->solved())
    {
        this->solveInRange(1, 9);
    }
}

void AbstractSudoku::print()
{
    this->matrix->print();
}

void
AbstractSudoku::solveInRange(int from, int to)
{
    int number = from;

    do
    {
        Matrix &currentMatrix = *this->matrix;
        Matrix matrix(currentMatrix);

        matrix.findSolutionForANumber(number);

        currentMatrix.merge(matrix);
        number++;
    } while (number <= to);
}