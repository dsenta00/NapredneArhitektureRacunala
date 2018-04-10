
#include <thread>
#include "Sudoku.h"

Sudoku::Sudoku(const char initialMatrix[LENGTH][LENGTH])
{
    this->matrix = Matrix(new SudokuMatrix(initialMatrix));
}

void
Sudoku::solve()
{
    while (!this->matrix->solved())
    {
        this->solveInRange(1, 9);
    }
}

void Sudoku::print()
{
    this->matrix->print();
}

void
Sudoku::solveInRange(int from, int to)
{
    int number = from;

    do
    {
        SudokuMatrix &currentMatrix = *this->matrix;
        SudokuMatrix matrix(currentMatrix);

        matrix.findSolutionForANumber(number);

        currentMatrix.merge(matrix);
        number++;
    } while (number <= to);
}
