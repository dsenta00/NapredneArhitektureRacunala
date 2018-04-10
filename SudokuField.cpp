#include "SudokuField.h"


char
SudokuField::getValue()
{
    return this->value;
}

SudokuField *
SudokuField::setValue(char value)
{
    this->value = value;

    return this;
}

SudokuField *
SudokuField::create()
{
    return new SudokuField();
}

int
SudokuField::getRowIndex()
{
    return this->rowIndex;
}

SudokuField *
SudokuField::setRowIndex(int rowIndex)
{
    this->rowIndex = rowIndex;

    return this;
}

int
SudokuField::getColumnIndex()
{
    return this->columnIndex;
}

SudokuField *
SudokuField::setColumnIndex(int columnIndex)
{
    this->columnIndex = columnIndex;

    return this;
}
