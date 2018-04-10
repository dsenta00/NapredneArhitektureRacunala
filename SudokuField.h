#pragma once

#include <cstdio>
#include <iostream>

#define SUDOKU_FIELD_EMPTY (0)
#define SUDOKU_FIELD_MARKED ('M')

class SudokuField {
public:
    char getValue();
    SudokuField *setValue(char value);
    int getRowIndex();
    SudokuField *setRowIndex(int rowIndex);
    int getColumnIndex();
    SudokuField *setColumnIndex(int columnIndex);

    static SudokuField *create();
protected:
    char value;
    int rowIndex;
    int columnIndex;
};
