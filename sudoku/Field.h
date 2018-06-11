#pragma once

#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define SUDOKU_FIELD_EMPTY (0)
#define SUDOKU_FIELD_MARKED ('M')

class Field {
public:
    char getValue();
    Field *setValue(char value);
    int getRowIndex();
    Field *setRowIndex(int rowIndex);
    int getColumnIndex();
    Field *setColumnIndex(int columnIndex);

    static Field *create();

    char value;
    int rowIndex;
    int columnIndex;
};