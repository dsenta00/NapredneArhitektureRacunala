#pragma once

#include <memory>
#include <functional>
#include "Field.h"

#define LENGTH (9)
#define BOX_LENGTH (3)
#define TOTAL_FIELDS ((LENGTH) * (LENGTH))
#define START_INDEX_BOX(_INDEX) (((_INDEX) / BOX_LENGTH) * BOX_LENGTH)
#define END_INDEX_BOX(_INDEX) ((((_INDEX) / BOX_LENGTH) * BOX_LENGTH) + BOX_LENGTH)

using FieldPtr = std::shared_ptr<Field>;

class Matrix {
public:
    explicit Matrix(const char initialMatrix[LENGTH][LENGTH]);
    Matrix(Matrix &matrix);
    void merge(Matrix &matrix);
    void findSolutionForANumber(int number);
    void print();
    bool solved();
protected:
    void foreach(std::function<void(Field *)> func);

    void fillMatrix(int number);
    void unfillMatrix();

    void markColumn(int columnIndex);
    void markRow(int rowIndex);
    void markBox(int rowIndex, int columnIndex);
    void markFields(int rowIndex, int columnIndex);

    bool boxContainsNumber(int rowIndex, int columnIndex, int number);

    bool boxHasTwoFilledRows(int rowIndex, int columnIndex);
    bool boxHasTwoFilledColumn(int rowIndex, int columnIndex);

    void findEmptyRowAndMark(int rowIndex, int columnIndex);
    void findEmptyColumnAndMark(int rowIndex, int columnIndex);

    void markColumnExceptInBox(int columnIndex, int j);
    void markRowExceptInBox(int rowIndex, int i);

    void solveEmptyFieldsInRow(int number);
    void solveEmptyFieldsInColumn(int number);
    void solveEmptyFieldsInBox(int number);
    void solveNumberAsAnOnlyOption();

    int countEmptyFieldsRow(int rowIndex);
    void solveEmptyFieldsRow(int rowIndex, int number);

    int countEmptyFieldsColumn(int columnIndex);
    void solveEmptyFieldsColumn(int columnIndex, int number);

    int countEmptyFieldsBox(int rowIndex, int columnIndex);
    void solveEmptyFieldsBox(int rowIndex, int columnIndex, int number);

    int solveEmptyField(Field *field);

    FieldPtr matrix[LENGTH][LENGTH];
    int numbers[LENGTH + 1];
    int total;
};