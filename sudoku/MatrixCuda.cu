#include <cstring>
#include <memory>
#include <functional>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "MatrixCuda.h"

#if defined(WIN32) || defined(WIN64)
#define and &&
#endif

__global__ void init(Field *field,
                     Field *fields,
                     int *numbers,
                     int *total,
                     int i,
                     int j,
                     char value)
{
    *(fields + LENGTH * i + j) = *field;

    if (value > SUDOKU_FIELD_EMPTY)
    {
        numbers[value]++;
        (*total)++;
    }
}

MatrixCuda::MatrixCuda(const char initialMatrix[LENGTH][LENGTH])
{
    cudaMalloc((void **)&this->matrix, 
               LENGTH * LENGTH * sizeof(Field));

    cudaMalloc((void **)&this->numbers, 
               (LENGTH + 1) * sizeof(int));

    cudaMalloc((void **)&this->total,
               sizeof(int));

    cudaMemset(this->numbers, 0, (LENGTH + 1) * sizeof(int));

    *this->total = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            Field *field;

            cudaMalloc((void **)&field, sizeof(Field));
            init << <1, 1 >> >(field, this->matrix, numbers, total, i, j, initialMatrix[i][j]);
        }
    }
}

MatrixCuda::MatrixCuda(MatrixCuda &matrix)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            Field *field;

            cudaMalloc((void **)&field, sizeof(Field));
            init<<<1,1>>>(field, this->matrix, numbers, total, i, j, (matrix.matrix + LENGTH * i + j)->getValue());
        }
    }
}

void
MatrixCuda::merge(MatrixCuda &matrix)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            char value = (matrix.matrix + LENGTH * i + j)->getValue();

            if (value > SUDOKU_FIELD_EMPTY)
            {
                Field *field = (this->matrix + LENGTH * i + j);

                if (field->getValue() > SUDOKU_FIELD_EMPTY)
                {
                    continue;
                }

                field->setValue(value);
                this->numbers[value]++;
                this->total++;
            }
        }
    }
}

void
MatrixCuda::findSolutionForANumber(int number)
{
    this->fillMatrix(number);

    this->solveEmptyFieldsInRow(number);
    this->solveEmptyFieldsInColumn(number);
    this->solveEmptyFieldsInBox(number);
    //this->solveNumberAsAnOnlyOption();

    this->unfillMatrix();
}

void
MatrixCuda::fillMatrix(int number)
{
    this->foreach([=](Field *field) {
        if (field->getValue() == number)
        {
            this->markFields(
                field->getRowIndex(),
                field->getColumnIndex()
                );
        }
    });

    for (int i = 0; i < LENGTH; i += BOX_LENGTH)
    {
        for (int j = 0; j < LENGTH; j += BOX_LENGTH)
        {
            if (boxContainsNumber(i, j, number))
            {
                continue;
            }

            bool twoFilledRows = boxHasTwoFilledRows(i, j);
            bool twoFilledColumns = boxHasTwoFilledColumn(i, j);

            if (twoFilledColumns and twoFilledRows)
            {
                continue;
            }

            if (twoFilledColumns)
            {
                findEmptyColumnAndMark(i, j);
            }

            if (twoFilledRows)
            {
                findEmptyRowAndMark(i, j);
            }
        }
    }
}

void
MatrixCuda::markColumn(int columnIndex)
{
    for (int counter = 0; counter < LENGTH; counter++)
    {
        auto *field = (this->matrix + LENGTH * counter + columnIndex);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
MatrixCuda::markRow(int rowIndex)
{
    for (int counter = 0; counter < LENGTH; counter++)
    {
        auto *field = (this->matrix + LENGTH * rowIndex + counter);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
MatrixCuda::markBox(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            auto *field = (this->matrix + LENGTH * startX + startY);

            if (field->getValue() == SUDOKU_FIELD_EMPTY)
            {
                field->setValue(SUDOKU_FIELD_MARKED);
            }
        }
    }
}

void
MatrixCuda::markFields(int rowIndex, int columnIndex)
{
    this->markRow(rowIndex);
    this->markColumn(columnIndex);
    this->markBox(rowIndex, columnIndex);
}

bool
MatrixCuda::boxContainsNumber(int rowIndex, int columnIndex, int number)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == number)
            {
                return true;
            }
        }
    }

    return false;
}

bool
MatrixCuda::boxHasTwoFilledRows(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    int filledRows = 3;

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == SUDOKU_FIELD_EMPTY)
            {
                filledRows--;
                break;
            }
        }
    }

    return filledRows == 2;
}

bool
MatrixCuda::boxHasTwoFilledColumn(int rowIndex, int columnIndex)
{
    int startX;
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    int filledColumns = 3;

    for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
    {
        for (startX = START_INDEX_BOX(rowIndex); startX < endX; startX++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == SUDOKU_FIELD_EMPTY)
            {
                filledColumns--;
                break;
            }
        }
    }

    return filledColumns == 2;
}

void
MatrixCuda::findEmptyRowAndMark(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == SUDOKU_FIELD_EMPTY)
            {
                markRowExceptInBox(startX, columnIndex);
                return;
            }
        }
    }
}

void
MatrixCuda::findEmptyColumnAndMark(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == SUDOKU_FIELD_EMPTY)
            {
                markColumnExceptInBox(startY, rowIndex);
                return;
            }
        }
    }
}

void
MatrixCuda::markColumnExceptInBox(int columnIndex, int j)
{
    int startY = START_INDEX_BOX(j);
    int endY = END_INDEX_BOX(j);

    for (int counter = 0; counter < LENGTH; counter++)
    {
        if (counter >= startY and counter < endY)
        {
            continue;
        }

        auto *field = (this->matrix + LENGTH * counter + columnIndex);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
MatrixCuda::markRowExceptInBox(int rowIndex, int i)
{
    int startX = START_INDEX_BOX(i);
    int endX = END_INDEX_BOX(i);

    for (int counter = 0; counter < LENGTH; counter++)
    {
        if (counter >= startX and counter < endX)
        {
            continue;
        }

        auto *field = (this->matrix + LENGTH * rowIndex + counter);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
MatrixCuda::solveEmptyFieldsInRow(int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        if (this->countEmptyFieldsRow(i) == 1)
        {
            this->solveEmptyFieldsRow(i, number);
        }
    }
}

void
MatrixCuda::solveEmptyFieldsInColumn(int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        if (this->countEmptyFieldsColumn(i) == 1)
        {
            this->solveEmptyFieldsColumn(i, number);
        }
    }
}

void
MatrixCuda::solveEmptyFieldsInBox(int number)
{
    for (int i = 0; i < LENGTH; i += 3)
    {
        for (int j = 0; j < LENGTH; j += 3)
        {
            if (this->countEmptyFieldsBox(i, j) == 1)
            {
                this->solveEmptyFieldsBox(i, j, number);
            }
        }
    }
}

void
MatrixCuda::unfillMatrix()
{
    this->foreach([=](Field *field) {
        if (field->getValue() == SUDOKU_FIELD_MARKED)
        {
            field->setValue(SUDOKU_FIELD_EMPTY);
        }
    });
}

int
MatrixCuda::countEmptyFieldsRow(int rowIndex)
{
    int counter = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        if ((this->matrix + LENGTH * i + rowIndex)->getValue() == SUDOKU_FIELD_EMPTY)
        {
            counter++;
        }
    }

    return counter;
}

void
MatrixCuda::solveEmptyFieldsRow(int rowIndex, int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        auto *field = (this->matrix + LENGTH * i + rowIndex);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(number);
            this->numbers[number]++;
            this->total++;
            this->markFields(i, rowIndex);

            return;
        }
    }
}

int
MatrixCuda::countEmptyFieldsColumn(int columnIndex)
{
    int counter = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        if ((this->matrix + LENGTH * columnIndex + i)->getValue() == SUDOKU_FIELD_EMPTY)
        {
            counter++;
        }
    }

    return counter;
}

void
MatrixCuda::solveEmptyFieldsColumn(int columnIndex, int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        auto *field = (this->matrix + LENGTH * columnIndex + i);

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(number);
            this->numbers[number]++;
            this->total++;
            this->markFields(columnIndex, i);

            return;
        }
    }
}

int
MatrixCuda::countEmptyFieldsBox(int rowIndex, int columnIndex)
{
    int counter = 0;
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if ((this->matrix + LENGTH * startX + startY)->getValue() == SUDOKU_FIELD_EMPTY)
            {
                counter++;
            }
        }
    }

    return counter;
}

void
MatrixCuda::solveEmptyFieldsBox(int rowIndex, int columnIndex, int number)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            auto *field = (this->matrix + LENGTH * startX + startY);

            if (field->getValue() == SUDOKU_FIELD_EMPTY)
            {
                field->setValue(number);
                this->numbers[number]++;
                this->total++;
                this->markFields(startX, startY);

                return;
            }
        }
    }
}

void
MatrixCuda::print()
{
    printf("\r\n");

    for (int i = 0; i < LENGTH; i++)
    {
        if (i % 3 == 0)
        {
            printf("+-----------------------+\r\n");
        }

        for (int j = 0; j < LENGTH; j++)
        {
            if (j % 3 == 0)
            {
                printf("| ");
            }

            auto *field = (this->matrix + LENGTH * i + j);

            switch (field->getValue())
            {
            case SUDOKU_FIELD_EMPTY:
                printf("  ");
                break;
            case SUDOKU_FIELD_MARKED:
                printf("M ");
                break;
            default:
                printf("%d ", field->getValue());
            }
        }

        printf("|\r\n");
    }

    printf("------------------------+");
    printf("\r\n");
}

void
MatrixCuda::solveNumberAsAnOnlyOption()
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            Field *field = (this->matrix + LENGTH * i + j);

            if (field->getValue() == SUDOKU_FIELD_EMPTY)
            {
                int resultNumber = this->solveEmptyField(field);

                if (resultNumber > SUDOKU_FIELD_EMPTY)
                {
                    field->setValue(resultNumber);
                    this->numbers[resultNumber]++;
                    this->total++;
                }
            }
        }
    }
}

int
MatrixCuda::solveEmptyField(Field *field)
{
    int numbers[LENGTH + 1] = { 0 };

    int rowIndex = field->getRowIndex();
    int columnIndex = field->getColumnIndex();

    for (int counter = 0; counter < LENGTH; counter++)
    {
        int number = (this->matrix + LENGTH * rowIndex + counter)->getValue();

        if (number > SUDOKU_FIELD_EMPTY)
        {
            numbers[number]++;
        }
    }

    for (int counter = 0; counter < LENGTH; counter++)
    {
        int number = (this->matrix + LENGTH * counter + columnIndex)->getValue();

        if (number > SUDOKU_FIELD_EMPTY)
        {
            numbers[number]++;
        }
    }

    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            int number = (this->matrix + LENGTH * startX + startY)->getValue();

            if (number > SUDOKU_FIELD_EMPTY)
            {
                numbers[number]++;
            }
        }
    }

    int numberOfEmpty = 0;

    for (int counter = 1; counter <= LENGTH; counter++)
    {
        if (numbers[counter] == 0)
        {
            numberOfEmpty++;
        }
    }

    if (numberOfEmpty == 1)
    {
        for (int counter = 1; counter <= LENGTH; counter++)
        {
            if (numbers[counter] == 0)
            {
                return counter;
            }
        }
    }

    return SUDOKU_FIELD_EMPTY;
}


void
MatrixCuda::foreach(std::function<void(Field *)> func)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            func((this->matrix + LENGTH * i + j));
        }
    }
}

bool
MatrixCuda::solved()
{
    return *this->total >= TOTAL_FIELDS;
}

MatrixCuda::~MatrixCuda()
{
    cudaFree(this->matrix);
}