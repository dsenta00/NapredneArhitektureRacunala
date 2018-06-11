#include <cstring>
#include "Matrix.h"
#include <memory>
#include <functional>


#if defined(WIN32) || defined(WIN64)
#define and &&
#endif

Matrix::Matrix(const char initialMatrix[LENGTH][LENGTH])
{
    memset(this->numbers, 0, sizeof(this->numbers));
    this->total = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            char value = initialMatrix[i][j];

            auto *field =
                Field::create()
                    ->setRowIndex(i)
                    ->setColumnIndex(j)
                    ->setValue(value);

            this->matrix[i][j] = FieldPtr(field);

            if (value > SUDOKU_FIELD_EMPTY)
            {
                this->numbers[value]++;
                this->total++;
            }
        }
    }
}

Matrix::Matrix(Matrix &matrix)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            char value = matrix.matrix[i][j]->getValue();

            auto *field =
                Field::create()
                    ->setRowIndex(i)
                    ->setColumnIndex(j)
                    ->setValue(value);

            this->matrix[i][j] = FieldPtr(field);

            if (value > SUDOKU_FIELD_EMPTY)
            {
                this->numbers[value]++;
                this->total++;
            }
        }
    }
}

void
Matrix::merge(Matrix &matrix)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            char value = matrix.matrix[i][j]->getValue();

            if (value > SUDOKU_FIELD_EMPTY)
            {
                Field *field = this->matrix[i][j].get();

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
Matrix::findSolutionForANumber(int number)
{
    this->fillMatrix(number);

    this->solveEmptyFieldsInRow(number);
    this->solveEmptyFieldsInColumn(number);
    this->solveEmptyFieldsInBox(number);
    //this->solveNumberAsAnOnlyOption();

    this->unfillMatrix();
}

void
Matrix::fillMatrix(int number)
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
Matrix::markColumn(int columnIndex)
{
    for (int counter = 0; counter < LENGTH; counter++)
    {
        auto *field = this->matrix[counter][columnIndex].get();

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
Matrix::markRow(int rowIndex)
{
    for (int counter = 0; counter < LENGTH; counter++)
    {
        auto *field = this->matrix[rowIndex][counter].get();

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
Matrix::markBox(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            auto *field = this->matrix[startX][startY].get();

            if (field->getValue() == SUDOKU_FIELD_EMPTY)
            {
                field->setValue(SUDOKU_FIELD_MARKED);
            }
        }
    }
}

void
Matrix::markFields(int rowIndex, int columnIndex)
{
    this->markRow(rowIndex);
    this->markColumn(columnIndex);
    this->markBox(rowIndex, columnIndex);
}

bool
Matrix::boxContainsNumber(int rowIndex, int columnIndex, int number)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if (this->matrix[startX][startY]->getValue() == number)
            {
                return true;
            }
        }
    }

    return false;
}

bool
Matrix::boxHasTwoFilledRows(int rowIndex, int columnIndex)
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
            if (this->matrix[startX][startY]->getValue() == SUDOKU_FIELD_EMPTY)
            {
                filledRows--;
                break;
            }
        }
    }

    return filledRows == 2;
}

bool
Matrix::boxHasTwoFilledColumn(int rowIndex, int columnIndex)
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
            if (this->matrix[startX][startY]->getValue() == SUDOKU_FIELD_EMPTY)
            {
                filledColumns--;
                break;
            }
        }
    }

    return filledColumns == 2;
}

void
Matrix::findEmptyRowAndMark(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if (this->matrix[startX][startY]->getValue() == SUDOKU_FIELD_EMPTY)
            {
                markRowExceptInBox(startX, columnIndex);
                return;
            }
        }
    }
}

void
Matrix::findEmptyColumnAndMark(int rowIndex, int columnIndex)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            if (this->matrix[startX][startY]->getValue() == SUDOKU_FIELD_EMPTY)
            {
                markColumnExceptInBox(startY, rowIndex);
                return;
            }
        }
    }
}

void
Matrix::markColumnExceptInBox(int columnIndex, int j)
{
    int startY = START_INDEX_BOX(j);
    int endY = END_INDEX_BOX(j);

    for (int counter = 0; counter < LENGTH; counter++)
    {
        if (counter >= startY and counter < endY)
        {
            continue;
        }

        auto *field = this->matrix[counter][columnIndex].get();

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
Matrix::markRowExceptInBox(int rowIndex, int i)
{
    int startX = START_INDEX_BOX(i);
    int endX = END_INDEX_BOX(i);

    for (int counter = 0; counter < LENGTH; counter++)
    {
        if (counter >= startX and counter < endX)
        {
            continue;
        }

        auto *field = this->matrix[rowIndex][counter].get();

        if (field->getValue() == SUDOKU_FIELD_EMPTY)
        {
            field->setValue(SUDOKU_FIELD_MARKED);
        }
    }
}

void
Matrix::solveEmptyFieldsInRow(int number)
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
Matrix::solveEmptyFieldsInColumn(int number)
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
Matrix::solveEmptyFieldsInBox(int number)
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
Matrix::unfillMatrix()
{
    this->foreach([=] (Field *field) {
        if (field->getValue() == SUDOKU_FIELD_MARKED)
        {
            field->setValue(SUDOKU_FIELD_EMPTY);
        }
    });
}

int
Matrix::countEmptyFieldsRow(int rowIndex)
{
    int counter = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        if (this->matrix[i][rowIndex]->getValue() == SUDOKU_FIELD_EMPTY)
        {
            counter++;
        }
    }

    return counter;
}

void
Matrix::solveEmptyFieldsRow(int rowIndex, int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        auto *field = matrix[i][rowIndex].get();

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
Matrix::countEmptyFieldsColumn(int columnIndex)
{
    int counter = 0;

    for (int i = 0; i < LENGTH; i++)
    {
        if (this->matrix[columnIndex][i]->getValue() == SUDOKU_FIELD_EMPTY)
        {
            counter++;
        }
    }

    return counter;
}

void
Matrix::solveEmptyFieldsColumn(int columnIndex, int number)
{
    for (int i = 0; i < LENGTH; i++)
    {
        auto *field = this->matrix[columnIndex][i].get();

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
Matrix::countEmptyFieldsBox(int rowIndex, int columnIndex)
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
            if (this->matrix[startX][startY]->getValue() == SUDOKU_FIELD_EMPTY)
            {
                counter++;
            }
        }
    }

    return counter;
}

void
Matrix::solveEmptyFieldsBox(int rowIndex, int columnIndex, int number)
{
    int startX = START_INDEX_BOX(rowIndex);
    int endX = END_INDEX_BOX(rowIndex);
    int startY;
    int endY = END_INDEX_BOX(columnIndex);

    for (; startX < endX; startX++)
    {
        for (startY = START_INDEX_BOX(columnIndex); startY < endY; startY++)
        {
            auto *field = this->matrix[startX][startY].get();

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
Matrix::print()
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

            auto *field = this->matrix[i][j].get();

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
Matrix::solveNumberAsAnOnlyOption()
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            Field *field = this->matrix[i][j].get();

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
Matrix::solveEmptyField(Field *field)
{
    int numbers[LENGTH + 1] = {0};

    int rowIndex = field->getRowIndex();
    int columnIndex = field->getColumnIndex();

    for (int counter = 0; counter < LENGTH; counter++)
    {
        int number = matrix[rowIndex][counter]->getValue();

        if (number > SUDOKU_FIELD_EMPTY)
        {
            numbers[number]++;
        }
    }

    for (int counter = 0; counter < LENGTH; counter++)
    {
        int number = matrix[counter][columnIndex]->getValue();

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
            int number = matrix[startX][startY]->getValue();

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
Matrix::foreach(std::function<void(Field *)> func)
{
    for (int i = 0; i < LENGTH; i++)
    {
        for (int j = 0; j < LENGTH; j++)
        {
            func(this->matrix[i][j].get());
        }
    }
}

bool
Matrix::solved()
{
    return this->total >= TOTAL_FIELDS;
}
