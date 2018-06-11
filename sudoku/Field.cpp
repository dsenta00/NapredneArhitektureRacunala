#include "Field.h"


char
Field::getValue()
{
    return this->value;
}

Field *
Field::setValue(char value)
{
    this->value = value;

    return this;
}

Field *
Field::create()
{
    return new Field();
}

int
Field::getRowIndex()
{
    return this->rowIndex;
}

Field *
Field::setRowIndex(int rowIndex)
{
    this->rowIndex = rowIndex;

    return this;
}

int
Field::getColumnIndex()
{
    return this->columnIndex;
}

Field *
Field::setColumnIndex(int columnIndex)
{
    this->columnIndex = columnIndex;

    return this;
}
