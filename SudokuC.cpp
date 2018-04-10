#include "SudokuC.h"

typedef struct {
    Sudoku *sudoku;
    int from;
    int to;
} RangeParameters;

static void *solveInRangeC(void *rangeParameters)
{
    Sudoku *sudoku = ((RangeParameters *) rangeParameters)->sudoku;
    int from = ((RangeParameters *) rangeParameters)->from;
    int to = ((RangeParameters *) rangeParameters)->to;

    sudoku->solveInRange(from, to);

    return NULL;
}

SudokuC::SudokuC(const char initialMatrix[LENGTH][LENGTH]) : Sudoku(initialMatrix)
{}

void
SudokuC::solveUsing2Threads()
{
    do
    {
        pthread_t t1;
        RangeParameters r1 = {this, 1, 4};
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        RangeParameters r2 = {this, 5, 9};
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);

    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing4Threads()
{
    do
    {
        pthread_t t1;
        RangeParameters r1 = {this, 1, 3};
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        RangeParameters r2 = {this, 4, 5};
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        RangeParameters r3 = {this, 6, 7};
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        RangeParameters r4 = {this, 8, 9};
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);
        pthread_join(t4, NULL);

    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing8Threads()
{
    do
    {
        pthread_t t1;
        RangeParameters r1 = {this, 1, 2};
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        RangeParameters r2 = {this, 3, 3};
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        RangeParameters r3 = {this, 4, 4};
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        RangeParameters r4 = {this, 5, 5};
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_t t5;
        RangeParameters r5 = {this, 6, 6};
        pthread_create(&t5, NULL, solveInRangeC, &r5);

        pthread_t t6;
        RangeParameters r6 = {this, 7, 7};
        pthread_create(&t6, NULL, solveInRangeC, &r6);

        pthread_t t7;
        RangeParameters r7 = {this, 8, 8};
        pthread_create(&t7, NULL, solveInRangeC, &r7);

        pthread_t t8;
        RangeParameters r8 = {this, 9, 9};
        pthread_create(&t8, NULL, solveInRangeC, &r8);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);
        pthread_join(t4, NULL);
        pthread_join(t5, NULL);
        pthread_join(t6, NULL);
        pthread_join(t7, NULL);
        pthread_join(t8, NULL);

    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing12Threads()
{
    do
    {
        pthread_t t1;
        RangeParameters r1 = {this, 1, 2};
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        RangeParameters r2 = {this, 3, 3};
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        RangeParameters r3 = {this, 4, 4};
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        RangeParameters r4 = {this, 5, 5};
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_t t5;
        RangeParameters r5 = {this, 6, 6};
        pthread_create(&t5, NULL, solveInRangeC, &r5);

        pthread_t t6;
        RangeParameters r6 = {this, 7, 7};
        pthread_create(&t6, NULL, solveInRangeC, &r6);

        pthread_t t7;
        RangeParameters r7 = {this, 8, 8};
        pthread_create(&t7, NULL, solveInRangeC, &r7);

        pthread_t t8;
        RangeParameters r8 = {this, 9, 9};
        pthread_create(&t8, NULL, solveInRangeC, &r8);

        pthread_t t9;
        RangeParameters r9 = {this, 1, 2};
        pthread_create(&t9, NULL, solveInRangeC, &r9);

        pthread_t t10;
        RangeParameters r10 = {this, 3, 3};
        pthread_create(&t10, NULL, solveInRangeC, &r10);

        pthread_t t11;
        RangeParameters r11 = {this, 4, 4};
        pthread_create(&t11, NULL, solveInRangeC, &r11);

        pthread_t t12;
        RangeParameters r12 = {this, 5, 5};
        pthread_create(&t12, NULL, solveInRangeC, &r12);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);
        pthread_join(t4, NULL);
        pthread_join(t5, NULL);
        pthread_join(t6, NULL);
        pthread_join(t7, NULL);
        pthread_join(t8, NULL);
        pthread_join(t9, NULL);
        pthread_join(t10, NULL);
        pthread_join(t11, NULL);
        pthread_join(t12, NULL);

    } while (!this->matrix->solved());
}
