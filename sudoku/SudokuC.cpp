#include "SudokuC.h"
#if defined(WIN32)
#include <windows.h>
#endif

typedef struct {
    AbstractSudoku *sudoku;
    int from;
    int to;
} RangeParameters;

static void *solveInRangeC(void *rangeParameters)
{
    AbstractSudoku *sudoku = ((RangeParameters *) rangeParameters)->sudoku;
    int from = ((RangeParameters *) rangeParameters)->from;
    int to = ((RangeParameters *) rangeParameters)->to;

    sudoku->solveInRange(from, to);

    return NULL;
}

SudokuC::SudokuC(const char initialMatrix[LENGTH][LENGTH]) : AbstractSudoku(initialMatrix)
{}

void
SudokuC::solveUsing2Threads()
{
    RangeParameters r1 = { this, 1, 4 };
    RangeParameters r2 = { this, 5, 9 };

    do
    {
#if defined(WIN32) || defined(WIN64)
        HANDLE hThread1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r1, 0, NULL);
        HANDLE hThread2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r2, 0, NULL);

        WaitForSingleObject(hThread1, INFINITE);
        WaitForSingleObject(hThread2, INFINITE);
#else
        pthread_t t1;
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
#endif
    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing4Threads()
{
    RangeParameters r1 = { this, 1, 3 };
    RangeParameters r2 = { this, 4, 5 };
    RangeParameters r3 = { this, 6, 7 };
    RangeParameters r4 = { this, 8, 9 };

    do
    {
#if defined(WIN32) || defined(WIN64)
        HANDLE hThread1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r1, 0, NULL);
        HANDLE hThread2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r2, 0, NULL);
        HANDLE hThread3 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r3, 0, NULL);
        HANDLE hThread4 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r4, 0, NULL);

        WaitForSingleObject(hThread1, INFINITE);
        WaitForSingleObject(hThread2, INFINITE);
        WaitForSingleObject(hThread3, INFINITE);
        WaitForSingleObject(hThread4, INFINITE);
#else
        pthread_t t1;
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);
        pthread_join(t4, NULL);
#endif
    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing8Threads()
{
    RangeParameters r1 = { this, 1, 2 };
    RangeParameters r2 = { this, 3, 3 };
    RangeParameters r3 = { this, 4, 4 };
    RangeParameters r4 = { this, 5, 5 };
    RangeParameters r5 = { this, 6, 6 };
    RangeParameters r6 = { this, 7, 7 };
    RangeParameters r7 = { this, 8, 8 };
    RangeParameters r8 = { this, 9, 9 };

    do
    {
#if defined(WIN32) || defined(WIN64)
        HANDLE hThread1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r1, 0, NULL);
        HANDLE hThread2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r2, 0, NULL);
        HANDLE hThread3 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r3, 0, NULL);
        HANDLE hThread4 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r4, 0, NULL);
        HANDLE hThread5 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r5, 0, NULL);
        HANDLE hThread6 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r6, 0, NULL);
        HANDLE hThread7 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r7, 0, NULL);
        HANDLE hThread8 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r8, 0, NULL);

        WaitForSingleObject(hThread1, INFINITE);
        WaitForSingleObject(hThread2, INFINITE);
        WaitForSingleObject(hThread3, INFINITE);
        WaitForSingleObject(hThread4, INFINITE);
        WaitForSingleObject(hThread5, INFINITE);
        WaitForSingleObject(hThread6, INFINITE);
        WaitForSingleObject(hThread7, INFINITE);
        WaitForSingleObject(hThread8, INFINITE);
#else
        pthread_t t1;
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_t t5;
        pthread_create(&t5, NULL, solveInRangeC, &r5);

        pthread_t t6;
        pthread_create(&t6, NULL, solveInRangeC, &r6);

        pthread_t t7;
        pthread_create(&t7, NULL, solveInRangeC, &r7);

        pthread_t t8;
        pthread_create(&t8, NULL, solveInRangeC, &r8);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);
        pthread_join(t4, NULL);
        pthread_join(t5, NULL);
        pthread_join(t6, NULL);
        pthread_join(t7, NULL);
        pthread_join(t8, NULL);
#endif
    } while (!this->matrix->solved());
}

void
SudokuC::solveUsing12Threads()
{
    RangeParameters r1 = { this, 1, 2 };
    RangeParameters r2 = { this, 3, 3 };
    RangeParameters r3 = { this, 4, 4 };
    RangeParameters r4 = { this, 5, 5 };
    RangeParameters r5 = { this, 6, 6 };
    RangeParameters r6 = { this, 7, 7 };
    RangeParameters r7 = { this, 8, 8 };
    RangeParameters r8 = { this, 9, 9 };
    RangeParameters r9 = { this, 1, 2 };
    RangeParameters r10 = { this, 3, 3 };
    RangeParameters r11 = { this, 4, 4 };
    RangeParameters r12 = { this, 5, 5 };

    do
    {
#if defined(WIN32) || defined(WIN64)
        HANDLE hThread1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r1, 0, NULL);
        HANDLE hThread2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r2, 0, NULL);
        HANDLE hThread3 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r3, 0, NULL);
        HANDLE hThread4 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r4, 0, NULL);
        HANDLE hThread5 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r5, 0, NULL);
        HANDLE hThread6 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r6, 0, NULL);
        HANDLE hThread7 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r7, 0, NULL);
        HANDLE hThread8 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r8, 0, NULL);
        HANDLE hThread9 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r5, 0, NULL);
        HANDLE hThread10 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r6, 0, NULL);
        HANDLE hThread11 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r7, 0, NULL);
        HANDLE hThread12 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)solveInRangeC, &r8, 0, NULL);

        WaitForSingleObject(hThread1, INFINITE);
        WaitForSingleObject(hThread2, INFINITE);
        WaitForSingleObject(hThread3, INFINITE);
        WaitForSingleObject(hThread4, INFINITE);
        WaitForSingleObject(hThread5, INFINITE);
        WaitForSingleObject(hThread6, INFINITE);
        WaitForSingleObject(hThread7, INFINITE);
        WaitForSingleObject(hThread8, INFINITE);
        WaitForSingleObject(hThread9, INFINITE);
        WaitForSingleObject(hThread10, INFINITE);
        WaitForSingleObject(hThread11, INFINITE);
        WaitForSingleObject(hThread12, INFINITE);
#else
        pthread_t t1;
        pthread_create(&t1, NULL, solveInRangeC, &r1);

        pthread_t t2;
        pthread_create(&t2, NULL, solveInRangeC, &r2);

        pthread_t t3;
        pthread_create(&t3, NULL, solveInRangeC, &r3);

        pthread_t t4;
        pthread_create(&t4, NULL, solveInRangeC, &r4);

        pthread_t t5;
        pthread_create(&t5, NULL, solveInRangeC, &r5);

        pthread_t t6;
        pthread_create(&t6, NULL, solveInRangeC, &r6);

        pthread_t t7;
        pthread_create(&t7, NULL, solveInRangeC, &r7);

        pthread_t t8;
        pthread_create(&t8, NULL, solveInRangeC, &r8);

        pthread_t t9;
        pthread_create(&t9, NULL, solveInRangeC, &r9);

        pthread_t t10;
        pthread_create(&t10, NULL, solveInRangeC, &r10);

        pthread_t t11;
        pthread_create(&t11, NULL, solveInRangeC, &r11);

        pthread_t t12;
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
#endif
    } while (!this->matrix->solved());
}
