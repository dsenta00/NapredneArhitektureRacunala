#include <thread>
#include "SudokuCpp.h"

SudokuCpp::SudokuCpp(const char initialMatrix[LENGTH][LENGTH]) : AbstractSudoku(initialMatrix)
{
}

void
SudokuCpp::solveUsing2Threads()
{
    do
    {
        std::thread t1(&SudokuCpp::solveInRange, this, 1, 4);
        std::thread t2(&SudokuCpp::solveInRange, this, 5, 9);

        t1.join();
        t2.join();
    } while (!this->matrix->solved());
}


void
SudokuCpp::solveUsing4Threads()
{
    do
    {
        std::thread t1(&SudokuCpp::solveInRange, this, 1, 3);
        std::thread t2(&SudokuCpp::solveInRange, this, 4, 5);
        std::thread t3(&SudokuCpp::solveInRange, this, 6, 7);
        std::thread t4(&SudokuCpp::solveInRange, this, 8, 9);

        t1.join();
        t2.join();
        t3.join();
        t4.join();
    } while (!this->matrix->solved());
}

void
SudokuCpp::solveUsing8Threads()
{
    do
    {
        std::thread t1(&SudokuCpp::solveInRange, this, 1, 2);
        std::thread t2(&SudokuCpp::solveInRange, this, 3, 3);
        std::thread t3(&SudokuCpp::solveInRange, this, 4, 4);
        std::thread t4(&SudokuCpp::solveInRange, this, 5, 5);
        std::thread t5(&SudokuCpp::solveInRange, this, 6, 6);
        std::thread t6(&SudokuCpp::solveInRange, this, 7, 7);
        std::thread t7(&SudokuCpp::solveInRange, this, 8, 8);
        std::thread t8(&SudokuCpp::solveInRange, this, 9, 9);

        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
        t8.join();
    } while (!this->matrix->solved());
}

void
SudokuCpp::solveUsing12Threads()
{
    do
    {
        std::thread t1(&SudokuCpp::solveInRange, this, 1, 1);
        std::thread t2(&SudokuCpp::solveInRange, this, 2, 2);
        std::thread t3(&SudokuCpp::solveInRange, this, 3, 3);
        std::thread t4(&SudokuCpp::solveInRange, this, 4, 4);
        std::thread t5(&SudokuCpp::solveInRange, this, 5, 5);
        std::thread t6(&SudokuCpp::solveInRange, this, 6, 6);
        std::thread t7(&SudokuCpp::solveInRange, this, 7, 7);
        std::thread t8(&SudokuCpp::solveInRange, this, 8, 9);
        std::thread t9(&SudokuCpp::solveInRange, this, 1, 1);
        std::thread t10(&SudokuCpp::solveInRange, this, 2, 2);
        std::thread t11(&SudokuCpp::solveInRange, this, 3, 3);
        std::thread t12(&SudokuCpp::solveInRange, this, 4, 4);

        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
        t8.join();
        t9.join();
        t10.join();
        t11.join();
        t12.join();
    } while (!this->matrix->solved());
}