#include "SudokuCpp.h"
#include "Timer.h"
#include "SudokuC.h"
#include "SudokuCuda.cuh"

const char matrix[LENGTH][LENGTH] = {
    0, 6, 0, 3, 0, 0, 8, 0, 4,
    5, 3, 7, 0, 9, 0, 0, 0, 0,
    0, 4, 0, 0, 0, 6, 3, 0, 7,

    0, 9, 0, 0, 5, 1, 2, 3, 8,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    7, 1, 3, 6, 2, 0, 0, 4, 0,

    3, 0, 6, 4, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 6, 0, 5, 2, 3,
    1, 0, 2, 0, 0, 9, 0, 8, 0,
};

template<typename T>
void solve(T sudoku)
{
    double t1 = 0;
    double t2 = 0;
    Timer timer;

    t1 = timer.elapsed();
    sudoku.solve();
    t2 = timer.elapsed();

    std::cout << "\tTime to solve with solve() function: " << t2 - t1 << std::endl << std::endl;
}


template<typename T>
void solveUsing2Threads(T sudoku)
{
    double t1 = 0;
    double t2 = 0;
    Timer timer;

    t1 = timer.elapsed();
    sudoku.solveUsing2Threads();
    t2 = timer.elapsed();

    std::cout << "\tTime to solve with solveUsing2Threads() function: " << t2 - t1 << std::endl;
}

template<typename T>
void solveUsing4Threads(T sudoku)
{
    double t1 = 0;
    double t2 = 0;
    Timer timer;

    t1 = timer.elapsed();
    sudoku.solveUsing4Threads();
    t2 = timer.elapsed();

    std::cout << "\tTime to solve with solveUsing4Threads() function: " << t2 - t1 << std::endl;
}

template<typename T>
void solveUsing8Threads(T sudoku)
{
    double t1 = 0;
    double t2 = 0;
    Timer timer;

    t1 = timer.elapsed();
    sudoku.solveUsing8Threads();
    t2 = timer.elapsed();

    std::cout << "\tTime to solve with solveUsing8Threads() function: " << t2 - t1 << std::endl;
}

template<typename T>
void solveUsing12Threads(T sudoku)
{
    double t1 = 0;
    double t2 = 0;
    Timer timer;

    t1 = timer.elapsed();
    sudoku.solveUsing12Threads();
    t2 = timer.elapsed();

    std::cout << "\tTime to solve with solveUsing12Threads() function: " << t2 - t1 << std::endl << std::endl;
}

int main()
{
    std::cout << " -- without any concurrent thread --" << std::endl << std::endl;

    solve(SudokuC(matrix));

    std::cout << " --        pthread library        --" << std::endl << std::endl;

    solveUsing2Threads(SudokuC(matrix));
    solveUsing4Threads(SudokuC(matrix));
    solveUsing8Threads(SudokuC(matrix));
    solveUsing12Threads(SudokuC(matrix));

    std::cout << " --      std::thread library      --" << std::endl << std::endl;

    solveUsing2Threads(SudokuCpp(matrix));
    solveUsing4Threads(SudokuCpp(matrix));
    solveUsing8Threads(SudokuCpp(matrix));
    solveUsing12Threads(SudokuCpp(matrix));

    std::cout << " --         using CUDA            --" << std::endl << std::endl;
    int count = 0;
    cudaGetDeviceCount(&count);

    if (count < 1)
    {
        std::cout << "\tNOT SUPPORTED!" << std::endl << std::endl;

        return EXIT_SUCCESS;
    }

    SudokuCuda *sudokuCuda = nullptr;
    cudaMalloc((void **)&sudokuCuda, sizeof(SudokuCuda));
    *sudokuCuda = SudokuCuda(matrix);

    solve(*sudokuCuda);

    cudaFree(sudokuCuda);

    return EXIT_SUCCESS;
}