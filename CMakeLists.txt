cmake_minimum_required(VERSION 3.8)
project(sudoku)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp SudokuCpp.h SudokuCpp.cpp SudokuField.h SudokuField.cpp SudokuMatrix.cpp SudokuMatrix.h Timer.h SudokuC.cpp SudokuC.h Sudoku.cpp Sudoku.h)
add_executable(sudoku ${SOURCE_FILES})