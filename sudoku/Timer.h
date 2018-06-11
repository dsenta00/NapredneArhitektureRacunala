#ifndef SUDOKU_TIMER_H
#define SUDOKU_TIMER_H

#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

class Timer  {
public:
    Timer() : begining(Clock::now()) { }
    void reset() { begining = Clock::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<std::chrono::microseconds>
            (Clock::now() - begining).count();
    }
private:
    std::chrono::time_point<Clock> begining;
};


#endif //SUDOKU_TIMER_H
