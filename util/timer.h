#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
using namespace std;

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    
    inline void time(){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]"<< endl;
    }
};

#endif