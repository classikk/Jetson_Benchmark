#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
using namespace std;

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> split = std::chrono::steady_clock::now();
    
    inline double seconds(){
        auto end = std::chrono::steady_clock::now();
        return (double)(end-start).count()/1000000000;
    }

    inline double time_split(){
        auto end = std::chrono::steady_clock::now();
        double time = (double)(end-split).count()/1000000000;
        split = end;
        return time;
    }

    inline void time_stamp(const char* msg){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]\t[dif " << time_split() << "s]\t " << msg << endl;
    }
    
    inline void time_stamp(){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]\t[dif " << time_split() << "s]" << endl;
    }
};

#endif