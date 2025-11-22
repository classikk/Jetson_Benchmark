#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> split = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> benchMark = std::chrono::steady_clock::now();
    std::vector<double> arr;
    int steps = -1;
    int n = 0;
    inline double seconds(){
        auto end = std::chrono::steady_clock::now();
        return (double)(end-start).count()/1000000000;
    }

    inline double time_split(std::chrono::time_point<std::chrono::steady_clock> &split){
        auto end = std::chrono::steady_clock::now();
        double time = (double)(end-split).count()/1000000000;
        split = end;
        return time;
    }

    inline void time_stamp(const char* msg){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]\t[dif " << time_split(split) << "s]\t " << msg << endl;
    }
    
    inline void time_stamp(){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]\t[dif " << time_split(split) << "s]" << endl;
    }

    inline void benchmark(int i){
        if (steps < i) {
            arr.push_back(0.0);
            steps += 1;
        } 
        if (i == 0) n+=1;
        arr[i] += time_split(benchMark);
    }
    inline void show_benchmark(){
        for (int i = 0; i <= steps; i++){
            cout << "[" << arr[i]/n << "s]\taverage on [" << i << "]" << endl;
        }
    }
};

#endif