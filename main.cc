#include <iostream>
#include <chrono>
#include <thread>
using namespace std;

static inline void timer(std::chrono::time_point<std::chrono::steady_clock> &start){
    auto time = std::chrono::steady_clock::now();
    cout << "[" << (double)(time-start).count()/1000000000 << "s]"<< endl;
}

int main() {
    std::chrono::time_point<std::chrono::steady_clock> start_time  = std::chrono::steady_clock::now();
    
    timer(start_time);
    return 0;
}