#include <iostream>
#include <chrono>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start;
    
    static inline void time(std::chrono::time_point<std::chrono::steady_clock> &start){
        auto end = std::chrono::steady_clock::now();
        cout << "[" << (double)(end-start).count()/1000000000 << "s]"<< endl;
    }
}

Timer start_timing(){
    return Timer{std::chrono::steady_clock::now()};
}
