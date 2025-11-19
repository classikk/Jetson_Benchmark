#include <iostream>
#include <chrono>
using namespace std;

static inline void timer(std::chrono::time_point<std::chrono::steady_clock> &start){
    auto time = std::chrono::steady_clock::now();
    cout << "[" << (start-time).count() << "s]"<< endl;
}

int main() {
    std::chrono::time_point<std::chrono::steady_clock> start_time  = std::chrono::steady_clock::now();
    timer(start_time);
    return 0;
}