#include <iostream>
#include <thread>
#include "timer.h"

using namespace std;

int main() {
    Timer t;
    this_thread::sleep_for(std::chrono::seconds(2));
    t.time();
    return 0;
}