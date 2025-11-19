#include <iostream>
#include <thread>
#include "util/timer.h"
#include "util/image.h"

using namespace std;

int main() {
    Timer t;
    RGB888 img = RGB888{a,10,20};
    cout << img.size << endl;
    t.time();
    return 0;
}