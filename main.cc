#include <iostream>
#include <thread>
#include "util/timer.h"
#include "util/image.h"
#include "util/display.h"

using namespace std;

int main() {
    Timer t;
    char* a;
    RGB888 img = RGB888{a,10,20};
    display();
    cout << img.size << endl;
    t.time();
    return 0;
}