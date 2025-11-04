#include <chrono>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

uint64_t cnow() {
    auto p = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(p.time_since_epoch()).count();
}

uint64_t now() {
    timeval tv;
    ::gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main() {
    auto t1 = now();
    auto t2 = cnow();
    ::usleep(15 * 1000);
    t1 = now() - t1;
    t2 = cnow() - t2;
    std::cout << t1 << " " << t2 << std::endl;
}

