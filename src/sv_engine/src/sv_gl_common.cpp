#include <ctime>
#include <cstdint>

uint64_t sv_get_timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t timestamp = ts.tv_sec * 1000000000ULL + ts.tv_nsec;

    return timestamp;
}
