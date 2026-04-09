#ifndef CUDARF_PROFILE_HPP
#define CUDARF_PROFILE_HPP

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

namespace cudarf
{
namespace profiling
{

static constexpr unsigned int EVENT_HISTORY_SIZE = 32;

class Interval
{
public:
    Interval(const std::string &name, cudaStream_t cuStream = 0);

    Interval(Interval &&other):
        name(std::move(other.name)),
        start(other.start),
        stop(other.stop)
    {
        other.start = 0;
        other.stop = 0;
    }

    Interval(const Interval &ev) = delete;
    Interval & operator=(const Interval &ev) = delete;
    Interval() = delete;
    ~Interval();

    void close();
    float get_duration() const;

    const std::string & get_name() const {return name;}

private:
    const std::string name;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaStream_t cuStream = 0;
};

template <typename TType, unsigned int TSizeMax>
struct Ringbuffer {
    Ringbuffer(): first(0), size(0)
    {
        std::fill(std::begin(values), std::end(values), std::numeric_limits<float>::lowest());
    }

    void add(TType value)
    {
        unsigned int dst;
        if (size == TSizeMax) {
            dst = first;
            first = (first + 1) % TSizeMax;
        } else {
            dst = (first + size) % TSizeMax;
            size++;
        }

        values[dst] = value;
    }

    TType mean()
    {
        if (size == 0) {return 0;}
        TType S = 0;

        for (unsigned int i = first, j = 0; j < size; i = (i + 1) % TSizeMax, j++) {
            S += values[i];
        }

        return S / size;
    }

    TType stdev(TType mean)
    {
        TType S = 0;

        for (unsigned int i = first, j = 0; j < size; i = (i + 1) % TSizeMax, j++) {
            S += (mean - values[i]) * (mean - values[i]);
        }

        return std::sqrt(S / size);
    }

    std::array<TType, TSizeMax> values;
    unsigned int first;
    unsigned int size;
};

class Events
{
public:
    Events(const std::string &name): name(name) {
        eventQueue.reserve(32);
        events.reserve(32);
    }

    Events(const Events &ev) = delete;
    Events & operator=(const Events &ev) = delete;

    int start_interval(const std::string &name, cudaStream_t cuStream);
    void stop_interval(int handle);
    void show(unsigned int oft = 0);
    void clear();
    std::shared_ptr<Events> add_child(const std::string &name);

private:
    std::string name;
    std::vector<std::unique_ptr<Interval>> events;
    std::unordered_map<std::string, Ringbuffer<float, EVENT_HISTORY_SIZE>> history;
    std::vector<int> eventQueue;
    std::vector<std::shared_ptr<Events>> children;
};

} // namespace profiling
} // namespace cudarf

#endif // CUDARF_PROFILE_HPP
