#include <cassert>
#include <cstring>

#include <cudarf/cudarf_profile.hpp>

#include "cuda_helpers.hpp"

using namespace cudarf;
using namespace cudarf::profiling;

profiling::Interval::Interval(const std::string &name, cudaStream_t cuStream):
    name(name),
    cuStream(cuStream)
{
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));
    CUDA_CHK(cudaEventRecord(start, cuStream));
}

void profiling::Interval::close()
{
    CUDA_CHK(cudaEventRecord(stop, cuStream));
}

float profiling::Interval::get_duration() const
{
    float milliseconds;
    CUDA_CHK(cudaEventSynchronize(stop));
    CUDA_CHK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

profiling::Interval::~Interval()
{
    if (start != nullptr) {
        CUDA_CHK(cudaEventDestroy(start));
    }
    if (stop != nullptr) {
        CUDA_CHK(cudaEventDestroy(stop));
    }
}

int profiling::Events::start_interval(const std::string &name, cudaStream_t cuStream)
{
    events.push_back(std::make_unique<Interval>(name, cuStream));
    return static_cast<int>(events.size()) - 1;
}

void profiling::Events::stop_interval(int handle)
{
    assert(handle >= 0);
    assert(handle < static_cast<int>(events.size()));
    events[handle]->close();
    eventQueue.push_back(handle);
}

void profiling::Events::clear()
{
    events.clear();
    eventQueue.clear();

    for (auto &child : children) {
        child->clear();
    }
}

void profiling::Events::show(unsigned int oft)
{
    char pad[oft + 1];
    memset(pad, ' ', oft);
    pad[oft] = '\0';

    if (children.size() || eventQueue.size()) {
        printf("%stime for '%s'\n", pad, name.c_str());
        printf("%s-------------\n", pad);
    }

    for (auto &child : children) {
        child->show(oft + 4);
    }

    for (int id: eventQueue) {
        assert(id < static_cast<int>(events.size()));

        const auto &evName = events[id]->get_name();
        Ringbuffer<float, EVENT_HISTORY_SIZE> &evHist = history[evName];
        evHist.add(events[id]->get_duration());
        float mean = evHist.mean();

        printf("%s%25s: avg: %.2fms | sd: %.2fms \n",
               pad, evName.c_str(), mean, evHist.stdev(mean));
    }

    printf("\n");
}

std::shared_ptr<Events> profiling::Events::add_child(const std::string &name)
{
    children.push_back(std::make_shared<Events>(name));
    return children.back();
}
