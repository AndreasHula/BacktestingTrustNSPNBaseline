#include <ctime>

#include "RandomSingleton.h"

boost::mt19937& RandomSingleton::Instance()
{
    boost::unique_lock<boost::mutex> lock(atomic);
    if(!rng.get())
        rng.reset(new boost::mt19937(rand()));
    return *rng.get();
}

bool RandomSingleton::randInitializer()
{
    srand(std::time(0));
    return true;
}

boost::thread_specific_ptr<boost::mt19937> RandomSingleton::rng;
boost::mutex RandomSingleton::atomic;
bool RandomSingleton::rand_initialized = randInitializer();
