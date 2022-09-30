#ifndef _RANDOMSINGLETON_H_
#define _RANDOMSINGLETON_H_

#include <boost/thread.hpp>
#include <boost/random.hpp>

#include "DefsNTools.h"

class RandomSingleton
{
    public:
        static boost::mt19937& Instance();

    private:
        static boost::thread_specific_ptr<boost::mt19937> rng;
        static boost::mutex atomic;
        static bool rand_initialized;

        static bool randInitializer();
};

#endif

