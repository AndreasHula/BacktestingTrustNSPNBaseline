#include <boost/bind.hpp>

#include "DefsNTools.h"

#include "MultiWorker.h"

const unsigned int cpus = 5;

using namespace std;

template <class T>
void ThreadedQueue<T>::queue(const T& item)
{
    {
        boost::unique_lock<boost::mutex> lock(mut);
        items.push_back(item);
    }
    cond.notify_one();
}

template <class T>
T ThreadedQueue<T>::dequeue()
{
    boost::unique_lock<boost::mutex> lock(mut);
    while(!items.size())
        cond.wait(lock);

    T item = items.front();
    items.pop_front();
    return item;
}

template <class T>
const size_t ThreadedQueue<T>::size() const
{
    boost::unique_lock<boost::mutex> lock(mut);
    return items.size();
}

MultiWorker::MultiWorker() : jobcount(0), threads()
{
    while(threads.size() != cpus)
        threads.push_back(ThreadPtr(new boost::thread(boost::bind(&MultiWorker::threadwork,this))));
}

MultiWorker::~MultiWorker()
{
    //queue a null pointer as a stop job for every thread
    for(ThreadPtrVec::iterator it = threads.begin(); it != threads.end(); ++it)
        jobs.queue(boost::shared_ptr<Job>());

    //wait until all threads have joined
    for(ThreadPtrVec::iterator it = threads.begin(); it != threads.end(); ++it)
        (*it)->join();
}

void MultiWorker::addJob(const Job& job)
{
    JobPtr jobptr(new Job(job));
    boost::unique_lock<boost::mutex> lock(atomic);
    waitlock.try_lock();
    ++jobcount;
    jobs.queue(jobptr);
}

void MultiWorker::threadwork()
{
    while(JobPtr job = jobs.dequeue())
    {
        (*job)();
        boost::unique_lock<boost::mutex> lock(atomic);
        if(!(--jobcount))
            waitlock.unlock();
    }
}

void MultiWorker::waitUntilDone() const
{
    boost::unique_lock<boost::mutex> lock(waitlock);
}

