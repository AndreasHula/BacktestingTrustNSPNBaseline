#ifndef _MULTIWORKER_H_
#define _MULTIWORKER_H_

#include <vector>
#include <deque>

#include <boost/thread.hpp>
#include <boost/function.hpp>
#include <boost/utility.hpp>

template <class T>
class ThreadedQueue : boost::noncopyable
{
    public:
        void queue(const T& item);
        T dequeue();

        const size_t size() const;

    private:
        boost::condition_variable cond;
        mutable boost::mutex mut;
        std::deque<T> items;
};

class MultiWorker : boost::noncopyable
{
    public:
        typedef boost::function<void ()> Job;

        MultiWorker();
        ~MultiWorker();

        void addJob(const Job& job);
        void waitUntilDone() const;
		
		static MultiWorker& instance()
		{
			static MultiWorker worker;
			return worker;
		}
        
    private:
        typedef boost::shared_ptr<Job> JobPtr;
        typedef boost::shared_ptr<boost::thread> ThreadPtr;
        typedef std::vector<ThreadPtr> ThreadPtrVec;

        void threadwork();

        ThreadedQueue<JobPtr> jobs;
        mutable boost::mutex waitlock; //locked while jobcount > 0
        size_t jobcount; //this is not the same as jobs.size()! it only drops to zero after the last job has been executed and not just dequeued
        boost::mutex atomic; //protects jobcount
        ThreadPtrVec threads;
};

#endif

