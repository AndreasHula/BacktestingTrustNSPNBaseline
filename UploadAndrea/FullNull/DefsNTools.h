#ifndef _DEFSNTOOLS_H_
#define _DEFSNTOOLS_H_

#include <string>
#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/cstdint.hpp>

class Exception
{
    public:
        Exception(const std::string& _message) : message(_message) {}
        
        const std::string& getMessage() const {return message;}
        
    private:
        std::string message;
};

#endif

