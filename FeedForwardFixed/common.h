#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <map>

class MyDebug {
public :
    MyDebug(){}
    MyDebug(const MyDebug &){}
    ~MyDebug(){
        std::cout << oss.str() << std::endl;
    }

    template<typename T>
    inline MyDebug & operator << ( const T & rhs) {
        oss << rhs ;
        oss << ' ' ;
        return *this ;
    }
private :
    std::ostringstream oss ;
};

inline MyDebug debug () {
    return MyDebug ();
}

#endif  // COMMON_H
