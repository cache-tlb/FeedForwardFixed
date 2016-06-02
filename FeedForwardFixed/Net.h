#ifndef NET_H
#define NET_H

#include "LayersFixed.h"
#include "MatFixed.h"

class Net {
public:
    Net() {}
    ~Net() {
        for (int i = 0; i < layers.size(); i++) {
            if (layers[i]) delete layers[i];
        }
    }

    void LoadJson(const std::string &path);
    void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);

protected:
    std::vector<Layer*> layers;
};

#endif // NET_H