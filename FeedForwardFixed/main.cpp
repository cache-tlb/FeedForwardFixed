#include "FixedPoint.h"
#include "FixedPointMath.h"
#include "Net.h"
#include <opencv2/opencv.hpp>
#include "common.h"

void Im2Mat(const cv::Mat &im, std::vector<Mat> &data) {
    data.clear();
    data.resize(3);
    int rows = im.rows, cols = im.cols;
    for (int c = 0; c < 3; c++) {
        data[c].allocate(rows, cols);
        for (int p = 0; p < rows*cols; p++) {
            data[c].at(p) = FixedPoint::fromFloat(im.at<cv::Vec3b>(p)[c]);
        }
    }
}

int main() {
    init();

    Net net;
    net.LoadJson("lenet_iter_5000.json");

    cv::Mat im = cv::imread("images/mnist_9.png");
    cv::imshow("im", im);
    cv::waitKey();
    std::vector<Mat> data, ret;
    Im2Mat(im, data);
    net.FeedForward(data, ret);
    int arg_max = -1;
    float max_arg = -1e10;
    for (int i = 0; i < ret.size(); i++) {
        float score = ret[i].at(0).toFloat();
        debug() << i << score;
        if (score > max_arg) {
            max_arg = score;
            arg_max = i;
        }
    }
    debug() << "class:" << arg_max;
    return 0;
}