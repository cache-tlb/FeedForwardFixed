#include "LayersFixed.h"
#include <cassert>

DataLayer *DataLayer::FromJson(Json::Value &value) {
    float scale = value["scale"].asFloat();
    DataLayer *ret = new DataLayer(scale);
    ret->layer_name_ = value["name"].asString();
    return ret;
}

void DataLayer::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    int n = src.size();
    int rows = src[0].rows_, cols = src[0].cols_;
    dst.clear();
    dst.resize(n);
    for (int k = 0; k < n; k++) {
        dst[k].allocate(rows, cols);
        for (int i = 0; i < rows*cols; i++) {
            dst[k].at(i) = src[k].at(i) * scale_;
        }
    }
}

void PaddingLayer::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    int rows = src[0].rows_, cols = src[0].cols_;
    dst.clear();
    dst.resize(src.size());
    for (int k = 0; k < src.size(); k++) {
        dst[k].allocate(rows + padding_size_*2, cols + padding_size_*2);
        dst[k].setTo(FixedPoint(0));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dst[k].at(i + padding_size_, j + padding_size_) = src[k].at(i, j);
            }
        }
    }
}

MaxPoolingLayer *MaxPoolingLayer::FromJson(Json::Value &value) {
    assert(value["pool"].asString() == "MAX");
    int stride = value["stride"].asInt();
    int size = value["size"].asInt();
    bool full_bound = true;
    MaxPoolingLayer *ret = new MaxPoolingLayer(size, size, stride, stride, full_bound);
    ret->layer_name_ = value["name"].asString();
    return ret;
}

void MaxPoolingLayer::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    int rows = src[0].rows_, cols = src[0].cols_;
    dst.clear();
    dst.resize(src.size());
    for (int k = 0; k < src.size(); k++) {
        if (full_boundary_){
            dst[k].allocate((rows-1)/stride_y_+1, (cols-1)/stride_x_+1);
        }
        else{
            dst[k].allocate(rows/stride_y_, cols/stride_x_);
        }
        dst[k].setTo(FixedPoint(0));
        int dst_rows = dst[k].rows_, dst_cols = dst[k].cols_;
        for (int ii = 0; ii < dst_rows; ii++) for (int jj = 0; jj < dst_cols; jj++) {
            FixedPoint maxv = src[k].at(0,0);
            for (int iii = ii*stride_y_; iii < ii*stride_y_+size_y_ && iii < rows; iii++) {
                for (int jjj = jj*stride_x_; jjj < jj*stride_x_+size_x_ && jjj < cols; jjj++) {
                    maxv = std::max(maxv, src[k].at(iii,jjj));
                }
            }
            dst[k].at(ii,jj) = maxv;
        }
    }
}

static void Json2Vec(Json::Value &value, std::vector<float> &res) {
    int n = value.size();
    res.resize(n);
    for (int i = 0; i < n; i++) {
        res[i] = value[i].asFloat();
    }
}

ConvLayerFixed *ConvLayerFixed::FromJson(Json::Value &value) {
    int rows = value["height"].asInt();
    int cols = value["width"].asInt();
    int channels = value["channels"].asInt();
    int num = value["num"].asInt();
    int pad = value["pad"].asInt();
    ConvLayerFixed * ret = new ConvLayerFixed(num, channels, rows, cols, pad);
    ret->layer_name_ = value["name"].asString();
    std::vector<float> weight, bias;
    Json2Vec(value["w"], weight);
    Json2Vec(value["b"], bias);
    ret->LoadWight(weight);
    ret->LoadBias(bias);
    return ret;
}

void ConvLayerFixed::LoadWight(const std::vector<float> &weight) {
    assert(weight.size() == filter_dim_[0]*filter_dim_[1]*filter_dim_[2]*filter_dim_[3]);
    int k = 0;
    for (int n = 0; n < filter_dim_[0]; n++) {
        for (int c = 0; c < filter_dim_[1]; c++) {
            for (int i = 0; i < filter_dim_[2]; i++) {
                for (int j = 0; j < filter_dim_[3]; j++) {
                    weight_[n][c].at(i,j) = FixedPoint::fromFloat(weight[k++]);
                }
            }
        }
    }
}

void ConvLayerFixed::LoadBias(const std::vector<float> &bias) {
    assert(bias.size() == filter_dim_[0]);
    for (int n = 0; n < filter_dim_[0]; n++) {
        bias_[n] = FixedPoint::fromFloat(bias[n]);
    }
}

void ConvLayerFixed::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    std::vector<Mat> padded;
    pad_layer.FeedForward(src, padded);
    int rows = padded[0].rows_, cols = padded[0].cols_;
    int dst_rows = rows - weight_[0][0].rows_ + 1, dst_cols = cols - weight_[0][0].cols_ + 1;
    dst.clear();
    dst.resize(filter_dim_[0]);
    Mat tmp(rows, cols);
    for (int i = 0; i < filter_dim_[0]; i++) {
        dst[i].allocate(dst_rows, dst_cols);
        dst[i].setTo(bias_[i]);
        for (int j = 0; j < filter_dim_[1]; j++) {
            Conv2D(padded[j], tmp, weight_[i][j]);
            dst[i] += tmp;
        }
    }
}

void InnerProductLayer::LoadWight(const std::vector<float> &weight) {
    assert(weight == weight_.rows_*weight_.cols_);
    for (int i = 0; i < weight.size(); i++) {
        weight_.at(i) = FixedPoint::fromFloat(weight[i]);
    }
}

void InnerProductLayer::LoadBias(const std::vector<float> &bias) {
    assert(bias.size() == bias_.size());
    for (int i = 0; i < bias_.size(); i++) {
        bias_[i] = FixedPoint::fromFloat(bias[i]);
    }
}

InnerProductLayer *InnerProductLayer::FromJson(Json::Value &value) {
    int channels = value["channels"].asInt();
    int num = value["num"].asInt();
    int rows = value["height"].asInt();
    int cols = value["width"].asInt();
    assert(num == 1 && channels == 1);
    InnerProductLayer * ret = new InnerProductLayer(rows, cols);
    ret->layer_name_ = value["name"].asString();
    std::vector<float> weight, bias;
    Json2Vec(value["w"], weight);
    Json2Vec(value["b"], bias);
    ret->LoadWight(weight);
    ret->LoadBias(bias);
    return ret;
}

void InnerProductLayer::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    int src_row = src[0].rows_, src_col = src[0].cols_, src_channel = src.size();
    assert(src_row*src_col*src_channel == weight_.cols_);
    dst.clear();
    dst.resize(weight_.rows_);
    for (int k = 0; k < weight_.rows_; k++) {
        dst[k].allocate(1,1);
        FixedPoint s = bias_[k];
        int index = 0;
        for (int c = 0; c < src_channel; c++) {
            for (int i = 0; i < src_row; i++) {
                for (int j = 0; j < src_col; j++) {
                    s += (src[c].at(i,j)*weight_.at(k, index++));
                }
            }
        }
        dst[k].at(0) = s;
    }
}

ReLuLayer * ReLuLayer::FromJson(Json::Value &value) {
    ReLuLayer *ret = new ReLuLayer;
    ret->layer_name_ = value["name"].asString();
    return ret;
}

void ReLuLayer::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    dst.clear();
    dst.resize(src.size());
    FixedPoint z = FixedPoint::fromFloat(0);
    int src_row = src[0].rows_, src_col = src[0].cols_;
    for (int i = 0; i < src.size(); i++) {
        dst[i].allocate(src_row, src_col);
        for (int k = 0; k < src_row*src_col; k++) {
            dst[i].at(k) = std::max(src[i].at(k), z);
        }
    }
}