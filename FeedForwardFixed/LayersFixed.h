#ifndef LAYERSFIXED_H
#define LAYERSFIXED_H

#include "MatFixed.h"
#include "json/json.h"
#include "common.h"

class Layer {
public:
    Layer() : layer_name_("abstract_layer") {}
    virtual ~Layer(){}
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) = 0;
    virtual std::string layer_name() {
        return layer_name_;
    }
protected:
    std::string layer_name_;
};

class DataLayer : public Layer {
public:
    DataLayer(float scale = 1) {
        scale_ = FixedPoint::fromFloat(scale);
    }
    virtual ~DataLayer() {}
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
    static DataLayer *FromJson(Json::Value &value);
private:
    // todo: add scale and mean subtract
    FixedPoint scale_;
};

class PaddingLayer : public Layer {
public:
    PaddingLayer() : padding_size_(0) {}
    PaddingLayer(int size) : padding_size_(size) {}
    void SetPaddingSize(int size) { padding_size_ = size; }
    virtual ~PaddingLayer() {}
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
private:
    int padding_size_;
};

class MaxPoolingLayer : public Layer {
public:
    MaxPoolingLayer() {}
    MaxPoolingLayer(int range_x, int range_y, int stride_x, int stride_y, bool full_boundary) 
        : size_x_(range_x), size_y_(range_y), stride_x_(stride_x), stride_y_(stride_y), full_boundary_(full_boundary) {}
    virtual ~MaxPoolingLayer() {}
    void SetParameter(int range_x, int range_y, int stride_x, int stride_y, bool full_boundary) {
        size_x_ = range_x; size_y_ = range_y;
        stride_x_ = stride_x; stride_y_ = stride_y;
        full_boundary_ = full_boundary;
    }
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
    static MaxPoolingLayer *FromJson(Json::Value &value);
private:
    int size_x_, size_y_, stride_x_, stride_y_;
    bool full_boundary_;
};


class ConvLayerFixed : public Layer {
private:
    std::vector<std::vector<Mat> > weight_;
    std::vector<FixedPoint> bias_;
    int filter_dim_[4];   // num, channel, rows, cols;
    PaddingLayer pad_layer;
public:
    ConvLayerFixed(int num, int channel, int rows, int cols, int pad): pad_layer(pad) {
        filter_dim_[0] = num;
        filter_dim_[1] = channel;
        filter_dim_[2] = rows;
        filter_dim_[3] = cols;
        bias_.resize(num);
        weight_.resize(num);
        for (int i = 0; i < num; i++) {
            weight_[i].resize(channel);
            for (int j = 0; j < channel; j++) {
                weight_[i][j].allocate(rows, cols);
            }
        }
    }

    virtual ~ConvLayerFixed() {}
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
    static ConvLayerFixed *FromJson(Json::Value &value);
    void LoadWight(const std::vector<float> &weight);
    void LoadBias(const std::vector<float> &bias);
};

class InnerProductLayer : public Layer {
private:
    Mat weight_;
    std::vector<FixedPoint> bias_;

public:
    InnerProductLayer(int rows, int cols) {
        bias_.resize(rows);
        weight_.allocate(rows, cols);
    }
    virtual ~InnerProductLayer() {}
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
    static InnerProductLayer *FromJson(Json::Value &value);
    void LoadWight(const std::vector<float> &weight);
    void LoadBias(const std::vector<float> &bias);
};

class ReLuLayer : public Layer {
public:
    ReLuLayer() {}
    virtual ~ReLuLayer() {}
    static ReLuLayer *FromJson(Json::Value &value);
    virtual void FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst);
};

#endif // LAYERSFIXED_H
