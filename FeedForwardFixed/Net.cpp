#include "Net.h"
#include "common.h"

void Net::LoadJson(const std::string &path) {
    Json::Value root;
    std::ifstream fin(path);
    Json::Reader reader;
    if (reader.parse(fin, root)) {
    } else {
        printf("error in parsing json file!\n");
    }
    for (int i = 0; i < root.size(); i++) {
        Layer *layer = NULL;
        Json::Value &value = root[i];
        if (value["type"].asString() == "CONVOLUTION") {
            layer = ConvLayerFixed::FromJson(value);
        } else if (value["type"].asString() == "INNER_PRODUCT") {
            layer = InnerProductLayer::FromJson(value);
        } else if (value["type"].asString() == "POOLING") {
            layer = MaxPoolingLayer::FromJson(value);
        } else if (value["type"].asString() == "RELU") {
            layer = ReLuLayer::FromJson(value);
        } else if (value["type"].asString() == "DATA") {
            layer = DataLayer::FromJson(value);
        } else {
            debug() << value["name"].asString() << "of type" << value["type"].asString() << "todo.";
        }
        if (layer) layers.push_back(layer);
    }
}

void Net::FeedForward(const std::vector<Mat> &src, std::vector<Mat> &dst) {
    std::vector<Mat> temp_src, temp_dst;
    layers[0]->FeedForward(src, temp_src);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->FeedForward(temp_src, temp_dst);
        // debug() << "after layer" << i << layers[i]->layer_name() << temp_src[0].rows_ << temp_src[0].cols_ << temp_dst[0].rows_ << temp_dst[0].cols_;
        temp_src.swap(temp_dst);
        temp_dst.clear();
    }
    dst.swap(temp_src);
}
