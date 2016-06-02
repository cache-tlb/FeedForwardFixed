#ifndef MATTFIXED
#define MATTFIXED

#include "FixedPoint.h"
#include <cstdlib>
#include <vector>

template <typename _Tp>
class Mat_ {
public:
    typedef _Tp FixedType;
    Mat_ () : rows_(0), cols_(0) {}

    ~Mat_() {}
    
    Mat_ (int rows, int cols) : rows_(rows), cols_(cols) {
        data_.resize(rows*cols);
    }

    Mat_ (const Mat_ & m): rows_(m.rows_), cols_(m.cols_) {
        data_ = m.data_;
    }

    Mat_& operator = (const Mat_& rhs) {
        if (this != &rhs) {
            data_ = rhs.data_;
        }
        return *this;
    }

    Mat_& operator += (const Mat_& rhs) {
        for (int i = 0; i < rows_ * cols_; i++) {
            data_[i] += rhs.data_[i];
        }
        return *this;
    }

    void allocate(int rows, int cols) {
        if (rows_*cols_ == rows*cols) return;
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows*cols);
    }

    void setTo(const FixedType &v) {
        for (int i = 0; i < rows_*cols_; i++) {
            data_[i] = v;
        }
    }

    FixedType at(int i, int j) const {
        return data_[i*cols_ + j];
    }
    FixedType& at(int i, int j) {
        return data_[i*cols_ + j];
    }

    FixedType at(int i) const {
        return data_[i];
    }
    FixedType& at(int i) {
        return data_[i];
    }

public:
    int rows_, cols_;
    std::vector<FixedType> data_;
};

typedef Mat_<FixedPoint> Mat;

template<typename FixedType>
void Conv2D(const Mat_<FixedType> &src, Mat_<FixedType> &dst, const Mat_<FixedType> &kernel) {
    int dst_rows = src.rows_ - kernel.rows_ + 1, dst_cols = src.cols_ - kernel.cols_ + 1;
    int kernel_rows = kernel.rows_, kernel_cols = kernel.cols_;
    dst.allocate(dst_rows, dst_cols);
    dst.setTo(FixedType(0));
    for (int y = 0; y < dst_rows; y++)  for (int x = 0; x < dst_cols; x++) {
        dst.at(y, x) = FixedType(0);
        for (int i = 0; i < kernel_rows; i++) for (int j = 0; j < kernel_cols; j++) {
            dst.at(y, x) += (src.at(y + i, x + j) * kernel.at(i, j));
        }
    }
}

#endif // MATTFIXED