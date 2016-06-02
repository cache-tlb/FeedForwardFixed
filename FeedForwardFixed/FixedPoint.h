#ifndef FIXEDPOINT_H
#define FIXEDPOINT_H

template <int F>
class FixedPoint_ {
    long long bits;
public:

    static const int SHIFT_DIGITS = F;

    FixedPoint_(): bits(0) {}

    FixedPoint_(const FixedPoint_ & that) : bits(that.bits) { }

    explicit FixedPoint_(long long bits) : bits(bits) {}

    static FixedPoint_ fromFloat(float x) {
        long long b = (long long)(x * (1 << SHIFT_DIGITS ));
        return FixedPoint_(b);
    }

    float toFloat() {
        return float(bits) / (1 << SHIFT_DIGITS);
    }

    FixedPoint_ operator + (const FixedPoint_ & that) const {
        return FixedPoint_(bits + that.bits);
    }

    FixedPoint_ operator - (const FixedPoint_ & that) const {
        return FixedPoint_(bits - that.bits);
    }

    FixedPoint_ operator * (const FixedPoint_ & that) const {
        long long res = bits * that.bits;
        return FixedPoint_(res >> SHIFT_DIGITS);
    }

    FixedPoint_ operator * (const long long x) const {
        return FixedPoint_(bits * x);
    }

    FixedPoint_& operator += (const FixedPoint_ & rhs) {
        bits += rhs.bits;
        return *this;
    }

    FixedPoint_& operator -= (const FixedPoint_ & rhs) {
        bits -= rhs.bits;
        return *this;
    }

    FixedPoint_& operator = (const FixedPoint_ &that) {
        if (&that != this) bits = that.bits;
        return *(this);
    }

    bool operator == (const FixedPoint_ &that) const {
        return bits == that.bits;
    }

    bool operator < (const FixedPoint_ &that) const {
        return bits < that.bits;
    }

    bool operator > (const FixedPoint_ &that) const {
        return bits > that.bits;
    }

    bool operator <= (const FixedPoint_ &that) const {
        return bits <= that.bits;
    }

    bool operator >= (const FixedPoint_ &that) const {
        return bits >= that.bits;
    }

    int sign() const {
        if (bits > 0) return 1;
        else if (bits < 0) return -1;
        else return 0;
    }

    FixedPoint_ rShift(int n) const {
        return FixedPoint_(bits >> n);
    }
};

typedef FixedPoint_<30> FixedPoint;


#endif // FIXEDPOINT_H
