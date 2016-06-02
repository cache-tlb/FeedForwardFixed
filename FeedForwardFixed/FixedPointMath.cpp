#include "FixedPointMath.h"
#include <cmath>

void init () {}

/*
const static int TABLE_SIZE = 32;
static FixedPoint atanh_table[TABLE_SIZE];

#ifdef _MSC_VER
double atanh(double x) {
    return (log(1+x) - log(1-x))/2;
}
float atanh(float x) {
    return (log(1+x) - log(1-x))/2;
}
#endif

void init () {
    for (int i = 0; i < TABLE_SIZE; i++) {
        float at = atanh(1./double(1 << i));
        atanh_table[i] = FixedPoint::fromFloat(at);
    }
}

void rotate_hyperbolic(FixedPoint &x, FixedPoint &y, int k, int sign) {
    FixedPoint tmp = FixedPoint::fromFloat(sign);
    FixedPoint tan_theta = tmp.rShift(k);
    FixedPoint xk1 = x + y*tan_theta;
    FixedPoint yk1 = x*tan_theta + y;
    x = xk1;
    y = yk1;
}


float exp_fixed(float f) {
    // this function only works in range [-1,1], with low precision
    const float K = 1.205136;
    FixedPoint z = FixedPoint::fromFloat(0);   // z is theta
    FixedPoint x = FixedPoint::fromFloat(K);   // x is cosh(theta)
    FixedPoint y = FixedPoint::fromFloat(0);   // y is sinh(theta)
    FixedPoint f_fix = FixedPoint::fromFloat(f);

    int k = 1, sign = 0;
    do {
        sign = (f_fix - z).sign();
        rotate_hyperbolic(x, y, k, sign);
        z = z + atanh_table[k] * sign;
        // printf("%f\n", z.toFloat());
        k++;
    } while (sign != 0 && k < TABLE_SIZE);

    // printf("x: %f, y: %f, x+y:%f\n", x.toFloat(), y.toFloat(), (x + y).toFloat());
    // printf("float: %f\n", exp(f));
    return (x + y).toFloat();
}
*/