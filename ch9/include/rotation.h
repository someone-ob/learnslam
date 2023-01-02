#ifndef ROTATION_H
#define ROTATION_H

#include <algorithm>
#include <cmath>
#include <limits>

//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion. 

// dot and cross production 

template<typename T>
inline T DotProduct(const T x[3], const T y[3]) {//点乘
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T>
//向量叉乘
inline void CrossProduct(const T x[3], const T y[3], T result[3]) {
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}


//////////////////////////////////////////////////////////////////


// Converts from a angle anxis to quaternion : 
/*
角轴[a0,a1,a2]变四元数[q0,q1,q2,q3]：
[q1,q2,q3]=[a0,a1,a2]*sin(theta/2)/theta 其中[a0,a1,a2]/theta=[n0,n1,n2] 单位长度向量n 
这里theta=sqrt(a0*a0+a1*a1+a2*a2) 角轴模长为旋转角度
定义sin(theta/2)/theta为k
当角不为0的时候，如常计算k. q0=cos（theta/2）
当角趋近于0的时候,k趋近于0.5.q0趋近于1
*/
template<typename T>
inline void AngleAxisToQuaternion(const T *angle_axis, T *quaternion) {
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    if (theta_squared > T(std::numeric_limits<double>::epsilon())) {
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta;
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else { // in case if theta_squared is zero
        const T k(0.5);
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

template<typename T>
inline void QuaternionToAngleAxis(const T *quaternion, T *angle_axis) {
    const T &q1 = quaternion[1];
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;

    // For quaternions representing non-zero rotation, the conversion
    // is numercially stable
    if (sin_squared_theta > T(std::numeric_limits<double>::epsilon())) {
        const T sin_theta = sqrt(sin_squared_theta);
        const T &cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which
        // means that angle for the angle_axis vector which is 2 * theta
        // would be greater than pi...

        const T two_theta = T(2.0) * ((cos_theta < 0.0)
                                      ? atan2(-sin_theta, -cos_theta)
                                      : atan2(sin_theta, cos_theta));
        const T k = two_theta / sin_theta;

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in derivative since
        // the argument is zero. By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used..
        const T k(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }

}

template<typename T>
//result= -R't
//此函数要求得result，即负的角轴求出R'，再乘以pt
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {//angle_axis是负的角轴,添加负号代表旋转方向相反,即为逆; pt是平移向量; result是相机中心点
    const T theta2 = DotProduct(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        //远离零，使用罗德里格斯公式
        //如果 angle_axis 向量的范数大于零，我们要小心只计算平方根。 否则我们得到除以零。
        // result = pt costheta +(w x pt) * sintheta +w (w . pt) (1 - costheta)
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = 1.0 / theta;

        const T w[3] = {angle_axis[0] * theta_inverse,
                        angle_axis[1] * theta_inverse,
                        angle_axis[2] * theta_inverse};//使旋转向量变为单位向量

        /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                                  w[2] * pt[0] - w[0] * pt[2],
                                  w[0] * pt[1] - w[1] * pt[0] };*/
        T w_cross_pt[3];
        CrossProduct(w, pt, w_cross_pt);//即得出(w x pt) 的结果
        // (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);
        const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);//即得出(w·pt) (1 - costheta)的结果


        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        // 在零附近，对应于向量 w 和角度 w 的旋转矩阵 R 的一阶泰勒近似为
        //   R = I + hat(w) * sin(theta)
        // 但 sintheta ~ theta 和 theta * w = angle_axis，这给了我们
        //  R = I + hat(w)
        // 实际执行与点 pt 的乘法，给我们
        // R * pt = pt + w x pt.
         // 在使用 Jets 进行评估时，切换到接近零的泰勒展开提供了有意义的导数。出于性能原因，对叉积进行显式内联评估。
        /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                                  angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                                  angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
        T w_cross_pt[3];
        CrossProduct(angle_axis, pt, w_cross_pt);

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

#endif // rotation.h