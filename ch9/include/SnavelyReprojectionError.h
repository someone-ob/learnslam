#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera,const T *const point,T *residuals) const {
        // camera是相机参数
        //point是路标点世界坐标
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);//得到畸变后的像素坐标predictions
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];
        //下面两步相当于变换矩阵T*p
        AngleAxisRotatePoint(camera, point, p);//p为相机旋转向量R乘以路标点世界坐标
        //p再加上平移向量
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        //得到未畸变过的归一化坐标
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // 得到畸变参数k1,k2
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        //得到径向畸变公式
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
        
        //distortion * xp是得到畸变后的归一化坐标
        //归一化坐标乘以焦距得到像素坐标(该数据集中没有cx，cy，默认为0)
        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h
