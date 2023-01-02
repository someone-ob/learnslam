#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui模块
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);//输出一个像素点和相机内参矩阵，输出该像素点在归一化平面上的坐标