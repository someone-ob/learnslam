#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui模块
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);//定义pose_estimation_2d2d 输入特征点集合1、特征点集合2和匹配点对，输出估计的旋转矩阵、估计的平移向量和本质矩阵，平移向量差了一个尺度因子(可详细阅读书上p175-p176)
  //R,t表示旋转矩阵和平移