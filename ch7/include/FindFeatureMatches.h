#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui模块
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);//定义find_feature_matches函数 输入图像1和图像2，输出特征点集合1、特征点集合2和匹配点对