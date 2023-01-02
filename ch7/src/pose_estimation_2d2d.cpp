#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui模块
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>
#include "FindFeatureMatches.h"
#include "poseEstimation2d2d.h"
#include  "pixel2cam.h"

using namespace std;
using namespace cv;

int main(int argc,char **argv){
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  //计时开始
  //-- 读取图像
  Mat img_1 = imread("/home/jzh/Code/learnslam/ch7/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片1 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  Mat img_2 = imread("/home/jzh/Code/learnslam/ch7/2.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片2 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  assert(img_1.data != nullptr && img_2.data != nullptr); //assert()为断言函数，如果它的条件返回错误，则终止程序执行
 
  //特征提取和特征匹配
  vector<KeyPoint> keypoints_1, keypoints_2;//特征点1 -> 图像1 特征点2 -> 图像2
  vector<DMatch> matches;//匹配 matches
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);//调用find_feature_matches函数
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;//输出匹配点数
  //-- 估计两张图像间运动
  Mat R, t;//R,t表示旋转矩阵和平移
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);//调用函数
  
  //-- 验证E=t^R*scale
  Mat  t_x=(Mat_<double>(3,3)<<0,-t.at<double>(2,0),t.at<double>(1,0),
  t.at<double>(2,0),0,-t.at<double>(0,0),
  -t.at<double>(1,0),t.at<double>(0,0),0);
  cout << "t^R=" << endl << t_x * R << endl;//输出t^R

  //-- 验证对极约束
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//相机内参矩阵
  for(int i = 0; i < matches.size(); i++)
  {
    DMatch m = matches[i];
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//像素坐标转相机归一化坐标
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1); 
    Mat d = y2.t() * t_x * R * y1;//对极几何的残差，结果应该为标量0 d = y2(T) * t(^)* R * y1 	就是视觉slan十四讲p167的式7.8
    cout << "The " << i << " epipolar constraint（匹配点对的对极几何残差）： " << d << " ！" << endl;
  }
 chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2- t1);
  cout << "执行程序所花的时间为" << time_used.count() << "秒！" << endl;
  return 0;
}