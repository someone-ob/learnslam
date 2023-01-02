#include <iostream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include<chrono>
#include "FindFeatureMatches.h"
#include "poseEstimation2d2d.h"
#include  "pixel2cam.h"

using namespace std;
using namespace cv;

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

/// 作图用
//inline表示内联函数
inline cv::Scalar get_color(float depth) //depth表示输入深度，get_color表示返回颜色信息
{
  float up_th = 50, low_th = 10, th_range = up_th - low_th;//这里相当于定义阈值
  if (depth > up_th) depth = up_th;//depth > 50 depth = 50
  if (depth < low_th) depth = low_th;//depth < 10 depth = 10
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range)); //Scalar()中的颜色顺序为BGR
  //B = 255 * depth / th_range =  255 * depth / 40，G = 0，R = 255 * (1 - depth / th_range) = 255 * (1 - depth /40 ) 
  //反映出的深度信息就是图像里的像素点越远，对应的特征点颜色越红
}

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

      //-- 三角化得到特征点在相机1的相机坐标系下的坐标
  vector<Point3d> points;//相机1中的特征点在其相机坐标系中的坐标
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);
 
  //-- 验证三角化点与特征点的重投影关系
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//相机内参矩阵
  Mat img1_plot = img_1.clone();
  Mat img2_plot = img_2.clone();
  for (int i = 0; i < matches.size(); i++) {
    // 第一个图
    float depth1 = points[i].z;  //图像1深度信息
    cout << "depth: " << depth1 << endl;//输出深度信息
    Mat pt1_cam = K * (Mat_<double>(3, 1) << points[i].x / points[i].z, points[i].y / points[i].z, 1); //pt1_cam表示的是归一化坐标(u, v, 1) 具体形式可参考视觉slam十四讲p168的式7.11
    //Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);//像素坐标转换为相机坐标
    Point2f pixel1 = keypoints_1[matches[i].queryIdx].pt;
    cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);//画圆
    
    Point2f residual1;//图像1残差
    residual1.x = pt1_cam.at<double>(0, 0) - pixel1.x;
    residual1.y = pt1_cam.at<double>(1, 0) - pixel1.y;
    cout << "图像1像素点的深度为" << depth1 << "平移单位 "<< endl;
    cout<<"图像1的残差为: " << residual1.x << ", " << residual1.y << ")像素单位 " << endl;
 
    // 第二个图
    Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;//相当于x2 = R * x1 + t的表达式 //pt2_trans表示的是归一化坐标(u, v, 1) 具体形式可参考视觉slam十四讲p168的式7.11
    float depth2 = pt2_trans.at<double>(2, 0); //图像2的深度信息
    Point2f pixel2 = keypoints_2[matches[i].trainIdx].pt;
    cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);//画圆
    Point2f residual2;//图像2残差
    residual2.x = pt2_trans.at<double>(0, 0) - pixel2.x;
    residual2.y = pt2_trans.at<double>(1, 0) - pixel2.y;
    cout << "图像2像素点的深度为" << depth2 << "平移单位 "<< endl;
    cout<<"图像2的残差为: " << residual2.x << ", " << residual2.y << ")像素单位 " << endl;
 
  }
  cv::imshow("img 1", img1_plot);//界面展示图像1的深度信息结果 红圈
  cv::imshow("img 2", img2_plot);//界面展示图像2的深度信息结果 红圈
  cv::waitKey();
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);//计算耗时
  cout << "执行程序所花费的时间为：" << time_used.count() << "秒！" << endl;
  return 0;
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );
 
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//相机内参矩阵
  vector<Point2f> pts_1, pts_2;//归一化平面上的点，triangulatePoints()函数的输入参数
  for (DMatch m:matches) {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }
 
  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);
 
  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);//取第i列
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}