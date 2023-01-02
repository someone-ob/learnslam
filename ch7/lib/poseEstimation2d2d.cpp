#include "poseEstimation2d2d.h"

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,std::vector<KeyPoint> keypoints_2,std::vector<DMatch> matches,Mat &R, Mat &t){
  Mat K=(Mat_<double>(3,3)<< 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;
  for (int i=0;i<matches.size();i++){
    points1.push_back(keypoints_1[matches[i].trainIdx].pt);
    points2.push_back(keypoints_2[matches[i].queryIdx].pt);
  }
  //-- 计算基础矩阵
  Mat fundamental_matrix;
  fundamental_matrix=findFundamentalMat(points1,points2,CV_FM_8POINT);
  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;//输出基础矩阵
  //-- 计算本质矩阵
  Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
  double focal_length = 521;      //相机焦距, TUM dataset标定值
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);//E = t(^) * R
  cout << "essential_matrix is " << endl << essential_matrix << endl;//输出本质矩阵
  //-- 计算单应矩阵
  //-- 但是本例中场景不是平面，单应矩阵意义不大
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, RANSAC, 3);//H = K * (R - tn(T) / d) * K(-1) 
  cout << "homography_matrix is " << endl << homography_matrix << endl;

  recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);
  cout << "R is " << endl << R << endl;//输出旋转矩阵
  cout << "t is " << endl << t << endl;//输出平移向量
}