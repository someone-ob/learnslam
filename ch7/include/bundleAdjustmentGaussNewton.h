#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui
#include <opencv2/calib3d/calib3d.hpp>//求解器头文件
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace cv;
using namespace Sophus;

// BA by g2o
//aligned_allocator是用来声明内存管理方式
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;//2d点 见视觉slam十四讲 p180-p187 需要输入2d点
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;//3d点 见视觉slam十四讲 p180-p187 需要输入3d点
// BA by gauss-newton
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);//定义bundleAdjustmentGaussNewton函数 输入3d点和2d点和相机内参矩阵，输出pose为优化变量 使用高斯牛顿法