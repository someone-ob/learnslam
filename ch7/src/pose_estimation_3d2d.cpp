#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui
#include <opencv2/calib3d/calib3d.hpp>//求解器头文件
#include <Eigen/Core>//eigen核心模块
#include <g2o/core/base_vertex.h>//g2o顶点（Vertex）头文件 视觉slam十四讲p141用顶点表示优化变量，用边表示误差项
#include <g2o/core/base_unary_edge.h>//g2o边（edge）头文件
#include <g2o/core/sparse_optimizer.h>//稠密矩阵求解
#include <g2o/core/block_solver.h>//求解器头文件
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿算法头文件
#include <g2o/solvers/dense/linear_solver_dense.h>//线性求解
#include <sophus/se3.hpp>//李群李代数se3
#include <chrono>
#include "FindFeatureMatches.h"
#include "poseEstimation2d2d.h"
#include  "pixel2cam.h"
#include "bundleAdjustmentGaussNewton.h"

 
using namespace std;
using namespace cv;
using namespace Sophus;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;//2d点 见视觉slam十四讲 p180-p187 需要输入2d点
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;//3d点 见视觉slam十四讲 p180-p187 需要输入3d点
 
void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
); //定义bundleAdjustmentG2O函数 输入3d点和2d点和相机内参矩阵，输出pose为优化变量 使用BA方法

int main(int argc,char ** argv){
  //-- 读取图像
  Mat img_1 = imread("/home/jzh/Code/learnslam/ch7/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片1 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  Mat img_2 = imread("/home/jzh/Code/learnslam/ch7/2.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片2 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  assert(img_1.data != nullptr && img_2.data != nullptr); //assert()为断言函数，如果它的条件返回错误，则终止程序执行

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2,keypoints_1,keypoints_2,matches);//定义find_feature_matches函数 输入图像1和图像2，输出特征点集合1、特征点集合2和匹配点对

  /************************调用opencv函数求解pnp******************************/
  // 建立3D点
  Mat d1 = imread("/home/jzh/Code/learnslam/ch7/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//初始化相机内参矩阵
  vector<Point3f> pts_3d;//3d路标点
  vector<Point2f> pts_2d;//2d像素点
  //遍历所有特征点，将keypoints_1的点归一化，并乘上深度图中的深度，得到3d点，存入3d路标点数组中，将keypoints_2中的点存入2d像素点数组中
  //这里将keypoints_1的特征点对应的相机坐标系的3d位置作为keypoints_2的匹配特征点所对应的世界坐标系的3d位置。这里相当于把keypoints_1所处的相机坐标系当成了世界坐标系
  //通过epnp求出这个世界坐标系到keypoints_2对应的相机坐标系的R和t
  for (DMatch m:matches){
    ushort d=d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d == 0)   // bad depth
      continue;
    float dd = d / 5000.0;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//像素坐标转相机归一化坐标
    pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
    pts_2d.push_back(keypoints_2[m.trainIdx].pt);
  }

  cout << "3d-2d pairs: " << pts_3d.size() << endl;//输出3d-2d pairs
 
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
  Mat r, t;//r表示旋转向量 t表示平移向量
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
  Mat R;//旋转矩阵R
  cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵 视觉slam十四讲p53式3.15
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
  cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;//输出使用opencv中使用pnp所花费的时间
  cout << "R=" << endl << R << endl;//输出旋转矩阵
  cout << "t=" << endl << t << endl;//输出平移向量

  //*************************使用BA方法*****************************************
  VecVector3d pts_3d_eigen;
  VecVector2d pts_2d_eigen;
  for (size_t i = 0; i < pts_3d.size(); ++i) //遍历3d点
  {
    pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
    pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
  }
 
  cout << "calling bundle adjustment by gauss newton" << endl;//输出calling bundle adjustment by gauss newton
  Sophus::SE3d pose_gn;
  cout<<pose_gn.matrix();
  t1 = chrono::steady_clock::now();//计时开始
  bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);//调用bundleAdjustmentGaussNewton函数
  t2 = chrono::steady_clock::now();//计时结束
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计时结束
  cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;//输出

   cout << "calling bundle adjustment by g2o" << endl;
  Sophus::SE3d pose_g2o;
  t1 = chrono::steady_clock::now();
  bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;//输出在G2O中求解pnp耗时
  return 0;
}



/// vertex and edges used in g2o ba
// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>//:表示继承，public表示公有继承；CurveFittingVertex是派生类，:BaseVertex<6, Sophus::SE3d>是基类
 {
public://以下定义的成员变量和成员函数都是公有的
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题
  // 重置
  virtual void setToOriginImpl() override //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
  {
    _estimate = Sophus::SE3d();
  }
 
  // left multiplication on SE3
  // 更新
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }
  // 存盘和读盘：留空
  virtual bool read(istream &in) override {}//istream类是c++标准输入流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
 
  virtual bool write(ostream &out) const override {}//ostream类是c++标准输出流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题
 
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}//使用列表赋初值
 
  virtual void computeError() override//virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
   {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);//创建指针v
    Sophus::SE3d T = v->estimate();//将estimate()值赋给V
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }
 
  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  } //雅克比矩阵表达式见 视觉slam十四讲p186式7.46
 
  virtual bool read(istream &in) override {}
 
  virtual bool write(ostream &out) const override {}
 
private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
 
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));//c++中的make_unique表示智能指针类型
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出
 
  // vertex
  // 往图中增加顶点
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);//对顶点进行编号，里面的0你可以写成任意的正整数，但是后面设置edge连接顶点时，必须要和这个一致
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);//添加顶点
 
  // K 相机内参矩阵
  Eigen::Matrix3d K_eigen;
  K_eigen <<
    K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);
 
  // edges
  // 往图中增加边
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i)//遍历2d点
   {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);//对顶点进行编号，里面的0你可以写成任意的正整数，但是后面设置edge连接顶点时，必须要和这个一致
    edge->setVertex(0, vertex_pose);  // 设置连接的顶点
    edge->setMeasurement(p2d);// 观测数值
    edge->setInformation(Eigen::Matrix2d::Identity());// 信息矩阵：协方差矩阵之逆
    optimizer.addEdge(edge);//添加边
    index++;
  }
 
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);//迭代次数10
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}