#include <iostream>
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/features2d/features2d.hpp>//opencv特征点
#include <opencv2/highgui/highgui.hpp>//opencv gui
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>//eigen核心模块
#include <Eigen/Dense>//eigen稠密矩阵
#include <Eigen/Geometry>//eigen几何模块
#include <Eigen/SVD>//SVD分解 线性代数求解
#include <g2o/core/base_vertex.h>//g2o顶点（Vertex）头文件 视觉slam十四讲p141用顶点表示优化变量，用边表示误差项
#include <g2o/core/base_unary_edge.h>//g2o边（edge）头文件
#include <g2o/core/block_solver.h>//求解器头文件
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿算法头文件
#include <g2o/core/optimization_algorithm_levenberg.h>//列文伯格——马尔夸特算法头文件
#include <g2o/solvers/dense/linear_solver_dense.h>//线性求解
#include <chrono>
#include <sophus/se3.hpp>//李群李代数se3
 #include "FindFeatureMatches.h"
#include "poseEstimation2d2d.h"
#include  "pixel2cam.h"
using namespace std;
using namespace cv;
 
 void bundleAdjustment(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);//定义bundleAdjustmentGaussNewton函数 输入3d点和2d点和相机内参矩阵
/// vertex and edges used in g2o ba
// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class VertexPose:public g2o::BaseVertex<6,Sophus::SE3d>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        virtual void setToOriginImpl() override{
            _estimate=Sophus::SE3d();
        }

        virtual void oplusImpl(const double *update) override{
            Eigen::Matrix<double,6,1> update_eigen;
            update_eigen<<update[0],update[1],update[2],update[3],update[4],update[5];
            _estimate=Sophus::SE3d::exp(update_eigen)*_estimate;
        }

        virtual bool read(istream &in) override{}

        virtual bool write(ostream &out) const override{}
};

class EdgeProjectXYZRGBDPoseOnly:public g2o :: BaseUnaryEdge<3, Eigen::Vector3d, VertexPose>{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}//使用列表赋初值
        virtual void computeError() override{
            const VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
            _error = _measurement-pose->estimate()*_point;
        }
        virtual void linearizeOplus() override{//virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
            VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);//创建指针pose
            Sophus::SE3d T = pose->estimate();//将estimate()值赋给T
            Eigen::Vector3d xyz_trans = T * _point;
            //<>指定子块大小，()指定子块起点
            _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
            _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);//hat返回向量对应的反对称矩阵
        }

        bool read(istream &in) {}//istream类是c++标准输入流的一个基类
        //可参照C++ Primer Plus第六版的6.8节
 
        bool write(ostream &out) const {}//ostream类是c++标准输出流的一个基类
        //可参照C++ Primer Plus第六版的6.8节
    protected:
        Eigen::Vector3d _point;
};

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);//定义pose_estimation_3d3d函数 

int main(int argc,char ** argv){
  //-- 读取图像
  Mat img_1 = imread("/home/jzh/Code/learnslam/ch7/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片1 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  Mat img_2 = imread("/home/jzh/Code/learnslam/ch7/2.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片2 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  assert(img_1.data != nullptr && img_2.data != nullptr); //assert()为断言函数，如果它的条件返回错误，则终止程序执行

  //特征提取和特征匹配
  vector<KeyPoint> keypoints_1, keypoints_2;//特征点1 -> 图像1 特征点2 -> 图像2
  vector<DMatch> matches;//匹配 matches
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);//调用find_feature_matches函数
  cout << "一共找到了" << matches.size() << "组匹配点" << endl;//输出匹配点数

   
  // 建立3D点
  Mat depth1 = imread("/home/jzh/Code/learnslam/ch7/1_depth.png", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat depth2 = imread("/home/jzh/Code/learnslam/ch7/2_depth.png", CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
  Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);//相机内参矩阵
  vector<Point3f> pts1, pts2;//3d路标点
 
  for (DMatch m:matches) {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)   // bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);//像素坐标转相机归一化坐标
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);//像素坐标转相机归一化坐标
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }
 
  cout << "3d-3d pairs: " << pts1.size() << endl;//输出3d-3d pairs
   Mat R, t;//旋转矩阵，平移向量
  pose_estimation_3d3d(pts1, pts2, R, t);//调用pose_estimation_3d3d函数
  cout << "ICP via SVD results: " << endl;//输出ICP via SVD results
  cout << "R = " << R << endl;//输出旋转矩阵
  cout << "t = " << t << endl;//输出平移向量
  cout << "R_inv = " << R.t() << endl;//输出 R.t()
  cout << "t_inv = " << -R.t() * t << endl;//输出 -R.t() * t

   cout << "calling bundle adjustment" << endl;//输出calling bundle adjustment
 
  bundleAdjustment(pts1, pts2, R, t);//调用bundleAdjustment函数
 
  // verify p1 = R * p2 + t
  for (int i = 0; i < 5; i++) 
  {
    cout << "p1 = " << pts1[i] << endl;//输出p1
    cout << "p2 = " << pts2[i] << endl;//输出p2
    cout << "(R*p2+t) = " <<
         R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
         << endl;//输出R*p2+t
    cout << endl;
  }
}

void pose_estimation_3d3d(const vector<Point3f> &pts1,const vector<Point3f> &pts2,Mat &R, Mat &t){
    Point3f p1,p2;      // center of mass 视觉slam十四讲中p197式7.51中的两个量
      int N = pts1.size();//定义N 视觉slam十四讲中p197式7.51中的n
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];//视觉slam十四讲中p197式7.51中左式累加部分
        p2 += pts2[i];//视觉slam十四讲中p197式7.51右右式累加部分
    }
    p1 = Point3f(Vec3f(p1) / N);//视觉slam十四讲中p197式7.51
    p2 = Point3f(Vec3f(p2) / N);//视觉slam十四讲中p197式7.51

    vector<Point3f> q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;//视觉slam十四讲中p197式7.53上面的式子
    q2[i] = pts2[i] - p2;//视觉slam十四讲中p197式7.53上面的式子
  }
 
  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();//将w初始化为0
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();//视觉slam十四讲中p198式7.57
  }
  cout << "W=" << W << endl;//输出W
 
  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();//求解U
  Eigen::Matrix3d V = svd.matrixV();//求解V
 
  cout << "U=" << U << endl;//输出U
  cout << "V=" << V << endl;//输出V
 
  Eigen::Matrix3d R_ = U * (V.transpose());//视觉slam十四讲中p198式7.59
  if (R_.determinant() < 0)//如果|R_| < 0 
   {
    R_ = -R_;//取反
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);//视觉slam十四讲中p198式7.54
 
  // convert to cv::Mat
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );//旋转矩阵
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));//平移向量
}

void bundleAdjustment(const vector<Point3f> &pts1,const vector<Point3f> &pts2,Mat &R, Mat &t) {
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolverX BlockSolverType;  //在这里要说明顶点的维数和边的维数
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));//c++中的make_unique表示智能指针类型

  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出
 
  // vertex
  VertexPose *pose = new VertexPose(); // camera pose 定义顶点变量pose
  pose->setId(0);//对顶点进行编号，里面的0你可以写成任意的正整数，但是后面设置edge连接顶点时，必须要和这个一致
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);//添加顶点
 
  // edges
  for (size_t i = 0; i < pts1.size(); i++) {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
      Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));//输入的是pts2的坐标
    edge->setVertex(0, pose);//设置边连接到的顶点
    edge->setMeasurement(Eigen::Vector3d(
      pts1[i].x, pts1[i].y, pts1[i].z)); //_measurement就是pts2
    edge->setInformation(Eigen::Matrix3d::Identity());//设置测量噪声的信息矩阵，即协方差矩阵的逆
    optimizer.addEdge(edge);//往优化器中添加边
  }
 
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
  optimizer.initializeOptimization(); //优化器初始化 	
  optimizer.optimize(10);//设置优化器迭代次数为10次，并执行优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;//输出消耗时间
 
  cout << endl << "after optimization:" << endl;//输出after optimization:
  cout << "T=\n" << pose->estimate().matrix() << endl;//输出优化后的T
 
  // convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();//输出旋转矩阵
  Eigen::Vector3d t_ = pose->estimate().translation();//输出平移向量
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}