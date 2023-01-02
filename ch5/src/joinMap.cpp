#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
 
using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;//6*1矩阵
 
// 在pangolin中画图，已写好，无需调整
//在pangolin中绘制点云图
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);
 
int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;    // 彩色图和深度图 定义彩色图向量类容器和深度图向量类容器
    TrajectoryType poses;         // 相机位姿
 
    ifstream fin("/home/jzh/Code/learnslam/ch5/pose.txt");//  ifstream fin("../pose.txt"); //创建文件输入流fin
    if (!fin) {
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
 
    for (int i = 0; i < 5; i++) //有五张图
     {
        //把彩色图放到colorImages中，把深度图放到depthImages中！
        boost::format fmt("/home/jzh/Code/learnslam/ch5/%s/%d.%s"); //图像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1读取原始图像
        //把位姿放到poses中！
        double data[7] = {0};//用数组存储单个位姿SE3d
        for (auto &d:data)//基于范围的for循环，auto表示自动类型推导
            fin >> d;//fin表示文件输入流
        //用李群存储的单个位姿
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }
 
    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;//x方向上的原点平移量
    double cy = 253.5;//y方向上的原点平移量
    double fx = 518.0;//焦距
    double fy = 519.0;//焦距
    double depthScale = 1000.0; //现实世界中1米在深度图中存储为一个depthScale值
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud; //创建一个vector<Vector6d>变量
    pointcloud.reserve(1000000);//reserve()函数用来给vector预分配存储区大小，但不对该段内存进行初始化
 
    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];//用SE3d表示的从当前相机坐标系到世界坐标系的变换
        //遍历每个像素点
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                 if (d == 0) continue; // 为0表示没有测量到
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;//真实世界中的深度值
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
 
                Vector6d p;//前三维表示点云的位置，后三维表示点云的颜色
                p.head<3>() = pointWorld;//head<n>()函数是对于Eigen库中的向量类型而言的，表示提取前n个元素
                //opencv中图像的data数组表示把其颜色信息按行优先的方式展成的一维数组！
                //color.step等价于color.cols
                //color.channels()表示图像的通道数
                p[5] = color.data[v * color.step + u * color.channels()];   // blue 蓝色分量
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green 绿色分量
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red 红色分量
                pointcloud.push_back(p);
            }
    }
 
    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);
    return 0;
}
 
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
 
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }
    //生成一个pangolin窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);//分别表示窗口名Point Cloud Viewer、窗口宽度=1024和窗口高度=768
    glEnable(GL_DEPTH_TEST);//根据物体远近，实现遮挡效果
    glEnable(GL_BLEND);//使用颜色混合模型，让物体显示半透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//GL_SRC_ALPHA表示使用源颜色的alpha值作为权重因子，GL_ONE_MINUS_SRC_ALPHA表示使用(1-源颜色的alpha值)作为权重因子
   
   //ProjectionMatrix()中各参数依次为图像宽度=1024、图像高度=768、fx=500、fy=500、cx=512、cy=389、最近距离=0.1和最远距离=1000
   //ModelViewLookAt()中各参数为相机位置，被观察点位置和相机哪个轴朝上
   //比如，ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)表示相机在(0, -0.1, -1.8)位置处观看视点(0, 0, 0)，并设置相机XYZ轴正方向为（0，-1，0），即右上前
 
   //创建交互视图，显示上一帧图像内容
 
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
 
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
   //SetBounds()内的前4个参数分别表示交互视图的大小，均为相对值，范围在0.0至1.0之间
   //第1个参数表示bottom，即为视图最下面在整个窗口中的位置
   //第2个参数为top，即为视图最上面在整个窗口中的位置
   //第3个参数为left，即视图最左边在整个窗口中的位置
   //第4个参数为right，即为视图最右边在整个窗口中的位置
   //第5个参数为aspect，表示横纵比
   
while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}