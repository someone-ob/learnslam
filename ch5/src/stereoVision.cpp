#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>//Eigen核心模块
#include <pangolin/pangolin.h>
#include <unistd.h>
 
using namespace std;
using namespace Eigen;
using namespace pangolin;
 
// 文件路径
string left_file = "/home/jzh/Code/learnslam/ch5/left.png";
string right_file = "/home/jzh/Code/learnslam/ch5/right.png";
 
//string left_file = "./left.png";
//string right_file = "./right.png";//该方法需要把两张图像放在build文件夹下面
 
 
// 在pangolin中画图，已写好，无需调整
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);
 
int main(int argc, char **argv) {
 
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;
 
    // 读取图像
    cv::Mat left = cv::imread(left_file, 0);//0表示返回一张灰度图
    cv::Mat right = cv::imread(right_file, 0);//0表示返回一张灰度图
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    //关于sgbm算法的经典参数配置
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); //注意disparity才是最后的视差图
 
    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;//前三维为X,Y,Z，表示位置信息，后一维表示颜色信息，在此处为灰度
 
    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) 
        {
            //设置一个Check，排除视差不在(10.0, 96.0)范围内的像素点
            if (disparity.at<float>(v, u) <= 10.0 || disparity.at<float>(v, u) >= 96.0) continue;
            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色 创建一个Vector4d类型的变量，前三维用来存储位置信息，后一维为归一化之后的灰度
 
            // 根据双目模型计算 point 的位置  根据双目模型恢复像素点的三维位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
             //将X,Y,Z赋值给point的前三维
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;
            pointcloud.push_back(point);
        }
 
    cv::imshow("disparity", disparity / 96.0);//disparity/96表示归一化之后的视差图像
    cv::waitKey(0);//停止执行，等待一个按键输入
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}
 
void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {
 
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }
    //创建一个pangolin窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);//分别表示窗口名Point Cloud Viewer、窗口宽度=1024和窗口高度=768
    glEnable(GL_DEPTH_TEST);//根据物体远近，实现遮挡效果
    glEnable(GL_BLEND); //使用颜色混合模型，让物体显示半透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//GL_SRC_ALPHA表示使用源颜色的alpha值作为权重因子，GL_ONE_MINUS_SRC_ALPHA表示使用(1-源颜色的alpha值)作为权重因子
 
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
   //ProjectionMatrix()中各参数依次为图像宽度=1024、图像高度=768、fx=500、fy=500、cx=512、cy=389、最近距离=0.1和最远距离=1000
   //ModelViewLookAt()中各参数为相机位置，被观察点位置和相机哪个轴朝上
   //比如，ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)表示相机在(0, -0.1, -1.8)位置处观看视点(0, 0, 0)，并设置相机XYZ轴正方向为（0，-1，0），即右上前
 
   //创建交互视图，显示上一帧图像内容
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
   //SetBounds()内的前4个参数分别表示交互视图的大小，均为相对值，范围在0.0至1.0之间
   //第1个参数表示bottom，即为视图最下面在整个窗口中的位置
   //第2个参数为top，即为视图最上面在整个窗口中的位置
   //第3个参数为left，即视图最左边在整个窗口中的位置
   //第4个参数为right，即为视图最右边在整个窗口中的位置
   //第5个参数为aspect，表示横纵比
 
 
    while (pangolin::ShouldQuit() == false) //如果pangolin窗口没有关闭，则执行
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存，使得前后帧不会互相干扰
 
        d_cam.Activate(s_cam);//激活显示，并设置相机状态
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);//设置背景颜色为白色
 
        glPointSize(2);
        glBegin(GL_POINTS); //绘制点云
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]); //设置颜色信息
            glVertex3d(p[0], p[1], p[2]);//设置位置信息
        }
        glEnd();
        pangolin::FinishFrame();//按照上面的设置执行渲染
        usleep(5000);   // sleep 5 ms 停止执行5毫秒
    }
    return;
}