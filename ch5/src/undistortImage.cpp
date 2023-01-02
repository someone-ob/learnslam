#include <opencv2/opencv.hpp>
#include <string>
 
using namespace std;
    //#region 图像去畸变方法（手动计算）
    // ①根据已畸变的图像的rows，cols和type(本例中是CV_8UC1)创建一个空白对象;
    // ②遍历空白对象的每个像素坐标(两个for循环嵌套)，即(u,v);
    // ③将遍历到的(u,v)转换成其对应的归一化坐标(x,y);
    //     注：归一化坐标即(X/Z,Y/Z,1),(X,Y,Z)为观测点的世界坐标根据相机位姿转换过来的，又称相机坐标；
    //     像素坐标为相机内参K乘以归一化坐标；K为三阶矩阵，转化为方程组为u=fx(X/Z)+cx,v=fy(Y/Z)+cy
    // ④计算畸变参数r，r = sqrt(x * x + y * y)，对x平方和y平方开根；
    // ⑤根据公式计算畸变后的归一化坐标；
    // ⑥再把畸变后的归一化坐标转化成畸变后的像素坐标；
    // ⑦把畸变后的像素坐标的像素信息赋给(u,v)这个位置；
    //流程：(u_undistort,v_undistort)→(x_undistort,y_undistort)→参数r→(x_distort，y_distort)→(u_distort,v_distort)→像素赋予
    // #endregion
string image_file = "/home/jzh/Code/learnslam/ch5/distorted.png";   

int main(int argc,char **argv){
  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05; // 畸变参数
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;  // 相机内参
    cv::Mat image=cv::imread(image_file,0);
    int rows=image.rows,cols=image.cols;
    cv::Mat image_undistort=cv::Mat(rows,cols,CV_8UC1);

    for(int v=0;v<rows;v++){
        for (int u = 0; u < cols; u++) {
            double x = (u - cx) / fx, y = (v - cy) / fy;
            double r = sqrt(x * x + y * y);//对x平方和y平方开根
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);//书上p102式5.12
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;//书上p102式5.12
            double u_distorted = fx * x_distorted + cx;//书上p102式5.13
            double v_distorted = fy * y_distorted + cy;//书上p102式5.13
            // 赋值 (最近邻插值) 
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);//将image(v_distorted,u_distorted)处的颜色信息赋值到image_undistort(v,u)处
            } else {
                image_undistort.at<uchar>(v, u) = 0;
             }
        }
    }
    // 画图去畸变后图像
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey(); //程序终止，等待一个按键输入
    return 0;
}