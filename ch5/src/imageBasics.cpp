#include <iostream>
#include <chrono>           //std::chrono 来给算法计时所需头文件

using namespace std;

#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/highgui/highgui.hpp>//opencv gui模块

int main(int argc,char **argv){
    cv::Mat image;
     image = cv::imread("/home/jzh/Code/learnslam/ch5/ubuntu.png"); //cv::imread函数读取指定路径下的图像

     //判断图像文件是否正确读取
     if (image.data==nullptr){
        cerr<<"文件不存在"<<endl;
        return 0;
     }
     //文件顺利读取，首先输出一些基本信息
     cout<<"图像宽为"<<image.cols<<"，高为"<<image.rows<<",通道数为"<<image.channels()<<endl;
     cv::imshow("image",image);
     cv::waitKey(0);    //暂停程序，等待一个按键输入

     //判断image的类型
     if(image.type()!=CV_8UC1&&image.type()!=CV_8UC3){
        cout<<"请输入一张彩色图或灰度图。"<<endl;
        return 0;
     }

    // 遍历图像, 请注意以下遍历方式亦可使用于随机像素访问
    // 使用 std::chrono 来给算法计时
    //使用指针遍历图像image中的像素
    //steady_clock是单调的时钟，相当于教练中的秒表，只会增长，适合用于记录程序耗时。
      chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  //size_t全称是size_type，它表示sizeof()函数的返回值，是无符号整型unsigned int变量。
  //使用size_t的目的是提供一种可移植的方法来声明与系统中可寻址的内存区域一致的长度。
  for (size_t y = 0; y < image.rows; y++) {
    // 用cv::Mat::ptr获得图像的行指针
    unsigned char *row_ptr = image.ptr<unsigned char>(y);  // row_ptr是第y行的头指针
    for (size_t x = 0; x < image.cols; x++) {
      // 访问位于 x,y 处的像素
      unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr 指向待访问的像素数据
      // 输出该像素的每个通道,如果是灰度图就只有一个通道
      for (int c = 0; c != image.channels(); c++) {
        unsigned char data = data_ptr[c]; // data为I(x,y)第c个通道的值
      }
    }
  }
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
  cout << "遍历图像用时：" << time_used.count() << " 秒。" << endl;

  // 关于 cv::Mat 的拷贝
  // 直接赋值并不会拷贝数据
  cv::Mat image_another = image;
  // 修改 image_another 会导致 image 发生变化
  image_another(cv::Rect(0, 0, 100, 100)).setTo(0); // 将左上角100*100的块置为黑色
  cv::imshow("image", image);
  cv::waitKey(0);//停止执行，等待一个按键输入
 
  // 使用clone函数来拷贝数据
  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);//将左上角100*100的块置为白色
  cv::imshow("image", image);
  cv::imshow("image_clone", image_clone);
  cv::waitKey(0);//停止执行，等待一个按键输入
 
  // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
  cv::destroyAllWindows();
  return 0;
}