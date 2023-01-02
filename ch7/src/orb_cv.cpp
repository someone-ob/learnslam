#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>//特征点头文件，处理特征点信息
#include <opencv2/highgui/highgui.hpp>//opencv gui头文件
#include <chrono>//用于计时的头文件
 
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  //-- 读取图像
  Mat img_1 = imread("/home/jzh/Code/learnslam/ch7/1.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片1 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  Mat img_2 = imread("/home/jzh/Code/learnslam/ch7/2.png", CV_LOAD_IMAGE_COLOR);//读取彩色图片2 CV_LOAD_IMAGE_COLOR表示返回的是一张彩色图
  assert(img_1.data != nullptr && img_2.data != nullptr); //assert()为断言函数，如果它的条件返回错误，则终止程序执行

    //-- 初始化
    vector<KeyPoint>keypoints_1, keypoints_2;//图片1 -> 关键点1 图片2 -> 关键点2
    Mat descriptors_1, descriptors_2;//描述子
    Ptr<FeatureDetector> detector = ORB::create(2000);//可以修改特征点的个数来增加匹配点数量 特征点检测
    Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//特征匹配

    //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//检测 Oriented FAST 角点前计时
  detector->detect(img_1, keypoints_1);//检测图片1的Oriented FAST 角点
  detector->detect(img_2, keypoints_2);//检测图片2的Oriented FAST 角点
 
  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);//计算图片1的描述子
  descriptor->compute(img_2, keypoints_2, descriptors_2);//计算图片2的描述子
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计算耗时
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算检测角点和计算描述子所用的时间
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;//输出extract ORB cost 

  Mat outimg1;//定义ORB特征显示结果的变量
  drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);//画出图像1的ORB特征点提取结果
  imshow("ORB features", outimg1);//显示图像1的ORB特征点提取结果
  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;//匹配matches
  t1 = chrono::steady_clock::now();//计时
  matcher->match(descriptors_1, descriptors_2, matches);//描述子1和描述子2进行匹配
  t2 = chrono::steady_clock::now();//计时
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;//输出match ORB cost
    //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  //minmax_element()为c++中定义的寻找最小值和最大值的函数。
  //第3个参数表示比较函数，默认从小到大，可以省略
  double min_dist = min_max.first->distance;//将两幅图像的ORB特征点之间的最小距离赋值给min_dist
  double max_dist = min_max.second->distance;//将两幅图像的ORB特征点之间的最大距离赋值给max_dist
 
  printf("-- Max dist : %f \n", max_dist);//	输出两幅图像的ORB特征点匹配的最大距离
  printf("-- Min dist : %f \n", min_dist);//	输出两幅图像的ORB特征点匹配的最小距离
 
  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;//
  for (int i = 0; i < descriptors_1.rows; i++)//遍历描述子
   {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) //不同的结果可以在这里设置
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误 30.0为经验值
    {
      good_matches.push_back(matches[i]);
    }
  }
    //-- 第五步:绘制匹配结果
  Mat img_match;//绘制匹配结果变量
  Mat img_goodmatch;//绘制剔除误匹配的匹配结果变量
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);//画出粗匹配匹配结果
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);//画出剔除误匹配的匹配结果
  imshow("all matches", img_match);//界面展示粗匹配匹配结果
  imshow("good matches", img_goodmatch);//界面展示剔除误匹配的匹配结果
  waitKey(0);
  return 0;
}