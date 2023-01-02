#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "/home/jzh/Code/learnslam/ch8/LK1.png";  // first image
string file_2 = "/home/jzh/Code/learnslam/ch8/LK2.png";  // second image

/// Optical flow tracker and interface  光流跟踪
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,//图像1
        const Mat &img2_,//图像2
        const vector<KeyPoint> &kp1_,//关键点1  -> 图像 1
        vector<KeyPoint> &kp2_,//关键点2  -> 图像 2
        vector<bool> &success_,//true if a keypoint is tracked successfully 关键点跟踪是正确的
        bool inverse_ = true, bool has_initial_ = false) ://bool型变量 判断是否采用反向光流
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}
 
    void calculateOpticalFlow(const Range &range);//定义calculateOpticalFlow（计算光流）函数
    //Range中有两个关键的变量start和end  Range可以用来表示矩阵的多个连续的行或列
    //Range表示范围从start到end，包含start，但不包含end
 
private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};
 
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,vector<bool> &success,bool inverse = false,bool has_initial_guess = false);//定义OpticalFlowSingleLevel函数 单层光流法
 
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,vector<bool> &success,bool inverse = false);//定义OpticalFlowMultiLevel 多层光流法

//双线性插值求灰度值
//因为img2中的像素坐标是浮点型数据，只能通过插值法获取灰度值
inline float GetPixelValue(const cv::Mat &img, float x, float y) // * get a gray scale value from reference image (bi-linear interpolated)  * @param img * @param x * @param y  * @return the interpolated value of this pixel
{
    // boundary check(边界检验)
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    //若传递的是img1以及对应的x,y,则xx,yy都为0；若传递的是img2的，由于有dx和dy，所以xx和yy的值实际上就是dx和dy
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {
    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);//0表示返回灰度图
    Mat img2 = imread(file_2, 0);//0表示返回灰度图

     // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    //GFTTDetector三个参数从左到右依次为
    //maxCorners表示最大角点数目。在此处为500。
    //qualityLevel表示角点可以接受的最小特征值，一般0.1或者0.01，不超过1。在此处为0.01。
    //minDistance表示角点之间的最小距离。在此处为20。
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    
    //利用OpenCV中的自带函数提取图像1中的GFTT角点
    //然后利用calcOpticalFlowPyrLK()函数跟踪其在图像2中的位置(u,v)
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);//调用opencv  OpticalFlowMultiLevel函数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;//输出使用高斯牛顿法计算光流使用时间

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;//status中元素表示对应角点是否被正确跟踪到，1为正确跟踪，0为错误跟踪
    vector<float> error; //error表示误差
    t1 = chrono::steady_clock::now();//开始计时
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);//调用opencv  calcOpticalFlowPyrLK函数来求解min = || I1(x,y) - I2(x + δx, y + δy) ||2 视觉slam十四讲p214式8.10
    t2 = chrono::steady_clock::now();//计时结束
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
    cout << "optical flow by opencv: " << time_used.count() << endl;//输出使用opencv函数计算光流的耗时

    // plot the differences of those functions
    Mat img2_single;//
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);//将灰度图转换成彩色图，彩色图中BGR各颜色通道值为原先灰度值
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }
 
    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    
 
    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    
     //画出角点连线图
    Mat imgMatches(img1.rows, img1.cols * 2, CV_8UC1);  //定义*行*列的Mat型变量
    Rect rect1 = Rect(0, 0, img1.cols, img1.rows);
    //Rect()有四个参数，第1个参数表示初始列，第2个参数表示初始行，
    //第3个参数表示在初始列的基础上还要加上多少列（即矩形区域的宽度），第4个参数表示在初始行的基础上还要加上多少行（即矩形区域的高度）
    Rect rect2 = Rect(img1.cols, 0, img2.cols, img2.rows);
    img1.copyTo(imgMatches(rect1));
    img2.copyTo(imgMatches(rect2));
 
    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    
    cv::waitKey(0);
 
    return 0;
}

void OpticalFlowSingleLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,vector<bool> &success,bool inverse, bool has_initial) {
    kp2.resize(kp1.size());//用kp1数组大小来初始化kp2
    success.resize(kp1.size());
    //定义了一个OpticalFlowTracker类型的变量tracker，并进行了初始化
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
    //parallel_for_()实现并行调用OpticalFlowTracker::calculateOpticalFlow()的功能
    
}

//使用高斯牛顿法求解图像2中相应的角点坐标
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;//最大迭代次数
    for (size_t i = range.start; i < range.end; i++)//对图像1中的每个GFTT角点进行高斯牛顿优化
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated 优化变量
 
        if (has_initial)//如果kp2进行了初始化，则执行
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }
 
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded
 
        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian 将H初始化为0
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias 将b初始化为0
        Eigen::Vector2d J;  // jacobian 雅克比矩阵J
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) 
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }  
            else 
            {
                // only reset b 只重置矩阵b。在反向光流中，海塞矩阵H在整个高斯牛顿迭代过程中均保持不变
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;//代价初始化为0 
            // compute cost and jacobian 计算代价和雅克比矩阵
            for (int x = -half_patch_size; x < half_patch_size; x++){
                for (int y = -half_patch_size; y < half_patch_size; y++)  //x,y是patch内遍历
                {
                    //(u, v)表示图像中的角点u表示x坐标，v表示y坐标
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);//误差 eij = I1(x,y)-I2(x+dx,y+dy)
                    // Jacobian
                    if (inverse == false) 
                    {
                        //最小二乘问题：min​F(dx,dy)=||f(dx,dy)||^2         f(dx,dy)=I(x,y)−I(x+dx,y+dy)
                        //J=(∂f/∂dx,∂f/∂dy)    f对dx求导，就是−I(x+dx,y+dy)对dx求导，I(x+dx,y+dy)即img2在对应坐标的像素值，对dx求导，就是求x方向的梯度。f对dy求导则同理可得
                        //图像x方向上的梯度就是x方向上两个相邻像素值相减，y方向上的梯度就是y方向上两个相邻像素值相减
                        //这里都是I(x+1,y)-I(x-1,y)，所以要乘以0.5
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) //采用反向光流时
                    {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;//b = -Jij * eij(累加和)
                    cost += error * error;//cost = || eij ||2 2范数
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();//H = Jij Jij(T)(累加和)
                    }
                }
            }
            // compute update
            //求解增量方程，计算更新量
            Eigen::Vector2d update = H.ldlt().solve(b);
 
            if (std::isnan(update[0]))//计算出来的更新量是非数字，光流跟踪失败，退出GN迭代
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
 
            if (iter > 0 && cost > lastCost) //代价不再减小，退出GN迭代
            {
                break;
            }
 
            // update dx, dy 更新优化变量和lastCost
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
 
            if (update.norm() < 1e-2) //更新量的模小于1e-2，退出GN迭代
            {
                // converge
                break;
            }
        }//GN法进行完一次迭代
 
        success[i] = succ;
 
        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}//对图像1中的所有角点都完成了光流跟踪

void OpticalFlowMultiLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,vector<bool> &success,bool inverse) {
    // parameters
    int pyramids = 4;//金字塔层数为4
    double pyramid_scale = 0.5;//每层之间的缩放因子设为0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};
 
    // create pyramids 创建图像金字塔
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
    vector<Mat> pyr1, pyr2; // image pyramids pyr1 -> 图像1的金字塔 pyr2 -> 图像2的金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) 
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } 
        else 
        {
            Mat img1_pyr, img2_pyr;
            //将图像pyr1[i-1]的宽和高各缩放0.5倍得到图像img1_pyr
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            //将图像pyr2[i-1]的宽和高各缩放0.5倍得到图像img2_pyr
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
    cout << "build pyramid time: " << time_used.count() << endl;//输出构建图像金字塔的耗时
 
    // coarse-to-fine LK tracking in pyramids 由粗至精的光流跟踪
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) 
    {
        auto kp_top = kp;//这里意思大概是视觉slam十四讲p215的把上一层的追踪结果作为下一层光流的初始值
        kp_top.pt *= scales[pyramids - 1];//
        kp1_pyr.push_back(kp_top);//最顶层图像1的角点坐标
        kp2_pyr.push_back(kp_top);//最顶层图像2的角点坐标：用图像1的初始化图像2的
    }
 
    for (int level = pyramids - 1; level >= 0; level--)//从最顶层开始进行光流追踪
    {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();//开始计时
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        //has_initial设置为true，表示图像2中的角点kp2_pyr进行了初始化
        t2 = chrono::steady_clock::now();//计时结束
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;//输出光流跟踪耗时
 
        if (level > 0) 
        {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;//pyramidScale等于0.5，相当于乘了2
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;//pyramidScale等于0.5，相当于乘了2
        }
    }
 
    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);//存输出kp2
}