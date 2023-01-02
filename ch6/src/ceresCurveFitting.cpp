#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>//ceres库头文件
#include <chrono>
 
using namespace std;

//ceres求解步骤
//定义CostFuntion(代价函数)
//构建Problem
//配置Solver
//定义Summary
//开始优化Solve
//输出结果SUmmary.BriefReport
 
// 代价函数的计算模型
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}//使用初始化列表赋值写法的构造函数
  // 残差的计算
  template<typename T>//函数模板，使得下面定义的函数可以支持多种不同的形参，避免重载函数的函数体重复设计。
  bool operator()(const T *const abc, T *residual) const //abc是模型参数，有3维，重载运算符()，即让()运算变成误差函数f()
    {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c) residual表示残差
    return true;
   //返回bool类型，计算结果已经存入函数外的residual变量中
  }
  const double _x, _y;    // x,y数据 结构体CURVE_FITTING_COST中的成员变量
};
 
int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
  int N = 100;                                 // 数据点
  double w_sigma = 1.0;                        // 噪声Sigma值 初始化为1
  double inv_sigma = 1.0 / w_sigma;            // 标准差的逆
  cv::RNG rng;                                 // OpenCV随机数产生器 RNG为OpenCV中生成随机数的类，全称是Random Number Generator
 
  vector<double> x_data, y_data;      // 数据  // double数据x_data, y_data
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;//相当于x范围是0-1
    x_data.push_back(x);//x_data存储的数值 所给的100个观测点数据
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    //rng.gaussian(w_sigma * w_sigma)为opencv随机数产生高斯噪声
   //rng.gaussian(val)表示生成一个服从均值为0，标准差为val的高斯分布的随机数  视觉slam十四讲p133式6.38上面的表达式
  }
 
  double abc[3] = {ae, be, ce};//定义优化变量
 
  // 构建最小二乘问题
  ceres::Problem problem;//定义一个优化问题类problem
  for (int i = 0; i < N; i++) {
    problem.AddResidualBlock(     // 向问题中添加误差项
      // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
        new CURVE_FITTING_COST(x_data[i], y_data[i])
      ),
      nullptr,            // 核函数，这里不使用，为空  添加损失函数(即鲁棒核函数)，这里不使用，为空
      abc                 // 待估计参数 优化变量，3维数组
    );
  }
 
  // 配置求解器
  ceres::Solver::Options options;     // 这里有很多配置项可以填 定义一个配置项集合类options
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; 
  // 增量方程如何求解 增量方程求解方式，本质上是稠密矩阵求逆的加速方法选择
  options.minimizer_progress_to_stdout = true;   
  // 输出到cout minimizer_progress_to_stdout表示是否向终端输出优化过程信息
 
  ceres::Solver::Summary summary; // 优化信息 利用ceres执行优化
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);  // 开始优化
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
 
  // 输出结果
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";//输出估计值
  for (auto a:abc) cout << a << " ";
  cout << endl;
 
  return 0;
}
 
