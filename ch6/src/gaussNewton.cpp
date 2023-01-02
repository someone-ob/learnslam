#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include<Eigen/Dense>
#include<chrono>

using namespace std;
using namespace Eigen;

int main(int argc,char **agrv){
    double ar=1,br=2,cr=1;  //真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值,并赋初始值
     int N=100;    //数据点数
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器 RNG为OpenCV中生成随机数的类，全称是Random Number Generator

    vector<double> x_data, y_data;      // double数据x_data, y_data
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;//相当于x范围是0-1
        x_data.push_back(x);//x_data存储的数值
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));//rng.gaussian(w_sigma * w_sigma)为opencv随机数产生高斯噪声
        //rng.gaussian(val)表示生成一个服从均值为0，标准差为val的高斯分布的随机数  视觉slam十四讲p133式6.38上面的表达式
     }
    // 开始Gauss-Newton迭代 求ae，be和ce的值，使得代价最小
     int iterations = 100;    // 迭代次数
    double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost  cost表示本次迭代的代价，lastCost表示上次迭代的代价
    //cost = error * error，error表示测量方程的残差

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//std::chrono是c++11引入的日期处理库，其中包含三种时钟（system_clock,steady_clock,high_resolution_clock）
     //t1表示steady_clock::time_point类型
    for (int i=0;i<iterations;i++){
        Matrix3d H=Matrix3d::Zero();        //创建近似海塞矩阵，初始值为零
        Vector3d g=Vector3d::Zero();        
        cost=0;
        for (int j=0;j<N;j++){
            double e=y_data[j]-exp(ae * x_data[j] * x_data[j]+ be* x_data[j] + ce) ;
            Vector3d J;        //创建雅克比矩阵
            J[0]=-x_data[j] * x_data[j]*exp(ae * x_data[j] * x_data[j]+ be* x_data[j] + ce) ;
            J[1]=-x_data[j]*exp(ae* x_data[j] * x_data[j]+ be * x_data[j] + ce) ;
            J[2]=-exp(ae * x_data[j] * x_data[j]+ be * x_data[j] + ce) ;
            H+=inv_sigma*inv_sigma*J*J.transpose();
            g+=-inv_sigma*inv_sigma*e*J;
            cost+=e*e;
        }
        // 求解线性方程 Hx=b
        Vector3d dx = H.ldlt().solve(g); //ldlt()表示利用Cholesky分解求dx
        if (isnan(dx[0]))//isnan()函数判断输入是否为非数字，是非数字返回真，nan全称为not a number
        {
            cout << "result is nan!" << endl;
            break;
        }
        if (i>0&&cost>=lastCost){
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }
        ae+=dx[0];
        be+=dx[1];
        ce+=dx[2];
        lastCost=cost;
        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<"\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}

