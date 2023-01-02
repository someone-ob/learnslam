#include "bundleAdjustmentGaussNewton.h"

// BA by gauss-newton
void bundleAdjustmentGaussNewton(const VecVector3d &points_3d,const VecVector2d &points_2d,const Mat &K,Sophus::SE3d &pose)//定义bundleAdjustmentGaussNewton函数 输入3d点和2d点和相机内参矩阵，输出pose为优化变量 使用高斯牛顿法
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;//6*1矩阵
    const int iterations=10;
    double cost=0,lastCost=0;
    double fx=K.at<double>(0,0);
    double fy=K.at<double>(1,1);
    double cx=K.at<double>(0,2);
    double cy =K.at<double>(1,2);

    for (int iter=0;iter<iterations;iter++){
        Matrix<double,6,6> H=Matrix<double,6,6>::Zero();
        Vector6d b =Vector6d::Zero();
        cost = 0;//代价函数值置为0
        // compute cost
        for (int i = 0; i < points_3d.size(); i++)//遍历3d点
        //计算增量方程中的H和b以及代价函数值
        {
            Eigen::Vector3d pc = pose * points_3d[i];//P' = TP =[X',Y',Z'] 视觉slam十四讲p186式7.38
            double inv_z = 1.0 / pc[2];//相当于1 / Z'
            double inv_z2 = inv_z * inv_z;//相当于( 1 / Z' ) * ( 1 / Z' )
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);//u = fx(X' / z') +cx,v = fy(Y' / z') +cy 视觉slam十四讲p186式7.41
            Eigen::Vector2d e = points_2d[i] - proj;
            //squaredNorm()是求矩阵/向量所有元素的平方和
            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;//2*6雅克比矩阵 视觉slam十四讲p186式7.46
            //雅克比矩阵表达式见 视觉slam十四讲p186式7.46
            J << -fx * inv_z,
            0,
            fx * pc[0] * inv_z2,
            fx * pc[0] * pc[1] * inv_z2,
            -fx - fx * pc[0] * pc[0] * inv_z2,
            fx * pc[1] * inv_z,
            0,
            -fy * inv_z,
            fy * pc[1] * inv_z2,
            fy + fy * pc[1] * pc[1] * inv_z2,
            -fy * pc[0] * pc[1] * inv_z2,
            -fy * pc[0] * inv_z;
 
            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        // -fx * inv_z 相当于-fx / Z'
        //0
        //fx * pc[0] * inv_z2相当于fx * X' / ( Z' * Z' )
        //-fx - fx * pc[0] * pc[0] * inv_z2相当于fx * X' * Y' / ( Z' * Z')
        //fx * pc[1] * inv_z相当于fx * Y' / Z'
        //0
        //-fy * inv_z相当于-fy / Z'
        //fy * pc[1] * inv_z2相当于fy * Y' / (Z' * Z')
        //fy + fy * pc[1] * pc[1] * inv_z2相当于fy + fy * Y'*Y' / (Z' * Z')
        //-fy * pc[0] * pc[1] * inv_z2相当于fy * X' * Y' / ( Z' * Z')
        //-fy * pc[0] * inv_z相当于-fy * X' / Z'
        Vector6d dx;
        dx=H.ldlt().solve(b);
        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;//输出代价
            break;
        }
        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;
        cout << "iteration " << iter << " cost=" << setprecision(12) << cost << endl;//输出迭代次数和代价
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }
  cout << "pose by g-n: \n" << pose.matrix() << endl;//输出pose by g-n
}