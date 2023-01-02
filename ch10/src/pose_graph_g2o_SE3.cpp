#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 本节使用李代数表达位姿图，节点和边的方式为自定义
 * 利用 g2o对sphere.g2o文件进行优化 优化前 用g20——viewer显示为椭球
* 用g2o的话 需要定义顶点和边
* 位姿图优化就是只优化位姿 不优化路标点
* 顶点应该相机的位姿
* 边是相邻两个位姿的变换
* error误差是观测的相邻相机的位姿变换的逆 
* 待优化的相邻相机的位姿变换
* 我们希望这个误差接近I矩阵 给误差取ln后 误差接近 0 
* 该程序用李代数描述误差

* 这里把J矩阵的计算放在JRInv(const SE3d & e)函数里
* 这里的J矩阵还不是雅克比矩阵 具体雅克比见书上公式 p272页 公式10.9 10.10
* 李代数应该是向量形式 
* 李代数的hat 也就是李代数向量变为反对称矩阵

 * **********************************************/

typedef Matrix<double, 6, 6> Matrix6d;

// 给定误差求J_R^{-1}的近似
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log());
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation());
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());
    // J = J * 0.5 + Matrix6d::Identity();
    J = Matrix6d::Identity();    // try Identity if you want
    return J;
}

// 李代数顶点
typedef Matrix<double, 6, 1> Vector6d;

class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        //读取数据
    virtual bool read(istream &is) override {
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];
        setEstimate(SE3d(
            Quaterniond(data[6], data[3], data[4], data[5]),
            Vector3d(data[0], data[1], data[2])
        ));
    }
    //将优化的位姿存入内存 
    virtual bool write(ostream &os) const override {
        os << id() << " ";
        Quaterniond q = _estimate.unit_quaternion();
        os << _estimate.translation().transpose() << " ";
        //coeffs顺序是 x y z w ,w是实部
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl;
        return true;
    }

    virtual void setToOriginImpl() override {
        _estimate = SE3d();//李代数
    }

    // 左乘更新
    virtual void oplusImpl(const double *update) override {
        Vector6d upd;//六维向量 upd接收 update
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate;//更新位姿
    }
};

// 两个李代数节点之边
// 定义边  两个李代数顶点的边 边就是两个顶点之间的变换 即位姿之间的变换
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //读取观测值和构造信息矩阵
    virtual bool read(istream &is) override {
        //这里观测值是位子之间的变换，当然包括旋转和平移 所以 data[]是7维 平移加四元数
        double data[7];
        for (int i = 0; i < 7; i++)
            is >> data[i];//流入data[]
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize();//归一化
        setMeasurement(SE3d(q, Vector3d(data[0], data[1], data[2])));
        for (int i = 0; i < information().rows() && is.good(); i++)
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)     //不是对角线的地方
                    information()(j, i) = information()(i, j);
            }
        return true;
    }
    //这个函数就是为了把优化好的相机位姿放进指定文件中去
    virtual bool write(ostream &os) const override {
        //v1,V2分别指向两个顶点
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]);
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]);
        os << v1->id() << " " << v2->id() << " ";   //把两个定点的编号流入os
        SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();     //获取单位四元数
        //先传入平移  再传入四元数
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix   信息矩阵
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    // 误差计算与书中推导一致
    virtual void computeError() override {
        //v1,V2分别指向两顶点的位姿
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    // 雅可比计算
    virtual void linearizeOplus() override {
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));          //计算d
        // 尝试把J近似为I
        //雅克比有两个,一个是误差对相机i位姿的雅克比,另一个是误差对相机j位姿的雅克比
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
    //     return 1;
    // }
    //将sphere.g2o文件流入fin
    ifstream fin("/home/jzh/Code/learnslam/ch10/sphere.g2o");
    // if (!fin) {
    //     cout << "file " << argv[1] << " does not exist." << endl;
    //     return 1;
    // }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType; //6,6是顶点和边的维度
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;   //线性求解
    //设置梯度下降的方法
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    //容器 vectices 和edges 存放各个顶点和边
    vector<VertexSE3LieAlgebra *> vectices;
    vector<EdgeSE3LieAlgebra *> edges;
    
    while (!fin.eof()) {
        string name;
        fin >> name;
        //将文件中的顶点数据流入,顶点就是各个相机的位姿
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);   //这里是setEstimate
            optimizer.addVertex(v);
            vertexCnt++;
            vectices.push_back(v);
            if (index == 0)
                v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2;     // 关联的两个顶点
            fin >> idx1 >> idx2;            //顶点的ID
            e->setId(edgeCnt++);        ///设置边的ID
            //设置顶点
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);       //读取观测值
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }
            //输出边的顶点的合的个数
    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();     //优化初始化
    optimizer.optimize(30);         //迭代次数

    cout << "saving optimization results ..." << endl;

    // 因为用了自定义顶点且没有向g2o注册，这里保存自己来实现
    // 伪装成 SE3 顶点和边，让 g2o_viewer 可以认出
    ofstream fout("result_lie.g2o");
    for (VertexSE3LieAlgebra *v:vectices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);         //把优化的顶点放进 result_lie.g2o
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout); //把优化的边放进 result_lie.g2o
    }
    fout.close();
    return 0;
}


