#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    BALProblem bal_problem("/home/jzh/Code/learnslam/ch9/problem-16-22106-pre.txt");//读取文件数据，存储需要的变量
    bal_problem.Normalize();//将路标点的坐标和每个相机的中心坐标归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);//添加噪声，具体细节见注释
    bal_problem.WriteToPLYFile("initial.ply");//再initial.ply文件中写入相机中心坐标和路标点世界坐标
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();//坐标参数数目(3)
    const int camera_block_size = bal_problem.camera_block_size();//相机参数数目(9/10)
    double *points = bal_problem.mutable_points();//路标点世界坐标
    double *cameras = bal_problem.mutable_cameras();//相机参数和世界坐标参数

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x and y position of the observation.
    const double *observations = bal_problem.observations();//观测点相机坐标(x,y)
    ceres::Problem problem;//定义一个优化问题类problem

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        //每个残差块以一个点和一个相机作为输入并输出一个二维残差
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];//每个相机对应的相机参数
        double *point = points + point_block_size * bal_problem.point_index()[i];//每个像素坐标对应的路标点世界坐标

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}