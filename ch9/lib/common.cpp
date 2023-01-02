#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

using namespace std;

typedef Eigen::Map<Eigen::VectorXd> VectorRef;//map(指向数据的指针，构造矩阵的行数和列数),map相当于引用普通的c++数组，进行矩阵操作，而不用copy数据
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    //int fscanf(文件指针，格式字符串，输入列表),功 能: 从一个流中执行格式化输入,fscanf遇到空格和换行时结束。
    //将fptr指向的文件数据传入value数组。返回值：整型，数值等于*value的个数
    //在读取稳健数据时使用了一个fscanf函数，他的特点是在完成读取后文件指针自动向下移，这就能解释整个过程是如何读取文件内所有数据的。
    int num_scanned = fscanf(fptr, format, value); 
    if (num_scanned != 1)
        std::cerr << "Invalid UW data file. ";
}

void PerturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; ++i)
        point[i] += RandNormal() * sigma;
}

double Median(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}
//  所读取的problem-16-22106-pre.txt文件的内容包括：
//（1）先读取前三个数：分别是相机视角数量、特征点数量、观测结果数量，也就是16张图像，共提取到22106个特征点，这些特征点共出现了83718次（同一特征点在多个视角下重复出现）。
//（2）观测结果：每行四个数据包括图像编号（相机视角编号）、特征点编号、像素坐标。例如“10 2     5.826200e+02 3.637200e+02”就表明2号特征点在10号图像内的成像坐标为(582.6,363.7)。
//（3）最后是相机参数及路标点世界坐标：每个视角相机参数共有9个，3轴旋转角度、3轴平移向量、焦距、2个畸变系数，如果转化为四元数表达旋转，就对应10个参数。每个路标点世界坐标包括三个参数。
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {
    FILE *fptr = fopen(filename.c_str(), "r");//打开只读文件

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    };

    // This wil die horribly on invalid files. Them's the breaks.
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    std::cout << "Header: " << num_cameras_  //16
              << " " << num_points_//22106
              << " " << num_observations_;//83718

    point_index_ = new int[num_observations_];// 每个observation对应的point index,即“10 2     5.826200e+02 3.637200e+02”中的第二个数据
    camera_index_ = new int[num_observations_];// 每个observation对应的camera index，即“10 2     5.826200e+02 3.637200e+02”中的第一个数据
    observations_ = new double[2 * num_observations_];//观测坐标(x,y),即“10 2     5.826200e+02 3.637200e+02”中的后两个数据

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternions;
    if (use_quaternions) {
        // Switch the angle-axis rotations to quaternions.
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_];
        double *original_cursor = parameters_;
        double *quaternion_cursor = quaternion_parameters;
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion(original_cursor, quaternion_cursor);
            quaternion_cursor += 4;
            original_cursor += 3;
            for (int j = 4; j < 10; ++j) {
                *quaternion_cursor++ = *original_cursor++;
            }
        }
        // Copy the rest of the points.
        for (int i = 0; i < 3 * num_points_; ++i) {
            *quaternion_cursor++ = *original_cursor++;
        }
        // Swap in the quaternion parameters.
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}

void BALProblem::WriteToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[9];
        if (use_quaternions_) {
            //OutPut in angle-axis format.
            QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
        } else {
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }
        for (int j = 0; j < 9; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string &filename) const {
    std::ofstream of(filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras(); ++i) {
        const double *camera = cameras() + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }
    of.close();
}

//C to AC功能：已知camera，取出相机旋转，相机中心信息
void BALProblem::CameraToAngelAxisAndCenter(const double *camera,double *angle_axis, double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        QuaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }
    // c = -R't
    //对平移量t进行旋转
    /*如何计算相机中心center的世界坐标
        PW_center:世界坐标系下的相机坐标
        PC_center:相机坐标系下的相机原点坐标（0,0,0）
        根据相机坐标系与世界坐标系的转换关系：PW_center×R+t=PC_center
        PW_center= -R't
     */
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;//角轴添加负号代表旋转方向相反,即为逆
    AngleAxisRotatePoint(inverse_rotation.data(),camera + camera_block_size() - 6,center);
    //camera：旋转3/4维，平移3维，内参3维，需要t，所以大小-6
    //camara和camara_block_size在common.h，参数parameter起始位，10或9
    VectorRef(center, 3) *= -1.0;
}

//与上面方向相反 已知相机旋转，相机中心空间坐标，获得camera参数
void BALProblem::AngleAxisAndCenterToCamera(const double *angle_axis,const double *center,double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        AngleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c
    AngleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}

void BALProblem::Normalize() {
    // Compute the marginal median of the geometry
    //归一化步骤：
    /*
    ①计算所有世界坐标点X/Y/Z的中位数median
    ②将所有点的X/Y/Z减去对应中位数，取绝对值，得到新的tmp数组
    ③取tmp数组的中位数median_absolute_deviation
    ④得到缩放参数scale = 100.0 / median_absolute_deviation
    ⑤所有点减去median乘以scale，得到新的世界坐标数组
    */
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;//三维向量，世界坐标包括三个参数x,y,z，则该向量X/Y/Z中位数。
    double *points = mutable_points();//所有世界坐标三个参数
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = Median(&tmp);//计算X/Y/Z中位数
    }

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);//每个点的x,y,z坐标
        tmp[i] = (point - median).lpNorm<1>();//lpNorm<1>即取L1范数，即绝对值的和。
    }

    const double median_absolute_deviation = Median(&tmp);

    // Scale so that the median absolute deviation of the resulting
    // reconstruction is 100
    const double scale = 100.0 / median_absolute_deviation;

    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i;
        CameraToAngelAxisAndCenter(camera, angle_axis, center);//求得每个相机视角的相机中心
        // center = scale * (center - median)
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);//相机中心点的归一化
        AngleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

void BALProblem::Perturb(const double rotation_sigma,const double translation_sigma,const double point_sigma) {
    assert(point_sigma >= 0.0);
    assert(rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    double *points = mutable_points();
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i;

        double angle_axis[3];
        double center[3];
        //先通过相机的旋转平移参数得到相机中心坐标
        //给角轴添加噪声
        //再根据角轴和中心坐标计算出新的平移参数
        //再给新的平移参数添加噪声
        CameraToAngelAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            PerturbPoint3(rotation_sigma, angle_axis);
        }
        AngleAxisAndCenterToCamera(angle_axis, center, camera);

        if (translation_sigma > 0.0)
            PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
}