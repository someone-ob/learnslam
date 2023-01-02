#pragma once

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    //显式函数：显式函数只通过自己的参数和返回值来和外界做数据交换，不会去读取或者修改函数的外部状态
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);//explicit关键字只能用于修饰只有一个参数的类构造函数, 它的作用是表明该构造函数是显示的, 而非隐式的

    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// 存储数据到.txt
    void WriteToFile(const std::string &filename) const;
    /// 存储点云文件
    void WriteToPLYFile(const std::string &filename) const;
    /// 归一化
    void Normalize();
    /// 给数据加噪声
    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);
    /// 相机参数数目：是否使用四元数
    int camera_block_size() const { return use_quaternions_ ? 10 : 9; }
    /// 特征点参数数目：3
    int point_block_size() const { return 3; }
    /// 相机视角数目
    int num_cameras() const { return num_cameras_; }
    /// 关键点数目
    int num_points() const { return num_points_; }
    /// 观测结果数目：多个视角下共观测到多少个特征点（同一特征点多次出现）
    int num_observations() const { return num_observations_; }
    /// 待估计的参数数目
    int num_parameters() const { return num_parameters_; }
    /// 观测结果：特征点编号首地址
    const int *point_index() const { return point_index_; }
    /// 观测结果：相机视角编号首地址
    const int *camera_index() const { return camera_index_; }
    /// 观测结果：观测点相机坐标首地址
    const double *observations() const { return observations_; }
    /// 所有待估计参数首地址（相机位姿参数+特征点坐标）
    const double *parameters() const { return parameters_; }
    /// 相机位姿参数首地址
    const double *cameras() const { return parameters_; }
    /// 特征点参数首地址=相机位姿参数首地址+相机位姿参数维度（9 or 10）*相机个数
    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }
    /// 相机参数和世界坐标参数
    double *mutable_cameras() { return parameters_; }
    //返回每个路标点世界坐标，包括三个参数
    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }
    /// 查找对应信息
    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }//第i个相机参数首地址
 
    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }//第i个特征点首地址
 
    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }
 
    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;// 相机视角数目
    int num_points_;// 关键点数目
    int num_observations_;// 观测结果数目：多个视角下共观测到多少个特征点（同一特征点多次出现）
    int num_parameters_;// 待估计的参数数目
    bool use_quaternions_;//是否使用四元数

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;//观测点相机坐标(x,y)
    double *parameters_;//相机参数及路标点世界坐标
};