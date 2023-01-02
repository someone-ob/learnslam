#pragma once

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>
using namespace std;
using namespace Eigen;

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);
