#include "FindFeatureMatches.h"

void find_feature_matches(const Mat &img_1, const Mat &img_2,vector<KeyPoint> &keypoints_1,vector<KeyPoint> &keypoints_2,vector<DMatch> &matches){
    Ptr<FeatureDetector> detector=ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Mat descriptors_1, descriptors_2;//描述子
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);
    descriptor->compute(img_1,keypoints_1,descriptors_1);
    descriptor->compute(img_2,keypoints_2,descriptors_2);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//特征匹配
    vector<DMatch> matches_1;
    matcher->match(descriptors_1,descriptors_2,matches_1);
    //-- 第四步:匹配点对筛选
  double min_dist = 150, max_dist = 0; //求最小距离
  //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
  for (int i = 0; i < descriptors_1.rows; i++) 
  {
    double dist = matches_1[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++)
  {
     if (matches_1[i].distance <= max(2 * min_dist, 30.0)){
      matches.push_back(matches_1[i]);
     }
  }
}