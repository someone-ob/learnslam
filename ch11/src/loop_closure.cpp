#include "DBoW3/DBoW3.h"//词袋支持头文件
#include <opencv2/core/core.hpp>//opencv核心模块
#include <opencv2/highgui/highgui.hpp>//gui模块
#include <opencv2/features2d/features2d.hpp>//特征点头文件
#include <iostream>
#include <vector>
#include <string>
 
using namespace cv;
using namespace std;
 
/***************************************************
 * 本节演示了如何根据前面训练的字典计算相似性评分
 * ************************************************/
int main(int argc, char **argv) {
    // read the images and database(读取图像和数据库)  
    cout << "reading database" << endl;//输出reading database(读取数据)
    DBoW3::Vocabulary vocab("/home/jzh/Code/learnslam/ch11/build/vocabulary.yml.gz");//vocabulary.yml.gz路径
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want: 
    if (vocab.empty()) {
        cerr << "Vocabulary does not exist." << endl;//输出Vocabulary does not exist
        return 1;
    }
    cout << "reading images... " << endl;//输出reading images...
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "../data/" + to_string(i + 1) + ".png";//图像读取路径
        images.push_back(imread(path));
    }
 
    // 这里我们用它们生成的字典比较它们本身的相似性，这可能会产生过拟合
    // detect ORB features
    cout << "detecting ORB features ... " << endl;//输出detecting ORB features ...(正在检测ORB特征)
    Ptr<Feature2D> detector = ORB::create();//默认图像500个特征点
    vector<Mat> descriptors;//描述子  将10张图像提取ORB特征并存放在vector容器里
    for (Mat &image:images) {
        vector<KeyPoint> keypoints;//关键点
        Mat descriptor;//描述子
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);//检测和计算
        descriptors.push_back(descriptor);
    }
 
    // we can compare the images directly or we can compare one image to a database 
    // images :
    cout << "comparing images with images " << endl;//输出comparing images with images
    for (int i = 0; i < images.size(); i++) 
    {
        DBoW3::BowVector v1;
        //descriptors[i]表示图像i中所有的ORB描述子集合，函数transform()计算出用先前字典来描述的单词向量，每个向量中元素的值要么是0，表示图像i中没有这个单词；要么是该单词的权重
        //BoW描述向量中含有每个单词的ID和权重，两者构成了整个稀疏的向量
        //当比较两个向量时，DBoW3会为我们计算一个分数
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); j++) 
        {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);//p296式(11.9)
            cout << "image " << i << " vs image " << j << " : " << score << endl;//输出一幅图像与另外一幅图像之间的相似度评分
        }
        cout << endl;
    }
 
    // or with database 
    //在进行数据库查询时，DBoW对上面的分数进行排序，给出最相似的结果
    cout << "comparing images with database " << endl;
    DBoW3::Database db(vocab, false, 0);
    for (int i = 0; i < descriptors.size(); i++) 
        db.add(descriptors[i]);
    cout << "database info: " << db << endl;//输出database info(数据库信息)为
    for (int i = 0; i < descriptors.size(); i++) 
    {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4);      // max result=4
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "done." << endl;
}