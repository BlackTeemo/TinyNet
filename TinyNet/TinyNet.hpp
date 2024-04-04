//  TinyNet.hpp
//  TinyNet
//  Created by Teemo on 2021/10/31.
// A TinyNet Lib


#ifndef TinyNet_hpp
#define TinyNet_hpp




#include"Operation_File.hpp"
#include"Opencv.hpp"

using namespace std;
using namespace cv;

#define Max 100000  //训练最大次数
#define End_of_training 1e-6  //训练结束变化值

class TinyNet{
public:
    
    TinyNet(string Route):Path(Route){}  //构造函数
    void SetLearn_α_Time(float α ,int MaxTime) ; //设置学习率 学习最大次数

protected:   //常量
    
    string Path ;
    float α = 0.2 ; //学习率
    int MaxTime = Max ;
    int isFeature_mapping = 0 ;

protected:  //变量
    vector<vector<float>>  meanStd ; //用来保存归一化处理后 每列的平均值及标准差
    Mat_<float> Train_Data ; //用来保存训练数据
    Mat_<float> Real_Data ; //用来保存文件中结果数据
    Mat_<float> Prediction_parameters ; //用来保存训练参数
    
protected: //对象
    
    Read_File Handle_file; //Read_File对象用来读取文件
    random_device Rand_num ; //定义random_device函数对象获得随机数

protected: //函数

    Mat_<float> gradient(Mat_<float> (*Processing_function)(Mat_<float> Value) = NULL) ; // 获得梯度
    void Update_parameters(float Sum); //更新参数
    void Feature_scaling(Mat_<float> &PreVal) ; //对于输入的值(要预测的参数)做归一化处理
    bool IsTruePath(Mat_<float> &Data); //通过返回的数据矩阵 判断是否为正确的路径
    void Feature_Mapping(Mat_<float> &Data);  //特征映射 处理非线形问题
    
};





#endif /* TinyNet_hpp */
