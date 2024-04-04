//
//  Learning_algorithm.hpp
//  TinyNet
//
//  Created by mac on 2021/11/9.
//
//
#pragma once

#ifndef Learning_algorithm_hpp
#define Learning_algorithm_hpp

#include"TinyNet.hpp"

#endif /* Learning_algorithm_hpp */

//线性梯度算法
class Linear_Regression:public TinyNet{
public:
    Linear_Regression(string Route):TinyNet(Route){} //调用TinyNet基类的构造函数
    
    void Gradient_Descent_train();  //多元梯度下降训练
    void Normal_Equation_train();   //正规方程训练
    
    float Linear_predict(Mat_<float> &PreVal) ; //预测函数
};

//分类算法
class Logistic_simulation:public TinyNet{
public:
    Logistic_simulation(string Route):TinyNet(Route){} //调用TinyNet基类的构造函数
    
    void logistic_train(int Normalized_feature_scaling = 1,int Feature_Mapping = 0);  //训练函数 Normalized_feature_scaling 代表是否进行特征缩放 Feature_Mapping表示是否进行特征映射     默认进行特征缩放 不进行特征映射
    float logistic_predict(Mat_<float> &PreVal) ; //预测函数
    
private:
    
    static Mat_<float> sigmoid(Mat_<float> Value) ;
    
};
