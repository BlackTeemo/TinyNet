//
//  Operation_File.hpp
//  TinyNet
//
//  Created by mac on 2021/10/31.
//

#ifndef Operation_File_hpp
#define Operation_File_hpp

#include"Opencv.hpp"
#include<fstream>
#include<string>
#include<vector>
#include<cmath>
#include<random>
#include<iostream>
using namespace std ;
using namespace cv ;


class Read_File{
public:  
    
    Mat_<float> Read_csv(string &Path,int Flag = 0,vector<vector<float>> *meanStd = NULL);    //读取csv格式文件  Flag表示是否进行缩放 meanStd存放meanStd地址
    
    void Normalized_feature_scaling(Mat_<float> &Target,vector<vector<float>> *meanStd) ; //标准化特征缩放函数
    
    void Last_column_optimization(Mat_<float> &Data); //将数据集矩阵的的最后一列转为负号
    
    Mat_<float> Difference(Mat_<float> &Prediction_parameters,Mat_<float> &Train_Data ,Mat_<float> Real_Data ); //将预测参数与训练数据运算 
    
private:
    
    void Split(string & Str,char &item,vector<vector<float>> &Data) ;  //分割函数
    
private:
    
    string Line = "" ;   //每行的数据
    char Csv_Separator = ',';   //csv文件分割符
};




#endif /* Operation_File_hpp */
