//
//  TinyNet.cpp
//  TinyNet
//
//  Created by mac on 2021/10/31.
//

#include "TinyNet.hpp"

void TinyNet::SetLearn_α_Time(float α ,int MaxTime){ //设置学习率 学习最大次数
    this->α = α ;
    this->MaxTime = MaxTime ;
}

void TinyNet::Update_parameters(float Sum){ //更新参数
    Prediction_parameters(0,0) -= Sum* abs(Prediction_parameters(0)) ;
    for( int i = 1; i < Prediction_parameters.cols; ++i)
        Prediction_parameters(0,i) =  Prediction_parameters(0,i) -  Sum * abs(Prediction_parameters(0,i)) ;
}

Mat_<float> TinyNet::gradient(Mat_<float> (*Processing_function)(Mat_<float> Value)){// 获得梯度
    
//    Mat_<float> Prediction_theta = Prediction_parameters.rowRange(1, Prediction_parameters.rows); //取第一行到最最后一行的向量
    
    if(Processing_function)
        return Train_Data.t() * ((*Processing_function)( Handle_file.Difference(Prediction_parameters, Train_Data,Real_Data))) / Train_Data.rows  ;
    return Train_Data.t() * ( Handle_file.Difference(Prediction_parameters, Train_Data,Real_Data)) / Train_Data.rows;
}

void TinyNet::Feature_scaling(Mat_<float> &PreVal){ //对于输入的值(要预测的参数)做归一化处理
    for( int i = 0; i < PreVal.rows; ++i){
        PreVal(i,0) = (PreVal(i,0) - meanStd[i][0]) / meanStd[i][1] ;
    }
}

bool TinyNet::IsTruePath(Mat_<float> &Data){ //通过返回的数据矩阵 判断是否为正确的路径
    if( Data.cols == 0 || Data.rows == 0){   //路径不存在
        cout<<Path<<"路径文件不存在"<<endl;
        return true;
    }
    return false ;
}

void TinyNet::Feature_Mapping(Mat_<float> &Data){  //特征映射 处理非线形问题 (暂时处理两个变量的 以后继续维护)
    Mat_<float>X1 =  Data.rowRange(0, 1).clone();
    Mat_<float>X2 =  Data.rowRange(1, 2).clone() ;
    Mat_<float>TempX1 ,TempX2 ;
    for( int i = 2; i <= 6; ++i){
        for( int j = i; j >= 0 ; --j){
            pow(X1,j,TempX1);
            pow(X1,i - j,TempX2);
            hconcat(Data, TempX1.mul(TempX2), Data) ;
        }
    }
}
