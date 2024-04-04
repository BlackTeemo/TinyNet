//
//  Operation_File.cpp
//  TinyNet
//
//  Created by mac on 2021/10/31.
//

#include "Operation_File.hpp"


void Read_File::Split(string & Str,char &item,vector<vector<float>> &Data){    //分割字符串将浮点类型存入二维向量Data
    int Size = (int)Str.size() ;
    int Left = 0, Right = 0 ;  //左右指针
    vector<float>Temp ; //临时一维浮点类型向量
    while( Left < Size ){
        Right = Left ;
        while(Right < Size && Str[Right] != item )
            Right++;
        
        Temp.emplace_back(atof(Str.substr(Left,Right-Left).c_str())) ;
        
        Left = Right ;
        while(Left < Size && Str[Left] == item)
            Left++ ;
    }
    Data.emplace_back(Temp) ;
}


Mat_<float> Read_File::Read_csv(string &Path,int Flag ,vector<vector<float>> *meanStd ){       //读取csv格式文件
    fstream f(Path.c_str(),ios::in);
    if( !f )
        return Mat_<float>(0,0) ;
    vector<vector<float>> Data ;

    
    while(getline(f,Line))
        Read_File::Split(Line, Csv_Separator, Data) ;
        
    Mat_<float> RetMat(Data.size(),Data[0].size()) ;  //创建矩阵
    
    for( int i = 0 ;i < Data.size(); ++i){
        for( int j = 0; j < Data[0].size(); ++j)
            RetMat(i,j) = Data[i][j] ;
    }  //矩阵赋值
    
    f.close() ; //关闭文件流
    if(Flag)
        Normalized_feature_scaling(RetMat,meanStd) ;   //标准化特征缩放处理
    
    return RetMat ;
    
}


void Read_File::Normalized_feature_scaling(Mat_<float> &Target,vector<vector<float>> *meanStd){  //标准化特征缩放函数
    int ColSize = Target.cols ; // 获得列数
    Mat_<float> ColMat ;
    Mat_<double>  Mean ,Stddev;
    for( int i = 0 ; i <  ColSize-1 ; ++i ){
        ColMat  =  Target.colRange(i, i+1) ;     //获得列矩阵
        meanStdDev(ColMat,Mean ,Stddev) ;       //获得该列的均值和标准差
        ColMat -= Mean(0);                  //让该列所有元素减去平均值

        ColMat /= Stddev(0) ;//让改列所有元素除以标准差

        vector<float> MeStd = {(float)Mean(0),(float)Stddev(0)} ;
        meanStd->emplace_back(MeStd);  //更新基类的meanStd
        
    }
}

void Read_File::Last_column_optimization(Mat_<float> &Data){ //将数据集矩阵的的最后一列转为负号 优化相关算法
    int Last_column = Data.cols-1 ;
    for( int i = 0; i < Data.rows; ++i){
        Data(i,Last_column) = -Data(i,Last_column) ; //变负处理
    }
}

Mat_<float> Read_File::Difference(Mat_<float> &Prediction_parameters,Mat_<float> &Train_Data ,Mat_<float> Real_Data ){ //将预测参数与训练数据运算
    
    Mat_<float> Prediction_Result = Train_Data * Prediction_parameters ;
    
    return Prediction_Result - Real_Data ; 
}
