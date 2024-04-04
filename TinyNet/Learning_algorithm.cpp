//
//  Learning_algorithm.cpp
//  TinyNet
//
//  Created by mac on 2021/11/9.
//

#include "Learning_algorithm.hpp"

//void Linear_Regression::Gradient_Descent_train(){  //多元梯度下降模拟
//    Train_Data = Handle_file.Read_csv(Path,1,&meanStd) ;   //开始读取目标csv文件
//
//    if( IsTruePath(Train_Data))  //路径不存在
//        return ;
//
//    int Time = 0 ; //训练次数
//
//    Prediction_parameters = Mat_<float>(1,Train_Data.cols); //生成一个1 * 数据列数 的矩阵
//
//    for( int i = 0; i < Prediction_parameters.cols;++i)
//        Prediction_parameters(i) = (float)(Rand_num() % 7 + 1);
//
//    Mat_<float> transformation_Data = Train_Data.colRange(0, Train_Data.cols - 1).clone().t() ;
//
//    float Sumc = sum(Train_Data.colRange( Train_Data.cols-1,Train_Data.cols).clone())[0] ;  //获得样本结果和
//
//    Mat_<float> Last_parameters ; //保存上一份Var
//
//    while (Time++ < MaxTime) {
//        Last_parameters = Prediction_parameters ;
//
//        float Average_error = (sum(Handle_file.Difference(Prediction_parameters, transformation_Data))[0] - Sumc) / Train_Data.rows; //平均误差
//
//        Update_parameters(Average_error * α); //更新参数
//
//        if( abs(cv::norm(Prediction_parameters) - cv::norm(Last_parameters)) <  End_of_training && Average_error < 10 )  //判断是否继续学习
//            break ;
//    }
//}

void Linear_Regression::Normal_Equation_train(){   //正规方程训练

    Train_Data = Handle_file.Read_csv(Path,0) ;   //开始读取目标csv文件 Flag赋值为0 不需要做归一化处理
    
    if( IsTruePath(Train_Data))  //路径不存在
        return ;
    
    
    //正规方程计算
    Mat_<float> X ;
    Mat_<float> Y = Train_Data.colRange(Train_Data.cols-1, Train_Data.cols).clone() ; // 得到正确数据列  构造Y向量
    
    Mat_<float> X0(Train_Data.rows,1,1) ;
    Mat_<float> X1 = Train_Data.colRange(0, Train_Data.cols-1).clone();
    cv::hconcat(X0, X1, X) ;  //左右拼接X0 X1矩阵得到X
    
    Mat_<float> Ret ;
    invert(X.t()*X,Ret);
    Prediction_parameters = Ret * X.t() * Y ;
    
    
    
}
float Linear_Regression::Linear_predict(Mat_<float> &PreVal){ //预测函数
    if(meanStd.size() >  0 )  //  如果训练时对数据进行归一化处理 则对输入参数进行归一化处理
        Feature_scaling(PreVal) ;
    cout<<Prediction_parameters(1)<<"X + "<<Prediction_parameters(2)<<"Y + " <<Prediction_parameters(0)<<endl;
    float Res = 0 ;
    for( int i = 1; i < Prediction_parameters.cols ;++i)
        Res += PreVal(0,i-1) * Prediction_parameters(0,i) ;
    
    return Res + Prediction_parameters(0,0) ;
}


void Logistic_simulation::logistic_train(int Normalized_feature_scaling ,int Feature_Mapping ){
    Train_Data = Handle_file.Read_csv(Path,Normalized_feature_scaling,&meanStd);
    
    if( IsTruePath(Train_Data))
        return ;
        
    Real_Data = Train_Data.colRange(Train_Data.cols - 1 , Train_Data.cols); //获得结果列
    
    Train_Data = Train_Data.colRange(0, Train_Data.cols - 1).clone() ;  //获得特征列
    
    //Mat_<float> temp(Train_Data.rows,1,(float)1) ;
    
    hconcat(Mat_<float>(Train_Data.rows,1,(float)1),Train_Data,Train_Data); //插入一列一
    
    if(Feature_Mapping){                           //进行特征映射处理非线形问题
        isFeature_mapping = Feature_Mapping ;
        this->Feature_Mapping(Train_Data);
    }
    

    Prediction_parameters = Mat_<float>(Train_Data.cols  , 1 ,(float)0);  //构建预测参数向量 初始化为0
    

    int Time = 0 ; //训练次数

    while(Time++ < MaxTime){
        Mat_<float> Last_parameters = Prediction_parameters ;
        
        //Mat_<float> After_Operate  = transformation_Data.t() * (sigmoid( Handle_file.Difference_sum(Prediction_parameters, transformation_Data,1)) - Real_Predict);
        Mat_<float> Gradient = gradient(sigmoid);

        Prediction_parameters -= α * Gradient ;
  

        if(  abs(cv::norm(Prediction_parameters) - cv::norm(Last_parameters)) <  End_of_training )  //判断是否继续学习
            break ;
    }
}

float Logistic_simulation::logistic_predict(Mat_<float> &PreVal){ //预测函数
    if(meanStd.size() >  0 )  //  如果训练时对数据进行归一化处理 则对输入参数进行归一化处理
        Feature_scaling(PreVal) ;

    if(isFeature_mapping)
        this->Feature_Mapping(PreVal);
    PreVal = PreVal.t();

//    for( int i = 0; i < Prediction_parameters.rows;++i)
//        cout<<Prediction_parameters(i)<<" ";
//    cout<<endl;
    
    
    return sigmoid(PreVal*Prediction_parameters)(0,0) ;
}

Mat_<float> Logistic_simulation::sigmoid(Mat_<float> Value){ //Mat_<float> *Processing function(Mat_<float> Value)
    Mat_<float> Ex ;
    exp(-Value,Ex);
    Ex += 1 ;
    return 1 / Ex ;
}

