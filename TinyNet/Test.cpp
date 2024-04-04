//
//  main.cpp
//  TinyNet
//
//  Created by mac on 2021/10/31.
//

#include "TinyNet.hpp"
#include"Operation_File.hpp"
#include"Learning_algorithm.hpp"
void Test(){
    
//    TinyNet it(string("/Users/mac/Documents/TinyNet/Train_Data/Data2.csv")) ;
//
//    Mat_<float> mat(1,3) ;
//    mat(0) = 0.7 ;
//    mat(1)= 1.2 ;
//    mat(2) = 4 ;
//
//    it.Gradient_Descent_simulation(mat, "/Users/mac/Documents/TinyNet/Train_Data/Data2.csv");
//    cout<<mat(1)<<"x+ "<<mat(2)<<"y"<<" + "<<mat(0)<<endl;
//    cout<<2000*mat(1) + 3*mat(2) + mat(0)<<endl;
    
//    Mat_<float> Mt =  it.Normal_Equation_simulation("/Users/mac/Documents/TinyNet/Train_Data/Data2.csv");
//    
//    cout<<"Gradient_Descent"<<endl;
//    cout<<mat(1)<<" x+ "<<mat(2)<<"y"<<" + "<<mat(0)<<endl;
//    cout<<"Normal_Equation"<<endl;
//    cout<<Mt(1)<<" x+ "<<Mt(2)<<"y"<<" + "<<Mt(0)<<endl;
    Mat_<float> Val = {1,-1,-1} ;
    cout<<Val<<endl;;

    Logistic_simulation it(string("/Users/mac/Documents/TinyNet/Train_Data/Data4.csv")) ;

    it.SetLearn_α_Time(0.1, 10000) ;
    it.logistic_train(0,0); 
    cout<<it.logistic_predict(Val) ;

//    Mat_<float> Val = {4215,4} ;
//    Val = Val.t();
//    Linear_Regression it(string("/Users/mac/Documents/TinyNet/Train_Data/Data2.csv"));
//    it.SetLearn_α_Time(0.1, 1000000);
//    it.Gradient_Descent_train();
//    cout<<it.Linear_predict(Val)<<endl;
    

}





int main(){
    
    Test() ;
    
    return 0 ;
}
