/*!
\file
\brief Simple network to learn XOR, using TensorFlow C++ APIs.
       Build TensorFlow .so files: https://github.com/FloopCZ/tensorflow_cc
       C++ TensorFlow example: https://matrices.io/training-a-deep-neural-network-using-only-tensorflow-c
       Related blog article: https://tecfront.blogspot.com/2019/04/the-treasured-or-dnn-in-tensorflow-c.html
*/

//c++ core libs
#include <iostream>
#include <vector>

//tensorflow libs
#include <tensorflow/cc/ops/standard_ops.h>         //operations
#include <tensorflow/cc/framework/gradients.h>      //optimisers
#include <tensorflow/core/framework/tensor.h>       //data
#include <tensorflow/core/framework/tensor_shape.h> //data
#include <tensorflow/cc/client/client_session.h>    //run

//namespaces
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

//shortcuts
namespace tf = tensorflow;

//PROGRAMME ENTRY POINT=========================================================
/*!
\brief Main function
*/
int main() {
  Scope R = Scope::NewRootScope();

  //NETWORK STRUCTURE-----------------------------------------------------------
  //input for network, and input for loss function
  //not fixed dimensions
  auto Input    = Placeholder(R, DT_FLOAT);
  auto Expected = Placeholder(R, DT_FLOAT);

  //layer variables
  auto Weight1 = Variable(R, {2,2}, DT_FLOAT); //2 inputs, 2 neurons
  auto Bias1   = Variable(R, {2},   DT_FLOAT); //for 2 neurons

  auto Weight2 = Variable(R, {2,1}, DT_FLOAT); //2 inputs, 1 neuron
  auto Bias2   = Variable(R, {1},   DT_FLOAT); //for 1 neuron  

  //init ops for layer variables
  auto Init_W1 = Assign(R, Weight1, RandomNormal(R,{2,2},DT_FLOAT));
  auto Init_B1 = Assign(R, Bias1,   RandomNormal(R,{2},  DT_FLOAT));
  auto Init_W2 = Assign(R, Weight2, RandomNormal(R,{2,1},DT_FLOAT));
  auto Init_B2 = Assign(R, Bias2,   RandomNormal(R,{1},  DT_FLOAT));

  //full layers with weights, biases, and activation functions
  auto Layer1 = Tanh(R, Add(R, MatMul(R,Input, Weight1),Bias1));
  auto Layer2 = Tanh(R, Add(R, MatMul(R,Layer1,Weight2),Bias2));

  //loss function
  auto Loss = Sum(R, Square(R, Sub(R,Expected,Layer2)), {0,1});

  //optimiser function
  vector<Output> Grad_Outputs;
  TF_CHECK_OK(
    AddSymbolicGradients(R, {Loss}, {Weight1,Weight2,Bias1,Bias2}, &Grad_Outputs)
  );
  auto Optim1 = ApplyGradientDescent(R, Weight1, Cast(R,0.01,DT_FLOAT), {Grad_Outputs[0]});
  auto Optim2 = ApplyGradientDescent(R, Weight2, Cast(R,0.01,DT_FLOAT), {Grad_Outputs[1]});
  auto Optim3 = ApplyGradientDescent(R, Bias1,   Cast(R,0.01,DT_FLOAT), {Grad_Outputs[2]});
  auto Optim4 = ApplyGradientDescent(R, Bias2,   Cast(R,0.01,DT_FLOAT), {Grad_Outputs[3]});

  //RUN THE NETWORK-------------------------------------------------------------
  //create session to start training
  ClientSession Sess(R);
  TF_CHECK_OK(
    Sess.Run({Init_W1,Init_W2,Init_B1,Init_B2}, nullptr)
  );

  //training data
  Tensor Inputs = Tensor(DT_FLOAT, TensorShape({4,2}));
  Tensor Labels = Tensor(DT_FLOAT, TensorShape({4,1}));

  Inputs.flat<float>()(0) = 0; //0 ^ 0
  Inputs.flat<float>()(1) = 0;
  Inputs.flat<float>()(2) = 0; //0 ^ 1
  Inputs.flat<float>()(3) = 1;
  Inputs.flat<float>()(4) = 1; //1 ^ 0
  Inputs.flat<float>()(5) = 0;
  Inputs.flat<float>()(6) = 1; //1 ^ 1
  Inputs.flat<float>()(7) = 1;

  Labels.flat<float>()(0) = 0; //0 ^ 0 == 0
  Labels.flat<float>()(1) = 1; //0 ^ 1 == 1
  Labels.flat<float>()(2) = 1; //1 ^ 0 == 1
  Labels.flat<float>()(3) = 0; //1 ^ 1 == 0

  //start training
  cout <<"\nTraining..." <<endl;
  long Steps = 5000;  
  vector<Tensor> Outputs;

  for (long I=0; I<Steps; I++){
    TF_CHECK_OK(
      Sess.Run({{Input,Inputs}, {Expected,Labels}}, 
               {Optim1,Optim2,Optim3,Optim4,Layer2}, nullptr)
    );    

    //log after every 100 steps
    if ((I+1)%100 == 0) {
      TF_CHECK_OK(
        Sess.Run({{Input,Inputs}, {Expected,Labels}}, 
                 {Loss}, &Outputs)
      );
      cout <<"Loss after " <<I+1 <<" steps: " 
           <<Outputs[0].scalar<float>() <<endl;
    }//log
  }//steps

  //start inference
  cout <<"\nInferring the original training data..." <<endl;
  Tensor Infer_Inputs = Tensor(DT_FLOAT, TensorShape({1,2}));

  for (long I=0; I<4; I++){    
    float X1 = Infer_Inputs.flat<float>()(0) = Inputs.flat<float>()(I*2);
    float X2 = Infer_Inputs.flat<float>()(1) = Inputs.flat<float>()(I*2+1);

    TF_CHECK_OK(
      Sess.Run({{Input,Infer_Inputs}}, {Layer2}, &Outputs)
    );
    cout <<X1 <<" ^ " <<X2 <<" = " <<Outputs[0].scalar<float>() <<endl;
  }
}//main

//end of file