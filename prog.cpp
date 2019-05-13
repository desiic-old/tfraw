//c++ core libs
#include <iostream>
#include <vector>

//tensorflow libs
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/gradients.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>

//namespaces
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

//shortcuts
namespace tf = tensorflow;

//PROGRAMME ENTRY POINT
/*!
\brief Main function
*/
int main() {
  Scope scope = Scope::NewRootScope();

  //input for network, and input for loss function
  auto x = Placeholder(scope, DT_FLOAT);
  auto y = Placeholder(scope, DT_FLOAT);

  //create single layer of single neuron with 
  //random weights & bias
  //2 inputs, 1 neuron
  auto layer1    = Variable(scope, {2,2}, DT_FLOAT);
  auto assign_l1 = Assign(scope, layer1,  
                   RandomNormal(scope,{2,2},DT_FLOAT));

  auto bias1     = Variable(scope, {2}, DT_FLOAT);
  auto assign_b1 = Assign(scope, bias1,  
                   RandomNormal(scope,{2},DT_FLOAT));

  auto layer2    = Variable(scope, {2,1}, DT_FLOAT);
  auto assign_l2 = Assign(scope, layer2,  
                   RandomNormal(scope,{2,1},DT_FLOAT));

  auto bias2     = Variable(scope, {1}, DT_FLOAT);
  auto assign_b2 = Assign(scope, bias2,  
                   RandomNormal(scope,{1},DT_FLOAT));

  //training steps (only 1)
  auto step1 = Tanh(scope, Add(scope, 
    MatMul(scope,x,layer1), 
    bias1
  ));
  auto step2 = Tanh(scope, Add(scope, 
    MatMul(scope,step1,layer2), 
    bias2
  ));

  //loss function
  auto cost = ReduceSum(scope, Square(scope,  
              Sub(scope,y,step2)), {0,1});

  //optimiser function
  //learning rate: 0.01
  //function:      gradient descent
  vector<Output> grad_outputs;
  TF_CHECK_OK(AddSymbolicGradients(scope, {cost}, 
              {layer1,layer2,bias1,bias2}, &grad_outputs));
  auto a1 = ApplyGradientDescent(scope, layer1,  
       Cast(scope,0.01,DT_FLOAT), {grad_outputs[0]});
  auto a2 = ApplyGradientDescent(scope, layer2,  
       Cast(scope,0.01,DT_FLOAT), {grad_outputs[1]});
  auto a3 = ApplyGradientDescent(scope, bias1,  
       Cast(scope,0.01,DT_FLOAT), {grad_outputs[2]});
  auto a4 = ApplyGradientDescent(scope, bias2,  
       Cast(scope,0.01,DT_FLOAT), {grad_outputs[3]});

  //create session to start training
  ClientSession session(scope);
  vector<Tensor> outputs;
  TF_CHECK_OK(session.Run({assign_l1,assign_l2,assign_b1,assign_b2}, nullptr));

  //training data
  Tensor OR_X = Tensor(DT_FLOAT, TensorShape({4,2}));
  Tensor OR_Y = Tensor(DT_FLOAT, TensorShape({4,1}));

  OR_X.flat<float>()(0) = 0; //0 | 0
  OR_X.flat<float>()(1) = 0;
  OR_X.flat<float>()(2) = 0; //0 | 1
  OR_X.flat<float>()(3) = 1;
  OR_X.flat<float>()(4) = 1; //1 | 0
  OR_X.flat<float>()(5) = 0;
  OR_X.flat<float>()(6) = 1; //1 | 1
  OR_X.flat<float>()(7) = 1;

  OR_Y.flat<float>()(0) = 0; //0 | 0 == 0
  OR_Y.flat<float>()(1) = 1; //0 | 1 == 1
  OR_Y.flat<float>()(2) = 1; //1 | 0 == 1
  OR_Y.flat<float>()(3) = 1; //1 | 1 == 1

  //start trainging, 1 extra round 
  //for printing at MAX_EPOCH
  long MAX_EPOCH = 1000;

  for (long i=0; i<MAX_EPOCH+1; i++){
    TF_CHECK_OK(session.Run({{x,OR_X}, {y,OR_Y}}, 
                {cost}, &outputs)); //loss calc

    if (i%100 == 0)
      cout <<"Loss after " <<i <<" steps " 
           <<outputs[0].scalar<float>() <<endl;

    //nullptr because the output from the run is useless, 
    //just training again&again
    TF_CHECK_OK(session.Run({{x,OR_X}, {y,OR_Y}}, 
                {a1,a2,a3,a4,step2}, nullptr));
  }
  }