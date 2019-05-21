#core
import sys;

#custom paths to modules
sys.path.append("../../src"); #parent dir of 'tfraw' dir

#custom modules
from tfraw.t import *;

#PROGRAMME ENTRY POINT
#init
init_ml();
Model_Dir = "/tmp/aaa-tf-prog";

#create model
Model = build_dnn_classifier(
  Num_Inputs   = 2,
  Hidden_Units = [4],
  Num_Outputs  = 2, #to match num classes,
  Model_Dir    = Model_Dir
);

#training and evaluation data
Training_Data={ #XOR, for example
  "Inputs": [[0,0],[0,1],[1,0],[1,1]],
  "Labels": [ 0,    1,    1,    0   ]
};

#train and evaluate
start_session();
Max_Log = 300; #Max_Log*Steps = number of batches to process
Steps   = 100; #see loss after every 100 steps

#show info and save data Max_Log times
for I in range (Max_Log):
  for J in range (Steps):
    Model.set_batch(get_rand_batch(Training_Data,4));
    Model.train(); #train with 1 batch
  #end for  

  #info every 100 steps
  Steps_Done = Steps*(I+1);
  Loss       = Model.get_loss();
  log("Steps done: {}, loss={}".format(Steps_Done,Loss));  

  #sava data every 1000 steps
  if (I+1)%10 == 0:
    Model.save_model(Global_Step=Steps_Done);
    Model.save_scalar(Name="loss", Step=Steps_Done, Value=Loss);
  #end if
#end for

#infer the original training data in feed dict
plog(Model.infer(Training_Data).tolist());

#clear temporary data
close_session();

#end of file
