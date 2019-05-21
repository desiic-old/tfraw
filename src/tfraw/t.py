"""
\file
\brief TensorFlow Raw library main class file,
       Reference: https://www.oreilly.com/ideas/building-deep-learning-neural-networks-using-tensorflow-layers
"""

#core
import os,sys;
import pprint     as pp;
import subprocess as sp;

#libs
import numpy      as np;
import tensorflow as tf;
from   random     import *;

#custom classes
from dotdict                  import *;
from tfraw.ann.dnn_classifier import *;

#shortcuts
log  = tf.logging.info;

#global vars
Sess = tf.InteractiveSession();

"""
\brief TensorFlow Raw library main class
"""
class t: #short for tensorflow

  """
  \brief Pretty print
  """
  @staticmethod
  def plog(Var):
    log(pp.pformat(Var,indent=2));
  #end def

  """
  \brief Halt the programme
  """
  @staticmethod
  def halt():
    os._exit(1);
  #end def

  """
  \brief Initialise TensorFlow
  """
  @staticmethod
  def init_ml():    
    tf.logging.set_verbosity(tf.logging.INFO);    
  #end def

  """
  \brief Remove a TensorFlow model dir
  """
  @staticmethod
  def rm_model_dir(Model_Dir):
    if (not os.path.isfile(Model_Dir+"/checkpoint")):
      log("Directory {} doesn't seem to be a model dir!".format(Model_Dir));
      return;
    else:
      log("Directory {} removed.".format(Model_Dir));
    #end if

    try:
      Out=sp.check_output(["rm","-r",Model_Dir],stderr=sp.STDOUT);
      if len(Out)>0:
        log(Out);
    except Exception as Err:
      log(Err);
    #end try    
  #end def 

  """
  \brief Build DNN model
  """
  @staticmethod
  def build_dnn_classifier(Num_Inputs,Hidden_Units,Num_Outputs,
  Activation=tf.nn.relu, Output_Activation=None, #None is identity
  Optimiser=tf.train.AdamOptimizer(1e-3), Model_Dir="/tmp/tfraw-dnn-classifier"): 

    #make the network
    #input layer, None is batch dimension (any batch_size)
    Net = tf.placeholder(shape=[None,Num_Inputs], dtype=tf.float32,
                         name="Inputs");
    
    #hidden layers
    I = 1;
    for Num_Units in Hidden_Units:
      Net = tf.layers.dense(inputs=Net, units=Num_Units, activation=Activation, 
                            name="Hidden"+str(I));
      I += 1;
    #end for

    #output layer
    Dropout = tf.layers.dropout(inputs=Net, rate=0.1, training=True);
    Outputs = tf.layers.dense(inputs=Dropout, units=Num_Outputs, activation=Output_Activation, 
                              name="Outputs");
    Probs = tf.nn.softmax(Outputs);

    #compute loss, None is batch dimension (any batch_size)   
    Expected_Probs = tf.placeholder(shape=[None,Num_Outputs], dtype=tf.float32,
                                    name="Expecteds");
    Loss  = tf.reduce_sum(tf.square(Expected_Probs-Probs));
    Train = Optimiser.minimize(Loss);

    #pass back model
    Model = dnn_classifier(Sess,Num_Outputs,Outputs,Loss,Probs,Train,Model_Dir);    
    return Model;
  #end def

  """
  \brief Start session
  """
  @staticmethod
  def start_session():
    Sess.run(tf.global_variables_initializer());
  #end def

  """
  \brief Close session
  """
  @staticmethod
  def close_session():
    Sess.close();
  #end def

  """
  \brief Get a random batch from dataset
  """
  def get_rand_batch(Dataset,Batch_Size):
    
    #make a list to shuffle
    List = [];
    for I in range(len(Dataset["Inputs"])):
      Input = Dataset["Inputs"][I];
      Label = Dataset["Labels"][I];
      List += [{"Input":Input, "Label":Label}];
    #end for

    #shuffle dataset
    shuffle(List);

    #get a batch
    Batch = {"Inputs":[], "Labels":[]};
    for I in range(Batch_Size):
      Batch["Inputs"] += [List[I]["Input"]];
      Batch["Labels"] += [List[I]["Label"]];
    #end for

    return Batch;
  #end def

  """
  \brief Convert feed_dict to tensor dict (with :0 postfixes)
  """
  @staticmethod
  def feed2tensor(Feed_Dict=None):
    Tensor_Dict = {};
    for Key in Feed_Dict:
      Tensor_Dict[Key+":0"] = Feed_Dict[Key];

    return Tensor_Dict;
  #end def

  """
  \brief Run a flow
  """
  @staticmethod
  def run_flow(Flow_To=None,Feed_Dict=None):
    return Sess.run(Flow_To,feed_dict=feed2tensor(Feed_Dict));
  #end def
#end class

#export shortcuts
plog = t.plog;
halt = t.halt;
init_ml              = t.init_ml;
rm_model_dir         = t.rm_model_dir;
build_dnn_classifier = t.build_dnn_classifier;
start_session        = t.start_session;
close_session        = t.close_session;
get_rand_batch       = t.get_rand_batch;
feed2tensor          = t.feed2tensor;
run_flow             = t.run_flow;

#end of file