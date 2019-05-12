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

#custom classes
from tfraw.dotdict import *;
from tfraw.model   import *;

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
  def build_dnn_model(Num_Inputs,Hidden_Units,Num_Outputs,
  Activation=tf.nn.relu, Output_Activation=None, #None is identity
  Optimiser=tf.train.AdamOptimizer(1e-3), Model_Dir="/tmp/tfraw-model"): 

    #make the network
    #input layer, None is batch dimension (any batch_size)
    Net = tf.placeholder(shape=[None,Num_Inputs], dtype=tf.float32,
                         name="inputs");
    
    #hidden layers
    I = 1;
    for Num_Units in Hidden_Units:
      Net = tf.layers.dense(inputs=Net, units=Num_Units, activation=Activation, 
                            name="hidden"+str(I));
      I += 1;
    #end for

    #output layer
    Outputs = tf.layers.dense(inputs=Net, units=Num_Outputs, activation=Output_Activation, 
                              name="outputs");

    #compute loss, None is batch dimension (any batch_size)   
    Probs = tf.placeholder(shape=[None,Num_Outputs], dtype=tf.float32,
                           name="probs");
    Loss  = tf.reduce_sum(tf.square(Probs-Outputs));
    Train = Optimiser.minimize(Loss);

    #pass back model
    Model = model(Sess,Outputs,Loss,Train,Model_Dir);    
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
  \brief Convert feed_dict to tensor dict
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
init_ml         = t.init_ml;
rm_model_dir    = t.rm_model_dir;
build_dnn_model = t.build_dnn_model;
start_session   = t.start_session;
feed2tensor     = t.feed2tensor;
run_flow        = t.run_flow;

#end of file