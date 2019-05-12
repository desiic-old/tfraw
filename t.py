"""
\file
\brief TensorFlow Raw library main class file
"""

#core
import os,sys;
import pprint     as pp;
import subprocess as sp;

#libs
import numpy      as np;
import tensorflow as tf;

#shortcuts
log  = tf.logging.info;
sess = tf.InteractiveSession();

"""
\brief TensorFlow Raw library main class
"""
class t: #short for tensorflow

  """
  \brief Pretty print
  """
  def pprint(Var):
    log(pp.pformat(Var,indent=2));
  #end def

  """
  \brief Halt the programme
  """
  def halt():
    os._exit(1);
  #end def

  """
  \brief Initialise TensorFlow
  """
  def init_ml():
    tf.logging.set_verbosity(tf.logging.INFO);
    sess.run(tf.global_variables_initializer());
  #end def

  """
  \brief Remove a TensorFlow model dir
  """
  def rm_model_dir(Model_Dir):
    if (not os.path.isfile(Model_Dir+"/checkpoint")):
      log("Directory {} doesn't seem to be a model dir!".format(Model_Dir));
      return;
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
  \brief Create DNN model
  """
  def create_dnn_model(Num_Inputs,Hidden_Units,Num_Outputs,
  Activation=tf.nn.relu, Output_Activation=None, #None is identity
  Optimiser=tf.train.AdamOptimizer(1e-3)): 

    #make the network
    Net = tf.placeholder(shape=(Num_Inputs), dtype=tf.float32);
    
    for Num_Units in Hidden_Units:
      Net = tf.layers.dense(inputs=Net, units=Num_Units, activation=Activation);

    Outputs = tf.layers.dense(inputs=Net, units=Num_Outputs, activation=Output_Activation);

    #compute loss
    Loss = 
  #end def
#end class

#export shortcuts
pprint = t.pprint;
halt   = t.halt;
init_ml          = t.init_ml;
rm_model_dir     = t.rm_model_dir;
create_dnn_model = t.create_dnn_model;

#end of file