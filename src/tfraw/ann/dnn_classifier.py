"""
\file
\brief Model class file
"""

#core
import copy;
import os;

#libs
import tensorflow as tf;

#custom modules
from tfraw.t import *;

"""
\brief Model class
"""
class dnn_classifier:
  Sess        = None; #session
  Num_Classes = None; #number of classes
  Outputs     = None; #tensor
  Loss        = None; #tensor
  Train       = None; #optimiser
  Feed_Dict   = None; #input & loss calculation feeds
  Tensor_Dict = None; #the same as Feed_Dict but with :0 affixes for keys

  #model dir data
  Model_Dir = None; #path for tensorflow to write data
  Saver     = None; #model data saver
  Writer    = None; #summary write to log scalars, eg. loss
  Write_Op  = None; #loss writing op

  """
  \brief Default constructor
  """
  def __init__(this,Sess,Num_Classes,Outputs,Loss,Probs,Train,Model_Dir):
    this.Sess        = Sess;
    this.Num_Classes = Num_Classes;
    this.Outputs     = Outputs;
    this.Loss        = Loss;
    this.Probs       = Probs;
    this.Train       = Train;

    #model dir data
    this.Model_Dir = Model_Dir;
    this.Saver     = tf.train.Saver();
    this.Writer    = tf.summary.FileWriter(Model_Dir);

    #scalar writing ops
    tf.summary.scalar("loss",this.Loss);
    this.Write_Op = tf.summary.merge_all();
  #end def

  """
  \brief Set training data (feed dict)
  """  
  def set_batch(this,Feed_Dict):
    from tfraw.t import feed2tensor;

    #convert labels to probabilities
    Labels          = Feed_Dict["Labels"];
    Expecteds_Probs = []; #expected probabilities

    for Label in Labels:
      Expecteds_Prob = [];
      for I in range(this.Num_Classes):
        if I==Label:
          Expecteds_Prob += [1];
        else:
          Expecteds_Prob += [0];
      #end for

      Expecteds_Probs += [Expecteds_Prob];
    #end for    

    Feed_Dict2 = copy.deepcopy(Feed_Dict);
    Feed_Dict2["Expecteds"] = Expecteds_Probs;
    del Feed_Dict2["Labels"];

    #convert to tensor dict    
    this.Feed_Dict   = Feed_Dict2;
    this.Tensor_Dict = feed2tensor(Feed_Dict2);
  #end def

  """
  \brief Train the model itself
  """
  def train(this):
    this.Train.run(feed_dict=this.Tensor_Dict);  
  #end def

  """
  \brief Get current loss
  """
  def get_loss(this):
    from tfraw.t import run_flow;
    return run_flow(this.Loss,Feed_Dict=this.Feed_Dict);
  #end def

  """
  \brief Infer to get results
  """
  def infer(this,Feed_Dict=None):
    from tfraw.t import run_flow;
    return run_flow(this.Probs,Feed_Dict=this.Feed_Dict); 
  #end def

  """
  \brief Save session and model data
  """
  def save_model(this, Global_Step=None):
    Saver = tf.train.Saver(); #all vars
    Saver.save(this.Sess, "{}/model.ckpt".format(this.Model_Dir), global_step=Global_Step);
  #end def

  """
  \brief Write a summary to model dir,
         Reference: https://github.com/tensorflow/tensorflow/issues/7089#issuecomment-295857875
  """
  def save_scalar(this, Name=None, Step=None, Value=None):
    Summary = tf.Summary();
    Summary.value.add(tag=Name, simple_value=Value);
    this.Writer.add_summary(Summary,Step);
    this.Writer.flush(); 

    #another method to save scalars:
    #Summary = this.Sess.run(this.Write_Op, {this.Loss:Value});
    #this.Writer.add_summary(Summary,Step);
    #this.Writer.flush();
  #end def
#end class

#end of file