"""
\file
\brief Model class file
"""

#core
import os;

#libs
import tensorflow as tf;

#custom modules
from tfraw.t import *;

"""
\brief Model class
"""
class model:
  Sess        = None; #session
  Outputs     = None; #tensor
  Loss        = None; #tensor
  Train       = None; #optimiser
  Feed_Dict   = None; #input & loss calculation feeds
  Tensor_Dict = None; #the same as Feed_Dict but with :0 affixes for keys

  #model dir data
  Model_Dir = None; #path for tensorflow to write data
  Saver     = None; #model data saver
  Writer    = None; #summary write to log scalars, eg. loss

  """
  \brief Default constructor
  """
  def __init__(this,Sess,Outputs,Loss,Train,Model_Dir):
    this.Sess    = Sess;
    this.Outputs = Outputs;
    this.Loss    = Loss;
    this.Train   = Train;

    #model dir data
    this.Model_Dir = Model_Dir;
    this.Saver     = tf.train.Saver();
    this.Writer    = tf.summary.FileWriter(Model_Dir);
  #end def

  """
  \brief Set training data (feed dict)
  """
  def set_feed_dict(this,Feed_Dict):
    from tfraw.t import feed2tensor;
    this.Feed_Dict   = Feed_Dict;
    this.Tensor_Dict = feed2tensor(Feed_Dict);
  #end def

  """
  \brief Train the model itself
  """
  def train(this,Steps=1):
    for I in range(Steps):  
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
  \brief Save session and model data
  """
  def save_model(this, Global_Step=None):
    Saver = tf.train.Saver(); #all vars
    Saver.save(this.Sess, "{}/model.ckpt".format(this.Model_Dir), global_step=Global_Step);
  #end def

  """
  \brief Write a summary to model dir
  """
  def save_scalar(this, Name=None, Step=None, Value=None):
    Summary = tf.Summary();
    Summary.value.add(tag=Name, simple_value=Value);
    this.Writer.add_summary(Summary,Step);
    this.Writer.flush(); 
  #end def
#end class

#end of file