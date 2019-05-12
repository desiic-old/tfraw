"""
\file
\brief TensorFlow Raw library main class file
"""

#core
import os;
import pprint     as pp;
import subprocess as sp;

#libs
import numpy      as np;
import tensorflow as tf;

#shortcut
log = tf.logging.info;

"""
\brief TensorFlow Raw library main class
"""
class t:

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
  \brief Remove a TensorFlow model dir
  """
  def rm_model_dir(Model_Dir):
    sp.check_output(["rm","-rf",Model_Dir]);
  #end def

  """
  \brief Initialise TensorFlow
  """
  def init():
    global log;
    tf.logging.set_verbosity(tf.logging.INFO);    
    log = tf.logging.info;
  #end def
#end class

#shortcuts
t.log = tf.logging.info;

#end of file