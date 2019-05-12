"""
\file
\brief Turn a dict into object accessiable with dot
"""

"""
\brief dot.notation access to dictionary attributes,
       Reference: https://stackoverflow.com/a/23689767/5581893
"""
class dotdict(dict):    
  __getattr__ = dict.get;
  __setattr__ = dict.__setitem__;
  __delattr__ = dict.__delitem__;
#end class

#end of file