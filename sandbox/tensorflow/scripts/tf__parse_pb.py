"""
import tensorflow as tf
from tensorflow.python.platform import gfile

#path to your .pb file
GRAPH_PB_PATH = '/tmp/mnist_saved_model/1551743470/saved_model.pb'
GRAPH_PB_PATH = '/home/sudi/TF.Models/mobilenet/mobilenet_v2_0.75_96_frozen.pb'


with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]

wts = [n for n in graph_nodes] if n.op=='Const']

from tensorflow.python.framework import tensor_util

for n in wts:
    print ("Name of the node - %s" % n.name)
    print ("Value - " )
    print (tensor_util.MakeNdarray(n.attr['value'].tensor))
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

#GRAPH_PB_PATH = './frozen_model.pb'
GRAPH_PB_PATH = '/home/sudi/TF.Models/mobilenet/mobilenet_v2_0.75_96_frozen.pb'

with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]

   names = []
   vals  = []
   for t in graph_nodes:
      print(t)

"""
      names.append(t.name)
      if t.op=='Const':
         vals.append(tensor_util.MakeNdarray(t.attr['value'].tensor))
      else:
         vals.append(t.input)

   for i in range(len(names)):
      print ("name=",names[i], "val=\n",vals[i],"\n")
"""
