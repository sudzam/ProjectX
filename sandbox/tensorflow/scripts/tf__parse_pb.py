"""
   SOURCES:
   1. https://www.tensorflow.org/guide/extend/model_files
   Proto Sources:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto

   Save  / Load:
    - https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527
    - https://medium.com/@prasadpal107/saving-freezing-optimizing-for-inference-restoring-of-tensorflow-models-b4146deb21b5
   USE: parse TF graph..
"""

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
from tensorflow.python.tools import optimize_for_inference_lib

#GRAPH_PB_PATH = './frozen_model.pb'
GRAPH_PB_PATH = '/home/sudi/TF.Models/mobilenet/mobilenet_v2_0.75_96_frozen.pb'
GRAPH_PB_PATH = '/home/sudi/TMP/helloTF/HelloTF_frozen.pb'


#saver = tf.train.Saver()

with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   graph = tf.get_default_graph()
   new_input = tf.placeholder(tf.float32, shape=[None, 3], name='NEW_INPUT')  # input


   tf.import_graph_def(
      graph_def,
      # usually, during training you use queues, but at inference time use placeholders
      # this turns into "input
      input_map={"I": new_input},
      return_elements=None,
      # if input_map is not None, needs a name
      name=None,
      op_dict=None,
      producer_op_list=None
   )
   v1 = sess.graph.get_tensor_by_name('W:0')
   sess.run(tf.assign(v1, [[6, 6], [5, 5], [4, 4]]))

   myl = [n.name for n in tf.get_default_graph().as_graph_def().node]
   print (myl)

   meta_graph_def = tf.train.export_meta_graph(filename='/tmp/my-model.meta')

   #print("##Operations=\n", graph_def.get_operations())

   graph_nodes=[n for n in graph_def.node]

   names = []
   vals  = []
   #for t in graph_nodes:
   #   print(t)

# save output graph
   outputGraph = optimize_for_inference_lib.optimize_for_inference(
              graph_def,
              ["NEW_INPUT"],    # an array of the input node(s)
              ["O"], # an array of output nodes
              tf.int32.as_datatype_enum)
   f = tf.gfile.FastGFile('OptimizedGraph.pb', "w")
   f.write(outputGraph.SerializeToString())

"""
      names.append(t.name)
      if t.op=='Const':
         vals.append(tensor_util.MakeNdarray(t.attr['value'].tensor))
      else:
         vals.append(t.input)

   for i in range(len(names)):
      print ("name=",names[i], "val=\n",vals[i],"\n")
      
graph1 = tf.Graph()
with graph1.as_default():
   tf.import_graph_def(graph_def)  # graph_def1 loaded somewhere

with tf.Session(graph=graph1) as sess:
   sess.run(tf.assign(W, [[1, 2], [3, 4], [5, 6]]))
   sess.run(tf.assign(b, [1, 1]))

f = tf.gfile.FastGFile('OptimizedGraph_NEW.pb', "w")
f.write(outputGraph.SerializeToString())
"""