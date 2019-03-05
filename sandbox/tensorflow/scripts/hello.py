# simple TF graph also showing how to save/restore & freeze

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'HelloTF'
DST_PATH   = '/home/sudi/TMP/helloTF'

input_graph_path            = DST_PATH + "/" + MODEL_NAME + '.pbtxt'
checkpoint_path             = DST_PATH + '/' + MODEL_NAME + '.ckpt'
output_frozen_graph_name    = DST_PATH + '/' + MODEL_NAME + '_frozen.pb'
output_optimized_graph_name = DST_PATH + '/' + MODEL_NAME + '_optimized.pb'

input_saver_def_path = ""
input_binary         = False
output_node_names    = "O"
restore_op_name      = "save/restore_all"
filename_tensor_name = "save/Const:0"

clear_devices = True

I = tf.placeholder(tf.float32, shape=[None,3],           name='I') # input
W = tf.Variable(tf.zeros(shape=[3,2]), dtype=tf.float32, name='W') # weights
b = tf.Variable(tf.zeros(shape=[2]),   dtype=tf.float32, name='b') # biases
O = tf.nn.relu(tf.matmul(I, W) + b,                      name='O') # activation / output

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  
  # save the graph (.pbtext)
  tf.train.write_graph(sess.graph_def,DST_PATH, input_graph_path)  

  # normally you would do some training here
  # but fornow we will just assign something to W
  sess.run(tf.assign(W, [[1, 2],[3,4],[5,6]]))
  sess.run(tf.assign(b, [1,1]))

  #save a checkpoint file, which will store the above assignment (.ckpt)
  saver.save(sess, checkpoint_path)

# Freeze the graph (.pb)
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name, output_frozen_graph_name, clear_devices, "")

print ("Frozen-Graph=",output_frozen_graph_name)
