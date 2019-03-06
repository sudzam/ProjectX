# simple TF graph also showing how to save/restore & freeze
"""
good tutorials:
 -  https://www.youtube.com/watch?v=yX8KuPZCAMo ()
 - https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/
 - https://petewarden.com/2017/06/22/what-ive-learned-about-neural-network-quantization/
 - https://www.tensorflow.org/lite/performance/post_training_quantization

convert to tflite:
 > toco --graph_def_file=/home/sudi/TMP/helloTF/HelloTF_frozen.pb --input_arrays="INPUT" --output_arrays="OUTPUT" --output_file=/home/sudi/TMP/helloTF/HelloTF.tflite
"""

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

MODEL_NAME = 'HelloTF'
DST_PATH   = '/home/sudi/TMP/helloTF'

input_graph_path            = DST_PATH + "/" + MODEL_NAME + '.pbtxt'
checkpoint_path             = DST_PATH + '/' + MODEL_NAME + '.ckpt'
checkpoint_path_quant       = DST_PATH + '/' + MODEL_NAME + '_quant.ckpt'
output_frozen_graph_name    = DST_PATH + '/' + MODEL_NAME + '_frozen.pb'
output_optimized_graph_name = DST_PATH + '/' + MODEL_NAME + '_optimized.pb'
eval_graph_name             = DST_PATH + '/' + MODEL_NAME + '_eval.ckpt'

input_saver_def_path = ""
input_binary         = False
output_node_names    = "OUTPUT"
restore_op_name      = "save/restore_all"
filename_tensor_name = "save/Const:0"

clear_devices = True

# placeholder => value will be provided in the future
I = tf.placeholder(tf.float32, shape=[None,3],           name='INPUT') # input

# variables are trainable (i.e. parameters)
W = tf.Variable(tf.zeros(shape=[3,2]), dtype=tf.float32, name='WEIGHT') # weights
b = tf.Variable(tf.zeros(shape=[2]),   dtype=tf.float32, name='bias') # biases
O = tf.nn.relu(tf.matmul(I, W) + b,                      name='OUTPUT') # activation / output

# expected output
y    = tf.placeholder(dtype=tf.float32, shape=O.shape)
# find square of difference
sq   = tf.square(O-y)
# sum of squares
loss = tf.reduce_sum(sq)

g=tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=100)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train     = optimizer.minimize(loss)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

# using 'with' allows implicit clean up
# so everything within the block of code can use 'sess'
# 'sess' will be closed after the last statement of the block

# instead of having: sess=tf.Session() ... sess.close()
with tf.Session() as sess:
  sess.run(init_op)
  
  # save the graph (.pbtext)
  tf.train.write_graph(sess.graph_def,DST_PATH, input_graph_path)


  # normally you would do some training here
  # but for now we will just assign something to W
  sess.run(tf.assign(W, [[1, 2],[3,4],[5,6]]))
  sess.run(tf.assign(b, [10,1]))

  # since I is a placeholder, you need to provide a value
  I_feed = [[1,2,3]]

  # expected values
  y_feed = [[5.3,32.43]]

  num_epoch = 1000
  print("#### TRAINING STARTS ####")
  for i in range (num_epoch):
    # notice how to extract multiple values in a single pass
    _,Lv,Ov = sess.run([train,loss,O],feed_dict={I:I_feed,y:y_feed})
  if i==num_epoch-1:
    print("Final Loss:",Lv)

  print ("#### AFTER TRAINING ####")
  print ("Weights:")
  print(sess.run([W]))
  print("\n")

  #save a checkpoint file, which will store the above assignment (.ckpt)
  saver.save(sess, checkpoint_path)

  # tf.contrib.quantize.create_eval_graph(input_graph=g)

  FileWriter = tf.summary.FileWriter('/tmp/TensorGraphs',sess.graph)

# Freeze the graph (.pb)
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name, output_frozen_graph_name, clear_devices, "")



print ("Frozen-Graph=",output_frozen_graph_name)
