# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:58:39 2018

@author: PARVA SHAH
"""


from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras.models import load_model

loaded_model=load_model('cardogcpunogray.hdf5')


MODEL_NAME="cardogcpunogray"
output_node_name="lastlayer/Softmax"
input_node_names=["conv2d_1_input"]
saver=tf.train.Saver()

tf.train.write_graph(K.get_session().graph_def, 'out', MODEL_NAME + '_graph.pbtxt')


tf.train.Saver().save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')


freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, False, 'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all", "save/Const:0", 'out1/frozen_' + MODEL_NAME + '.pb', True, "")

input_graph_def = tf.GraphDef()
with tf.gfile.Open('out1/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

with tf.gfile.FastGFile('out1/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

print("graph saved!")



"""
 from tensorflow.tools.graph_transforms import TransformGraph
    transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
    constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
"""