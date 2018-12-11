import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def leaky_relu(x, leak=0.2, name="leaky_relu"):
    return tf.maximum(x, leak*x)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name) as scope:
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name) as scope:
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False, reuse = False):
  with tf.variable_scope(name, reuse = reuse) as scope:
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])
    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, reuse = False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear", reuse = reuse) as scope:
    matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def conv3d( inputs, num_filter, 
            filter_depth = 5, filter_h = 5, filter_w = 5, 
            stride_depth = 2, stride_h = 2, stride_w = 2,
            stddev = 0.02,
            name = "conv3d", output_with_weights = False, reuse = False ):
  """
    Args:
      inputs: tensor, assume shape[ batch, in_depth, in_height, in_width, in_channels ]
  """
  with tf.variable_scope( name, reuse = reuse ) as scope:
    in_channel = inputs.get_shape()[ -1 ]
    out_channel = num_filter
    w = tf.get_variable( 'w', [ filter_depth, filter_h, filter_w, in_channel, out_channel ],
                          initializer = tf.random_normal_initializer( stddev = stddev ) )
    conv = tf.nn.conv3d( inputs, w, [ 1, stride_depth, stride_h, stride_w, 1 ],
                        padding = "SAME" )
    bias = tf.get_variable( 'bias', [ num_filter ], initializer = tf.constant_initializer( 0.0 ) )
    conv = tf.reshape( tf.nn.bias_add( conv, bias ), conv.get_shape() )
    if output_with_weights:
      return conv, w, bias
    return conv

def deconv3d( inputs, output_shape,
              filter_depth = 5, filter_h = 5, filter_w = 5,
              stride_depth = 2, stride_h = 2, stride_w = 2,
              stddev = 0.02, name = "deconv3d", output_with_weights = False, reuse = False ):
  """
    Args:
      inputs: [batch, depth, height, width, in_channels]
      output_shape: [depth, height, width, output_channels, in_channels]
  """
  with tf.variable_scope( name, reuse = reuse ) as scope:
    in_channel = inputs.get_shape()[ -1 ]
    out_channel = output_shape[ -1 ]
    w = tf.get_variable( "w", [ filter_depth, filter_h, filter_w, out_channel, in_channel ],
                          initializer = tf.random_normal_initializer( stddev = stddev )  )
    # deconv = tf.nn.conv3d_transpose( inputs, [ 1, filter_depth, filter_h, filter_w, 1 ],
    #                                   output_shape,
    #                                  [ 1, stride_depth, stride_h, stride_w, 1 ], 
    #                                  padding = "SAME", name = "dconv3d" )
    deconv = tf.nn.conv3d_transpose( inputs, w,
                                    output_shape,
                                    [ 1, stride_depth, stride_h, stride_w, 1 ], 
                                    padding = "SAME", name = "dconv3d" )
    bias = tf.get_variable( 'bias', [ out_channel ], initializer = tf.constant_initializer( 0.0 ) )
    deconv = tf.reshape( tf.nn.bias_add( deconv, bias ), deconv.get_shape() )
    if output_with_weights:
      return deconv, w, bias 
    return deconv
