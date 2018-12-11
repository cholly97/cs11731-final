import math
import numpy as np 
import tensorflow as tf 
from dc_gan_util import *

gf_dim = 16
df_dim = 16
batch_size = 16

encode_dim = 100
img_encode_bn = True
img_decode_bn = True
img_disc_bn = True

word_encode_bn = False
word_decode_bn = False
word_disc_bn = False

def leaky_relu(x, leak=0.2, name="leaky_relu"):
    return tf.maximum(x, leak*x)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu), stddev = 1.0)
    res = mu + tf.exp(log_var/2)*eps*0.1
    return res

def img_encoder(inputs):
    with tf.variable_scope('img_encoder', reuse = tf.AUTO_REUSE) as scope:
        bn_0 = batch_norm(name='img_ebn_0')
        bn_1 = batch_norm(name='img_ebn_1')
        
        if img_encode_bn:
            h0 = tf.nn.relu(bn_0(conv3d(inputs, 32, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv3d(inputs, 32, name='h0_conv'))

        if img_encode_bn:
            h1 = tf.nn.relu(bn_1(conv3d(h0, 64, name='h1_conv')))
        else:
            h1 = tf.nn.relu(conv3d(h0, 64, name='h1_conv'))

        z_mu_l0 = tf.nn.relu(h1)

        bn_2 = batch_norm(name='img_ebn_0')
        bn_3 = batch_norm(name='img_ebn_1')
        
        if img_encode_bn:
            h2 = tf.nn.relu(bn_2(conv3d(inputs, 32, name='h2_conv')))
        else:
            h2 = tf.nn.relu(conv3d(inputs, 32, name='h2_conv'))

        if img_encode_bn:
            h3 = tf.nn.relu(bn_3(conv3d(h0, 64, name='h3_conv')))
        else:
            h3 = tf.nn.relu(conv3d(h0, 64, name='h3_conv'))

        z_logvar_l0 = tf.nn.relu(h3)

        return z_mu_l0, z_logvar_l0, sample_z(z_mu_l0, z_logvar_l0)

def img_decoder(z, output_depth, output_width, output_height, c_dim):
    with tf.variable_scope('img_decoder', reuse = tf.AUTO_REUSE) as scope:
        g_bn0 = batch_norm(name='img_g_bn0')
        g_bn1 = batch_norm(name='img_g_bn1')
        g_bn2 = batch_norm(name='img_g_bn2')
        s_d, s_w, s_h = output_depth, output_width, output_height
        s_d2, s_h2, s_w2 = conv_out_size_same(s_d, 2), conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_d4, s_h4, s_w4 = conv_out_size_same(s_d2, 2), conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_d8, s_h8, s_w8 = conv_out_size_same(s_d4, 2), conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

        z_, _, _ = linear(z, gf_dim*4*s_d8*s_h8*s_w8, 'h0_lin', with_w=True, reuse = tf.AUTO_REUSE)

        h0 = tf.reshape(z_, [-1, s_d8, s_h8, s_w8, gf_dim * 4])
        if img_decode_bn:
            h0 = tf.nn.relu(g_bn0(h0))
        else:
            h0 = tf.nn.relu(h0)

        h1, _, _ = deconv3d(
            h0, [batch_size, s_d4, s_h4, s_w4, gf_dim*2], name='h1', output_with_weights=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h1 = tf.nn.relu(g_bn1(h1))
        else:
            h1 = tf.nn.relu(h1)

        h2, _, _ = deconv3d(
            h1, [batch_size, s_d2, s_h2, s_w2, gf_dim*1], name='h2', output_with_weights=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h2 = tf.nn.relu(g_bn2(h2))
        else:
            h2 = tf.nn.relu(h2)

        h3, _, _ = deconv3d(
            h2, [batch_size, s_d, s_h, s_w, c_dim], name='h3', output_with_weights=True, reuse = tf.AUTO_REUSE)

        return tf.nn.tanh(h3), h3

