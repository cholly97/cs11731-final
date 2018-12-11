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
        bn_1 = batch_norm(name='img_ebn_1')
        bn_2 = batch_norm(name='img_ebn_2')
        bn_3 = batch_norm(name='img_ebn_3')
        # print("img_encode input shape", inputs.get_shape().as_list())
        if img_encode_bn:
            h0 = tf.nn.relu(bn_1(conv3d(inputs, 32, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv3d(inputs, 32, name='h0_conv'))
        # print("img_encode h0 shape", h0.get_shape().as_list())
        if img_encode_bn:
            h1 = tf.nn.relu(bn_2(conv3d(h0, 64, name='h1_conv')))
        else:
            h1 = tf.nn.relu(conv3d(h0, 64, name='h1_conv'))
        # print("img_encode h1 shape", h1.get_shape().as_list())
        # if img_encode_bn:
        #     h2 = tf.nn.relu(bn_3(conv3d(h1, 128, filter_depth = 3, filter_h = 3, filter_w = 3, name='h2_conv')))
        # else:
        #     h2 = tf.nn.relu(conv3d(h1, 128, filter_depth = 3, filter_h = 3, filter_w = 3, name='h2_conv'))
        # # print('h2 shape', h2.shape)
        # # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        # rs = tf.reshape(h2, [batch_size, -1])
        # # print("rs shape", rs.get_shape().as_list())

        # z_mu_l0 = tf.nn.relu(linear(rs, 256, 'z_mu_l0'))
        # # print("img_encode z_mu_l0 shape", z_mu_l0.get_shape().as_list())
        # # z_logvar_l0 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='z_logvar_l0')
        # z_logvar_l0 = tf.nn.relu(linear(tf.reshape(h2, [batch_size, -1]), 256, 'z_logvar_l0'))

        # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        print("h1 shape", h1.get_shape().as_list())
        rs = tf.reshape(h1, [batch_size, -1])
        # print("rs shape", rs.get_shape().as_list())

        z_mu_l0 = tf.nn.relu(linear(rs, 128, 'z_mu_l0'))
        # print("img_encode z_mu_l0 shape", z_mu_l0.get_shape().as_list())
        # z_logvar_l0 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='z_logvar_l0')
        z_logvar_l0 = tf.nn.relu(linear(tf.reshape(h1, [batch_size, -1]), 128, 'z_logvar_l0'))

        z_mu_l1 = tf.layers.dense(z_mu_l0, encode_dim, activation = None, name='z_mu_l1')
        # print("img_encode z_mu_l1 shape", z_mu_l1.get_shape().as_list())
        z_logvar_l1 = tf.layers.dense(z_logvar_l0, encode_dim, activation=tf.nn.softplus, name='z_logvar_l1') 

        return z_mu_l1, z_logvar_l1, sample_z(z_mu_l1, z_logvar_l1)

def img_decoder(z, output_depth, output_width, output_height, c_dim):
    with tf.variable_scope('img_decoder', reuse = tf.AUTO_REUSE) as scope:
        g_bn0 = batch_norm(name='img_g_bn0')
        g_bn1 = batch_norm(name='img_g_bn1')
        g_bn2 = batch_norm(name='img_g_bn2')
        g_bn3 = batch_norm(name='img_g_bn3')
        s_d, s_w, s_h = output_depth, output_width, output_height
        s_d2, s_h2, s_w2 = conv_out_size_same(s_d, 2), conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_d4, s_h4, s_w4 = conv_out_size_same(s_d2, 2), conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_d8, s_h8, s_w8 = conv_out_size_same(s_d4, 2), conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_d16, s_h16, s_w16 = conv_out_size_same(s_d8, 2), conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # z_dense = tf.layers.dense(z, 256, activation = tf.nn.relu, name='z_dense', reuse = tf.AUTO_REUSE)

        # z_, h0_w, h0_b = linear(z, gf_dim*8*s_d16*s_h16*s_w16, 'h0_lin', with_w=True, reuse = tf.AUTO_REUSE)

        # h0 = tf.reshape(z_, [-1, s_d16, s_h16, s_w16, gf_dim * 8])
        z_, h0_w, h0_b = linear(z, gf_dim*4*s_d8*s_h8*s_w8, 'h0_lin', with_w=True, reuse = tf.AUTO_REUSE)

        h0 = tf.reshape(z_, [-1, s_d8, s_h8, s_w8, gf_dim * 4])
        if img_decode_bn:
            h0 = tf.nn.relu(g_bn0(h0))
        else:
            h0 = tf.nn.relu(h0)

        # h1, h1_w, h1_b = deconv3d(
        #     h0, [batch_size, s_d8, s_h8, s_w8, gf_dim*4], name='h1', output_with_weights=True, reuse = tf.AUTO_REUSE)
        # if img_decode_bn:
        #     h1 = tf.nn.relu(g_bn1(h1))
        # else:
        #     h1 = tf.nn.relu(h1)

        # h2, h2_w, h2_b = deconv3d(
        #     h1, [batch_size, s_d4, s_h4, s_w4, gf_dim*2], name='h2', output_with_weights=True, reuse = tf.AUTO_REUSE)
        h2, h2_w, h2_b = deconv3d(
            h0, [batch_size, s_d4, s_h4, s_w4, gf_dim*2], name='h2', output_with_weights=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h2 = tf.nn.relu(g_bn2(h2))
        else:
            h2 = tf.nn.relu(h2)

        h3, h3_w, h3_b = deconv3d(
            h2, [batch_size, s_d2, s_h2, s_w2, gf_dim*1], name='h3', output_with_weights=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h3 = tf.nn.relu(g_bn3(h3))
        else:
            h3 = tf.nn.relu(h3)

        h4, h4_w, h4_b = deconv3d(
            h3, [batch_size, s_d, s_h, s_w, c_dim], name='h4', output_with_weights=True, reuse = tf.AUTO_REUSE)

        # print("img recon logit", h4.get_shape().as_list())
        return tf.nn.tanh(h4), h4

