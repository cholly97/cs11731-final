import math
import numpy as np 
import tensorflow as tf 
from dc_gan_util import *

gf_dim = 64
df_dim = 64
batch_size = 100

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
            h0 = tf.nn.relu(bn_1(conv2d(inputs, 32, k_h=3, k_w=3, name='h0_conv')))
        else:
            h0 = tf.nn.relu(conv2d(inputs, 32, k_h=3, k_w=3, name='h0_conv'))
        # print("img_encode h0 shape", h0.get_shape().as_list())
        if img_encode_bn:
            h1 = tf.nn.relu(bn_2(conv2d(h0, 64, k_h=3, k_w=3, name='h1_conv')))
        else:
            h1 = tf.nn.relu(conv2d(h0, 64, k_h=3, k_w=3, name='h1_conv'))
        # print("img_encode h1 shape", h1.get_shape().as_list())
        if img_encode_bn:
            h2 = tf.nn.relu(bn_3(conv2d(h1, 128, k_h=3, k_w=3, name='h2_conv')))
        else:
            h2 = tf.nn.relu(conv2d(h1, 128, k_h=3, k_w=3, name='h2_conv'))
        # print('h2 shape', h2.shape)
        # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        rs = tf.reshape(h2, [batch_size, -1])
        # print("rs shape", rs.get_shape().as_list())

        z_mu_l0 = tf.nn.relu(linear(rs, 256, 'z_mu_l0'))
        # print("img_encode z_mu_l0 shape", z_mu_l0.get_shape().as_list())
        # z_logvar_l0 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='z_logvar_l0')
        z_logvar_l0 = tf.nn.relu(linear(tf.reshape(h2, [batch_size, -1]), 256, 'z_logvar_l0'))

        z_mu_l1 = tf.layers.dense(z_mu_l0, encode_dim, activation = None, name='z_mu_l1')
        # print("img_encode z_mu_l1 shape", z_mu_l1.get_shape().as_list())
        z_logvar_l1 = tf.layers.dense(z_logvar_l0, encode_dim, activation=tf.nn.softplus, name='z_logvar_l1') 

        return z_mu_l1, z_logvar_l1, sample_z(z_mu_l1, z_logvar_l1)

"""
def img_encoder(inputs):
    with tf.variable_scope('img_encoder', reuse = tf.AUTO_REUSE) as scope:
        bn_0 = batch_norm(name='img_ebn_0')
        bn_1_1 = batch_norm(name='img_ebn_1_1')
        bn_1_2 = batch_norm(name='img_ebn_1_2')
        bn_2 = batch_norm(name='img_ebn_2')
        bn_3_1 = batch_norm(name='img_ebn_3_1')
        bn_3_2 = batch_norm(name='img_ebn_3_2')
        bn_4 = batch_norm(name='img_ebn_4')
        # bn_5 = batch_norm(name='img_ebn_5')
        # bn_6 = batch_norm(name='img_ebn_6')
        # bn_7 = batch_norm(name='img_ebn_7')
        # bn_8 = batch_norm(name='img_ebn_8')
        # bn_9 = batch_norm(name='img_ebn_9')

        # print("img_encode input shape", inputs.get_shape().as_list())
        h0 = tf.nn.relu(bn_0(conv2d(inputs, 32, k_h=5, k_w=5, name='h0_conv')))

        # print("img_encode h0 shape", inputs.get_shape().as_list())
        h1_1 = tf.nn.relu(bn_1_1(conv2d(h0, 32, k_h=3, k_w=3, d_h = 1, d_w = 1, name='h1_1_conv')))

        h1_2 = tf.nn.relu(bn_1_2(conv2d(h1_1, 32, k_h=3, k_w=3, d_h = 1, d_w = 1, name='h1_2_conv')))

        # print("img_encode h1 shape", inputs.get_shape().as_list())
        h2 = tf.nn.relu(bn_2(conv2d(h1_2 + h0, 64, k_h=3, k_w=3, name='h2_conv')))

        h3_1 = tf.nn.relu(bn_3_1(conv2d(h2, 64, k_h=3, k_w=3,d_h = 1, d_w = 1, name='h3_1_conv')))

        h3_2 = tf.nn.relu(bn_3_2(conv2d(h3_1, 64, k_h=3, k_w=3,d_h = 1, d_w = 1, name='h3_2_conv')))

        h4 = tf.nn.relu(bn_4(conv2d(h3_2 + h2, 128, k_h=3, k_w=3, name='h4_conv')))

        # h5_1 = tf.nn.relu(bn_5_1(conv2d(h4, 128, k_h=3, k_w=3, name='h5_1_conv')))
        # h5_2 = tf.nn.relu(bn_5_2(conv2d(h5_1, 128, k_h=3, k_w=3, name='h5_2_conv')))

        # z_mu_l0 = tf.layers.dense(h1, 256, activation = tf.nn.relu, name='z_mu_l0')
        z_mu_l0 = tf.nn.relu(linear(tf.reshape(h4, [batch_size, -1]), 512, 'z_mu_l0'))
        # print("img_encode z_mu_l0 shape", z_mu_l0.get_shape().as_list())
        # z_logvar_l0 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='z_logvar_l0')
        z_logvar_l0 = tf.nn.relu(linear(tf.reshape(h4, [batch_size, -1]), 512, 'z_logvar_l0'))

        z_mu_l1 = tf.layers.dense(z_mu_l0, encode_dim, activation = None, name='z_mu_l1')
        # print("img_encode z_mu_l1 shape", z_mu_l1.get_shape().as_list())
        z_logvar_l1 = tf.layers.dense(z_logvar_l0, encode_dim, activation=None, name='z_logvar_l1') 

        return z_mu_l1, z_logvar_l1, sample_z(z_mu_l1, z_logvar_l1)
"""
'''
def img_decoder(z, output_width, output_height):
    with tf.variable_scope('img_decoder', reuse = tf.AUTO_REUSE) as scope:
        g_bn0 = batch_norm(name='img_g_bn0')
        g_bn1 = batch_norm(name='img_g_bn1')
        g_bn2_1 = batch_norm(name='img_g_bn2_1')
        g_bn2_2 = batch_norm(name='img_g_bn2_2')
        g_bn3 = batch_norm(name='img_g_bn3')
        g_bn4_mu = batch_norm(name='img_g_bn4_mu')
        g_bn4_1 = batch_norm(name='img_g_bn4_1')
        g_bn4_2 = batch_norm(name='img_g_bn4_2')
        g_bn5 = batch_norm(name='img_g_bn5')
        s_w, s_h =  output_width, output_height
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # z_dense = tf.layers.dense(z, 256, activation = None, name='z_dense', reuse = tf.AUTO_REUSE)

        z_, h0_w, h0_b = linear(z, gf_dim*8*s_h16*s_w16, 'h0_lin', with_w=True, reuse = tf.AUTO_REUSE)

        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        if img_decode_bn:
            h0 = tf.nn.relu((h0))
        else:
            h0 = tf.nn.relu(h0)

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, gf_dim*4], name='h1',k_h = 3, k_w = 3, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h1 = tf.nn.relu(g_bn1(h1))
        else:
            h1 = tf.nn.relu(h1)

        h2_1, h2_1_w, h2_1_b = deconv2d(
            h1, [batch_size, s_h8, s_w8, gf_dim*4], name='h2_1',k_h = 3, k_w = 3, d_h=1,d_w=1, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h2_1 = tf.nn.relu(g_bn2_1(h2_1))
        else:
            h2_1 = tf.nn.relu(h2_1)

        h2_2, h2_2_w, h2_2_b = deconv2d(
            h2_1, [batch_size, s_h8, s_w8, gf_dim*4], name='h2_2',k_h = 3, k_w = 3, d_h=1,d_w=1, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h2_2 = tf.nn.relu(g_bn2_2(h2_2))
        else:
            h2_2 = tf.nn.relu(h2_2)

        h3, h3_w, h3_b = deconv2d(
            h2_2 + h1, [batch_size, s_h4, s_w4, gf_dim*2], name='h3',k_h = 3, k_w = 3, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h3 = tf.nn.relu(g_bn3(h3))
        else:
            h3 = tf.nn.relu(h3)

        h4_1, h4_1_w, h4_1_b = deconv2d(
            h3, [batch_size, s_h4, s_w4, gf_dim*2], name='h4_1',k_h = 3, k_w = 3,d_h=1,d_w=1, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h4_1 = tf.nn.relu(g_bn4_1(h4_1))
        else:
            h4_1 = tf.nn.relu(h4_1)

        h4_2, h4_2_w, h4_2_b = deconv2d(
            h4_1, [batch_size, s_h4, s_w4, gf_dim*2], name='h4_2',k_h = 3, k_w = 3,d_h=1,d_w=1, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h4_2 = tf.nn.relu(g_bn4_2(h4_2))
        else:
            h4_2 = tf.nn.relu(h4_2)

        h5, h5_w, h5_b = deconv2d(
            h4_2 + h3, [batch_size, s_h2, s_w2, gf_dim*1], name='h5',k_h = 3, k_w = 3, with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h5 = tf.nn.relu(g_bn5(h5))
        else:
            h5 = tf.nn.relu(h5)

        h4_mu, h4_mu_w, h4_mu_b = deconv2d(
            h5, [batch_size, s_h, s_w, c_dim], name='h4_mu',k_h = 3, k_w = 3, with_w=True, reuse = tf.AUTO_REUSE)

        # h4_sigma, h4_sigma_w, h4_sigma_b = deconv2d(
        #     h3, [batch_size, s_h, s_w, c_dim], name='h4_sigma', with_w=True, reuse = tf.AUTO_REUSE)

        # print("img recon logit", h4.get_shape().as_list())
        return tf.nn.relu(h4_mu), tf.nn.relu(h4_mu)
'''

def img_decoder(z, output_width, output_height, c_dim):
    with tf.variable_scope('img_decoder', reuse = tf.AUTO_REUSE) as scope:
        g_bn0 = batch_norm(name='img_g_bn0')
        g_bn1 = batch_norm(name='img_g_bn1')
        g_bn2 = batch_norm(name='img_g_bn2')
        g_bn3 = batch_norm(name='img_g_bn3')
        s_w, s_h =  output_width, output_height
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # z_dense = tf.layers.dense(z, 256, activation = tf.nn.relu, name='z_dense', reuse = tf.AUTO_REUSE)

        z_, h0_w, h0_b = linear(z, gf_dim*8*s_h16*s_w16, 'h0_lin', with_w=True, reuse = tf.AUTO_REUSE)

        h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
        if img_decode_bn:
            h0 = tf.nn.relu(g_bn0(h0))
        else:
            h0 = tf.nn.relu(h0)
        print("decoder h0", h0.get_shape().as_list())

        h1, h1_w, h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, gf_dim*4], name='h1', with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h1 = tf.nn.relu(g_bn1(h1))
        else:
            h1 = tf.nn.relu(h1)

        h2, h2_w, h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, gf_dim*2], name='h2', with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h2 = tf.nn.relu(g_bn2(h2))
        else:
            h2 = tf.nn.relu(h2)

        h3, h3_w, h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, gf_dim*1], name='h3', with_w=True, reuse = tf.AUTO_REUSE)
        if img_decode_bn:
            h3 = tf.nn.relu(g_bn3(h3))
        else:
            h3 = tf.nn.relu(h3)

        h4, h4_w, h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, c_dim], name='h4', with_w=True, reuse = tf.AUTO_REUSE)

        # print("img recon logit", h4.get_shape().as_list())
        return tf.nn.tanh(h4), h4

