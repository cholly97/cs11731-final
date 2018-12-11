import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# training and plot from https://github.com/wiseodd/generative-models/blob/master/VAE/vanilla_vae/vae_tensorflow.py

mnist_moving = np.load(open('../data/mnist_test_seq.npy', mode = 'rb')) # (depth: 20, batch_size: 10000, height: 64, width: 64)
mnist_moving = np.float32(mnist_moving/255)
num_frames = mnist_moving.shape[0]
num_examples = mnist_moving.shape[1]
output_depth = 8
output_height = mnist_moving.shape[2]
output_width = mnist_moving.shape[3]
mb_size = 16
z_dim = 128
c_dim = 1

num_plots = 16

def plot(samples):
    fig = plt.figure(figsize=(output_depth, num_plots))
    gs = gridspec.GridSpec(num_plots, output_depth)
    gs.update(wspace=0.05, hspace=0.05)

    for i, images in enumerate(samples):
        for j, image in enumerate(images):
            ax = plt.subplot(gs[i * output_depth + j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(image.reshape(output_width, output_height), cmap='Greys_r')

    return fig

X = tf.placeholder(tf.float32, shape=[mb_size, output_depth, output_width, output_height, c_dim])
z = tf.placeholder(tf.float32, shape=[num_plots, z_dim])

z_mu, z_logvar, z_sample = img_encoder(X[:,:-1,])
# z_mu, z_logvar, z_sample = img_encoder(X)
tanh_h4, h_4 = img_decoder(z_sample, output_depth, output_width, output_height, c_dim)

tanh_h4_sample, h_4_sample = img_decoder(z, output_depth, output_width, output_height, c_dim)

mean_squared_loss = tf.reduce_mean(tf.squared_difference(tanh_h4, X), [1,2,3,4])
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
vae_loss = mean_squared_loss + kl_loss

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb = #TODO

    _, loss = sess.run([solver, vae_loss], feed_dict={X: X_mb})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {}'.format(loss.tolist()))
        print()

        # samples = sess.run(tanh_h4_sample, feed_dict={z: np.random.randn(num_plots, z_dim)})
        samples = sess.run(tanh_h4, feed_dict={X: X_mb})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)