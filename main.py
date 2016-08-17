

import os.path
import time

import numpy as np
import tensorflow as tf
import cv2

import bouncing_balls as b
import layer_def as ld


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('hidden_size', 20,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('max_step', 50000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .5,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """batch size for training""")


def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3])

    # possible dropout inside
    keep_prob = tf.placeholder("float")

    # create network
    # encodeing part first
    # conv1
    conv1 = ld.conv_layer(x, 3, 1, 32, "encode_1")
    # conv2
    conv2 = ld.conv_layer(conv1, 3, 2, 64, "encode_2")
    # conv3
    conv3 = ld.conv_layer(conv2, 3, 1, 64, "encode_3")
    # conv4
    conv4 = ld.conv_layer(conv3, 3, 2, 128, "encode_4")
    # conv5
    conv5 = ld.conv_layer(conv4, 1, 1, 64, "encode_5")
    # fc5 
    fc5 = ld.fc_layer(conv5, 512, "encode_6", True, False)
    # dropout maybe
    fc5_dropout = tf.nn.dropout(fc5, keep_prob)
    # y 
    y = ld.fc_layer(fc5_dropout, (FLAGS.hidden_size) * 2, "encode_7", False, True)
    mean, stddev = tf.split(1, 2, y)
    stddev =  tf.sqrt(tf.exp(stddev))
    # now decoding part
    # sample distrobution
    epsilon = tf.random_normal(mean.get_shape())
    y_sampled = mean + epsilon * stddev
    # fc8
    fc8 = ld.fc_layer(y_sampled, 512, "decode_8", False, False)
    # fc9
    fc9 = ld.fc_layer(fc8, 64*8*8, "decode_9", False, False)
    conv9 = tf.reshape(fc9, [-1, 8, 8, 64])
    # conv10
    conv10 = ld.transpose_conv_layer(conv9, 1, 1, 128, "decode_10")
    # conv11
    conv11 = ld.transpose_conv_layer(conv10, 3, 2, 64, "decode_11")
    # conv12
    conv12 = ld.transpose_conv_layer(conv11, 3, 1, 64, "decode_12")
    # conv13
    conv13 = ld.transpose_conv_layer(conv12, 3, 2, 32, "decode_13")
    # conv14
    conv14 = ld.transpose_conv_layer(conv13, 3, 1, 3, "decode_14", True) # set activation to linear!!!!!!!!1
    # x_prime
    x_prime = conv14
    x_prime = tf.nn.sigmoid(x_prime)

    # now calc loss 
    epsilon = 1e-8
    # calc loss from vae
    loss_vae = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                         2.0 * tf.log(stddev + epsilon) - 1.0))
    # log loss for reconstruction
    loss_reconstruction = tf.reduce_sum(-x * tf.log(x_prime + epsilon) -
                  (1.0 - x) * tf.log(1.0 - x_prime + epsilon)) 
    # save for tensorboard
    tf.scalar_summary('loss_vae', loss_vae)
    tf.scalar_summary('loss_reconstruction', loss_reconstruction)
    # calc total loss 
    loss = tf.reduce_mean(loss_vae + loss_reconstruction)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      dat = b.bounce_vec(32, FLAGS.num_balls, FLAGS.batch_size)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t
      print(elapsed)

      if step%100 == 0:
        _ , loss_vae_r, loss_reconstruction_r, y_sampled_r, x_prime_r = sess.run([train_op, loss_vae, loss_reconstruction, y_sampled, x_prime],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("loss vae value at " + str(loss_vae_r))
        print("loss reconstruction value at " + str(loss_reconstruction_r))
        print("min sampled vector " + str(np.min(y_sampled_r)))
        print("max sampled vector " + str(np.max(y_sampled_r)))
        print("time per batch is " + str(elapsed))
        cv2.imwrite("real_balls.jpg", np.uint8(dat[0, :, :, :]*255))
        cv2.imwrite("generated_balls.jpg", np.uint8(x_prime_r[0, :, :, :]*255))
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()



