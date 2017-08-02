from __future__ import print_function

import numpy as np
import tensorflow as tf
import cPickle as pickle
import math

#length = 221135
image_width = 72
image_height= 52
pixel_depth = 255


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def ChangeLabel(dataset_label):
    data=np.zeros((np.shape(dataset_label)[0],6))
    for i in range(np.shape(dataset_label)[0]):
        data[i,dataset_label[i]]=1
    return data

# Mutilayer model
sess = tf.InteractiveSession()

#Input data
with open('dataset.pickle','rb') as f:
    save = pickle.load(f)
    dataset_label = save['label']
    dataset_img = save['img']

with open('testset.pickle','rb') as f:
    save = pickle.load(f)
    test_label = save['label']
    test_img = save['img']

dataset_img = dataset_img[:,0:3744] # change to 52*72
dataset_label=ChangeLabel(dataset_label)

train_label=dataset_label[0:200000,:]
train_img= dataset_img[0:200000,:]


valid_label=dataset_label[200001:210000,:]
valid_img=dataset_img[200001:210000,:]


test_label=ChangeLabel(test_label)
test_img=test_img[:,0:3744]


n_sample = len(train_img)
x = tf.placeholder(tf.float32, shape=[None,image_width*image_height])
y_ = tf.placeholder(tf.float32, shape=[None, 6])


#layer 1
W_conv1 = weight_variable([5,5,1,20])
#variable_summaries(W_conv1)
b_conv1 = bias_variable([20])
#variable_summaries(b_conv1)
x_image = tf.reshape(x,[-1,52,72,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#tf.summary.histogram('activations',h_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layer 2
W_conv2 = weight_variable([5, 5, 20, 50])
#variable_summaries(W_conv2)
b_conv2 = bias_variable([50])
#variable_summaries(b_conv2)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#tf.summary.histogram('activations',h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#FCN
W_fc1 = weight_variable([10 * 15 * 50, 800])
#variable_summaries(W_fc1)
b_fc1 = bias_variable([800])
#variable_summaries(b_fc1)

h_pool2_flat = tf.reshape(h_pool2, [-1, 10*15*50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#tf.summary.histogram('activations',h_fc1)

keep_prob = tf.placeholder(tf.float32)
#tf.summary.scalar('dropout_keep_probability',keep_prob)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([800, 6])
b_fc2 = bias_variable([6])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/media/songguoxian/b1efdde7-81bb-4db1-897b-b5ff506288bc/songguoxian/Dataset/'
                                     'NetworkRecord/train', sess.graph)
test_writer = tf.summary.FileWriter( '/media/songguoxian/b1efdde7-81bb-4db1-897b-b5ff506288bc/songguoxian/Dataset/'
                                     'NetworkRecord/test')

batch_size = 200
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):

        offset = (i * batch_size) % (train_label.shape[0] - batch_size)
        batch_data = train_img[offset:(offset + batch_size), :]
        batch_labels = train_label[offset:(offset + batch_size), :]
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_data, y_: batch_labels, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        #train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})
        train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

    train_writer.close()
    test_writer.close()

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: test_img, y_:test_label, keep_prob: 1.0}))

