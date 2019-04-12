#-*- coding: utf-8 -*-
#https://medium.com/trackin-datalabs/input-data-tf-data-으로-batch-만들기-1c96f17c3696
# https://sebastianwallkoetter.wordpress.com/2017/11/26/tfrecords-via-tensorflow-dataset-api/
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py#L66
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py#L100
# https://www.tensorflow.org/tutorials/estimators/cnn
# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/2017/examples/07_convnet_mnist_starter.py
# https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
# https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab
# https://www.kaggle.com/mitch9090/fruit-360-cnn-tensorflow
import tensorflow as tf
import numpy as np
import os
import time
import cv2
import glob

folderName = 'Feb20'
signalType = '2Dsignal'

BATCH_SIZE = 32
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 770

#tfrecords_path = '/Volumes/My Passport Pro/MISEON_DROPBOX/Dropbox/EWHA/LAB/Top engineering/2019/radar/occupancy/data/' + folderName + '/'
tfrecords_path = './'
TRAIN_FILE = tfrecords_path + 'radarOCC_'+folderName+'_'+signalType+'_'+'train.tfrecords'
VALID_FILE = tfrecords_path + 'radarOCC_'+folderName+'_'+signalType+'_'+'validation.tfrecords'
TEST_FILE = tfrecords_path + 'radarOCC_'+folderName+'_'+signalType+'_'+'test.tfrecords'

CHANNELS = 1 # originally 2 in counterfeit detection

LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000

HIDDEN1_UNITS = [3,3,16]
MAXPOOL1_UNITS = [2,2]
HIDDEN2_UNITS = [3,3,32]
MAXPOOL2_UNITS = [2,2]
HIDDEN3_UNITS = [3,3,32]
MAXPOOL3_UNITS = [2,2]
HIDDEN4_UNITS = [3,3,64]
MAXPOOL4_UNITS = [2,2]
FC1_UNITS = 32
DROP_PROB = 0.5

FINAL_IMG_HEIGHT = np.uint32(np.ceil(((((((IMAGE_HEIGHT-HIDDEN1_UNITS[0]+1)/MAXPOOL1_UNITS[0] - HIDDEN2_UNITS[0]+1)/MAXPOOL2_UNITS[0]) - HIDDEN3_UNITS[0]+1)/MAXPOOL3_UNITS[0]) - HIDDEN4_UNITS[0]+1)/MAXPOOL4_UNITS[0]))
FINAL_IMG_WIDTH = np.uint32(np.ceil(((((((IMAGE_WIDTH-HIDDEN1_UNITS[1]+1)/MAXPOOL1_UNITS[1] - HIDDEN2_UNITS[1]+1)/MAXPOOL2_UNITS[1]) - HIDDEN3_UNITS[1]+1)/MAXPOOL3_UNITS[1]) - HIDDEN4_UNITS[1]+1)/MAXPOOL4_UNITS[1]))
print(FINAL_IMG_WIDTH)
print(FINAL_IMG_HEIGHT)
tf.device('/gpu:0')

def inference(images, phase, hidden1_units, maxpool1_units, hidden2_units, maxpool2_units, hidden3_units, maxpool3_units, hidden4_units, maxpool4_units, fc1_units):
#    Args:
#        images: Image placeholder, from inputs()
#        hidden1_units : Size of the first hidden layer
#        maxpool1_units : Size of the first max pooling layer
#        hidden2_units : Size of the second hidden layer
#        maxpool2_units : Size of the second max pooling layer
#        phase : if phase == True, training, else, testing
#                -> Training :
#                    Normalize layer activations according to mini-batch statistics.
#                    During the training step, update population statistics approximation
#                    via moving average of mini-batch statistics
#                -> Testing :
#                    Normalize layer activations according to estimated population statistics.
#                    Do not update population statistics according to mini-batch statistics from test data.
#        Returns :
#            logits : Output tensor with the computed logits.

    # Hidden 1
    with tf.name_scope('convLayer1'):
        conv1 = tf.layers.conv2d(inputs=images, filters= hidden1_units[2],kernel_size = hidden1_units[0:2], padding="valid", activation = None, name = 'conv1')
        # Because of batch normalization, bias can be ignored
        # since the effect of beta parameter of batch normalization will be canceled by the subsequent mean subtraction
        batch_norm1 = tf.layers.batch_normalization(conv1, center=True, scale=True, training=phase,name = 'batch_norm1')
        # Batch normalization should be applied before nonlinearity to be more Gaussian distribution.
        hidden1 = tf.nn.relu(batch_norm1, name='relu1')

    # max pooling1
    with tf.name_scope('maxPool1'):
        maxpool1 = tf.layers.max_pooling2d(hidden1, pool_size= maxpool1_units , strides=2, padding='SAME', name="maxpool1")

    # Hidden 2
    with tf.name_scope('convLayer2'):
        conv2 = tf.layers.conv2d(inputs=maxpool1, filters=hidden2_units[2], kernel_size=hidden2_units[0:2], padding="valid", activation = None, name='conv2')
        batch_norm2 = tf.layers.batch_normalization(conv2, center=True, scale=True, training=phase, name='batch_norm2')
        hidden2 = tf.nn.relu(batch_norm2, name='relu2')

    # maxpooling2
    with tf.name_scope('maxPool2'):
        maxpool2 = tf.layers.max_pooling2d(hidden2, pool_size=maxpool2_units, strides=2, padding="same", name="maxpool2")

    # Hidden 3
    with tf.name_scope('convLayer3'):
        conv3 = tf.layers.conv2d(inputs=maxpool2, filters=hidden3_units[2], kernel_size=hidden3_units[0:2], padding="valid", activation = None, name='conv3')
        batch_norm3 = tf.layers.batch_normalization(conv3, center=True, scale=True, training=phase, name='batch_norm3')
        hidden3 = tf.nn.relu(batch_norm3, name='relu3')

    # maxpooling3
    with tf.name_scope('maxPool3'):
        maxpool3 = tf.layers.max_pooling2d(hidden3, pool_size=maxpool3_units, strides=2, padding="same", name="maxpool3")

    # Hidden 4
    with tf.name_scope('convLayer4'):
        conv4 = tf.layers.conv2d(inputs=maxpool3, filters=hidden4_units[2], kernel_size=hidden4_units[0:2], padding="valid", activation = None, name='conv4')
        batch_norm4 = tf.layers.batch_normalization(conv4, center=True, scale=True, training=phase, name='batch_norm4')
        hidden4 = tf.nn.relu(batch_norm4, name='relu4')

    # maxpooling4
    with tf.name_scope('maxPool4'):
        maxpool4 = tf.layers.max_pooling2d(hidden4, pool_size=maxpool4_units, strides=2, padding="same", name="maxpool4")
        print(maxpool4.get_shape())

    with tf.name_scope('drop1'):
        drop = tf.layers.dropout(maxpool4, DROP_PROB,training=phase,name="dropout1")

    # Fully connected layer 1
    with tf.name_scope('FC1'):
        # Downsampled image due to maxpooling layers
#       maxPoolSize = np.array(maxpool1_units)*np.array(maxpool2_units)
#       fcSize = np.int32(np.ceil(np.array([IMAGE_HEIGHT, IMAGE_WIDTH])/maxPoolSize))
        flat = tf.reshape(maxpool4,[-1,FINAL_IMG_HEIGHT*FINAL_IMG_WIDTH*hidden4_units[2]])
        print(flat.get_shape())
        dense1 = tf.layers.dense(inputs=flat, units=fc1_units, activation=None,name="fc1")
        batch_norm5 = tf.layers.batch_normalization(dense1, center=True, scale=True, training=phase, name='batch_norm_fc1')
        fullyConnected1 = tf.nn.relu(batch_norm5, name='relu_fc2')


    # Fully connected layer 2
    with tf.name_scope('FC2'):
        CF_logits = tf.layers.dense(inputs=fullyConnected1, units=2, activation = None,name="logits")

    return CF_logits


def loss_function(logits, labels):
    # Calculates the loss from the logits and the labels.
#
#    Args:
#        logits: Logits tensor, float, [batch_size, NUM_CLASSES=2]
#        labels: Lables tensor, int32, [batch_size]
#
#    Returns:
#        loss: Loss tensor of type float.
    labels = tf.cast(labels,tf.float32)
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits= logits, labels = labels), name="loss")
    return loss

def accuracy_function(logits, labels):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits,1), tf.argmax(labels,1)),tf.float32),name="acc")
    return accuracy

def training(loss, learning_rate):
    # Sets up the training Ops
    # Creates a summarizer to track the loss over time in Tensorboard
    # Creates an optimizer and applies the gradients to all trainable variables
    # The Op returned by this function is what must be passed to the 'sess.run()'
    # call to cause the model to train.
#
#    Args:
#        loss: loss tensor, from loss().
#        learning_rate: The learning rate to use for gradient descent.
#
#    Returns:
#        train_op: The Op for training

#    # Add a scalar summary for the snapshot loss.
#    tf.summary.scalar('loss',loss)

    # Create the Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def parse(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_height': tf.FixedLenFeature([], tf.int64),
                'image_width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
                })
    # Convert from a scalar string tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image,tf.float32)
    image_height = tf.cast(features['image_height'], tf.int32)
    image_width = tf.cast(features['image_width'], tf.int32)
    image = tf.reshape(image,[image_height, image_width,1])

    ### 수정 필요
    image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width = IMAGE_WIDTH)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label,2)
    return image, label

def make_iterator(filenames) :
    # Repeat : Create duplicates of the existing data in Dataset
    dataNum = sum(1 for _ in tf.python_io.tf_record_iterator(filenames))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)

    # Shuffle : randomly shuffle the data
    dataset = dataset.shuffle(dataNum)

    # Batch : sequentially divide dataset by the specified batch size.
    padded_shapes = (tf.TensorShape([IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]), tf.TensorShape([2]))
    dataset = dataset.padded_batch(BATCH_SIZE,padded_shapes=padded_shapes,drop_remainder=False)
    iterator = dataset.make_initializable_iterator()
    return iterator


def run_training():
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS], name="input")
        y = tf.placeholder(tf.int32,shape=[None,2], name="label")
        train = tf.placeholder(tf.bool,name="istrain")

        tr_iterator = make_iterator(TRAIN_FILE)
        tr_image_batch, tr_label_batch = tr_iterator.get_next()
        vl_iterator = make_iterator(VALID_FILE)
        vl_image_batch, vl_label_batch = vl_iterator.get_next()
        test_iterator = make_iterator(TEST_FILE)
        test_image_batch, test_label_batch = test_iterator.get_next()

        # Build a graph that computes predictions from the inference model.
        logits = inference(x, train, HIDDEN1_UNITS, MAXPOOL1_UNITS, HIDDEN2_UNITS, MAXPOOL2_UNITS, HIDDEN3_UNITS, MAXPOOL3_UNITS, HIDDEN4_UNITS, MAXPOOL4_UNITS, FC1_UNITS)
        softmaxOut = tf.nn.softmax(logits,name="softmax")

        # loss
        loss = loss_function(logits, y)
        tf.summary.scalar('loss',loss)

        # accuracy
        accuracy = accuracy_function(logits, y)
        tf.summary.scalar('accuracy',accuracy)

        # Add to the Graph operations that train the model
        train_op = training(loss, LEARNING_RATE)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Create a session for running operations in the Graph.
        merged = tf.summary.merge_all()

        with tf.Session().as_default() as sess:
            # Initialize the variables (the trained variables and the epoch counter)
            print("BATCH_SIZE : ",BATCH_SIZE)
            sess.run(init_op)
            saver = tf.train.Saver()

            # To visualize using Tensorboard
            train_writer = tf.summary.FileWriter('./train/', sess.graph)
            valid_writer = tf.summary.FileWriter('./valid/', sess.graph)

            #### You have to create folders to store checkpoints
            ckpt = tf.train.get_checkpoint_state('./checkpoints/checkpoint')

            print('start training')
            epoch_count = 0
            tr_prev_loss = 0
            prev_count = 0
            start_time =time.time()
            for epoch in range(NUM_EPOCHS):
                if epoch > 0:
                    tr_prev_loss = tr_loss
                tr_loss = 0.0
                tr_acc = 0.0
                vl_loss = 0.0
                vl_acc = 0.0
                count = 0

                # training
                sess.run(tr_iterator.initializer)
                try:
                    while True:
                        images, labels = sess.run([tr_image_batch, tr_label_batch])
                        count += 1
                        _, _, loss_val, acc_val = sess.run([train_op, logits, loss, accuracy], feed_dict={train:True, x:images, y:labels})
                        tr_acc += acc_val
                        tr_loss += loss_val
                except tf.errors.OutOfRangeError:
                    summary = sess.run(merged,feed_dict={train:True, x:images, y:labels})
                    train_writer.add_summary(summary,epoch)
                    pass
                if tr_loss < 0.005:
                    epoch_count +=1
                    print("epoch_count : ",epoch_count)
                else:
                    epoch_count = 0
                # validation
                sess.run(vl_iterator.initializer)
                count = 0
                try:
                    while True:
                        images, labels = sess.run([vl_image_batch, vl_label_batch])
                        count += 1
                        _, loss_val, acc_val = sess.run([logits,loss, accuracy], feed_dict={train:False,x:images, y:labels})
                        vl_loss += loss_val
                        vl_acc += acc_val
                except tf.errors.OutOfRangeError:
                    summary = sess.run(merged,feed_dict={train:False,x:images,y:labels})
                    valid_writer.add_summary(summary, epoch)
                    pass
                print('#####Done training for %d epochs. tr_loss : %.5f tr_acc : %.5f vl_loss : %.5f vl_acc : %.5f' %(epoch,tr_loss,tr_acc,vl_loss,vl_acc))
                if epoch_count > 5:
                    break
            saver.save(sess, 'checkpoints/radarOCC_model_'+folderName+signalType+'_fix.ckpt')
            print("Total time : {0} seconds".format(time.time()-start_time))
            tf.train.write_graph(sess.graph.as_graph_def(), "./checkpoints/","graph.pb")

            test_loss = 0.0
            test_acc = 0.0
            sess.run(test_iterator.initializer)
            try:
                while True:
                    images, labels = sess.run([test_image_batch, test_label_batch])
                    count += 1
                    _, loss_val, acc_val = sess.run([logits,loss,accuracy], feed_dict={train:False,x:images, y:labels})
                    test_acc += acc_val
                    test_loss += loss_val
            except tf.errors.OutOfRangeError:
                pass
            print('#####Finally test. test_loss : %.5f test_acc : %.5f' %(test_loss,test_acc))

def test(filename):
    graph = tf.get_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('checkpoints/radarOCC_model_'+folderName+signalType+'_fix.ckpt.meta')
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "checkpoints/radarOCC_model_"+folderName+signalType+'_fix.ckpt')
    x = graph.get_tensor_by_name("input:0")
    y = graph.get_tensor_by_name("label:0")
    train = graph.get_tensor_by_name("istrain:0")
    out = graph.get_tensor_by_name("softmax:0")

    test_iterator = make_iterator(filename)
    test_image_batch, test_label_batch = test_iterator.get_next()

    errorNum = 0
    sess.run(test_iterator.initializer)
    try:
        while True:
            images, labels = sess.run([test_image_batch, test_label_batch])
            softmax= sess.run([out], feed_dict={train:False, x:images, y:labels})

            predict = np.argmax(softmax,1)
            groundTruth = np.argmax(labels,1)
            errorNum += sum(abs(predict-groundTruth))

    except tf.errors.OutOfRangeError:
        pass
    print('$$$$$Final errorNum : %d ' %(errorNum))

def main():
    run_training()
#    test(TRAIN_FILE)
#    test(VALID_FILE)
#    test(TEST_FILE)
if __name__ == '__main__':
    main()
