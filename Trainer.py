import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from Utils import utils
from Utils import Tools
import Model
from Config import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

session=tf.compat.v1.Session()
images_ph=tf.placeholder(tf.float32,shape=[None,height,width,color_channels])
labels_ph=tf.placeholder(tf.float32,shape=[None,number_of_classes])

#training happens here
def trainer(network,number_of_images):
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=labels_ph)

    cost=tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)

    optimizer=tf.train.AdamOptimizer().minimize(cost)
    session.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(model_save_name, graph=tf.get_default_graph())
    merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=4)
    counter=0
    for epoch in range(epochs):
        tools = utils()
        for batch in range(int(number_of_images / batch_size)):
            counter+=1
            images, labels = tools.batch_dispatch()
            if images == None:
                break
            loss,summary = session.run([cost,merged], feed_dict={images_ph: images, labels_ph: labels})
            print('loss', loss)
            session.run(optimizer, feed_dict={images_ph: images, labels_ph: labels})

            print('Epoch number ', epoch, 'batch', batch, 'complete')
            writer.add_summary(summary,counter)
        saver.save(session, os.path.join(model_save_name))

if __name__=="__main__":
    tools=utils()
    model=Tools()
    network=Model.generate_model(images_ph,number_of_classes)
    print (network)
    number_of_images = sum([len(files) for r, d, files in os.walk("rawdata/data")])
    trainer(network,number_of_images)