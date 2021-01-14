import random
import tensorflow as tf
import tensorflow as tf
import os
import numpy as np

def create_record(records_path, data_path):
    # 声明一个TFRecordWriter
    writer = tf.io.TFRecordWriter(records_path)
    # 读取图片信息，并且将读入的图片顺序打乱
    img_list = []
    filename = os.listdir(data_path)

    g = os.walk(data_path)
    training_set = []
    for path,dir_list,file_list in g:
        file_list.sort()
        for file_name in file_list:  
            image = tf.io.read_file(data_path + '/' + file_name)
            image = tf.image.decode_jpeg(image)
            training_set.append(image.numpy())   

    cnt = 0
    for i in range(len(training_set)):
        data_raw = training_set[i].tobytes()
        # 声明将要写入tfrecord的key值（即图片，标签）
        example = tf.train.Example(
           features=tf.train.Features(feature={
                "ID": tf.train.Feature(float_list=tf.train.FloatList(value=[float(file_name[:-4])])),
                "width": tf.train.Feature(float_list=tf.train.FloatList(value=[float(training_set[i].shape[0])])),
                "height": tf.train.Feature(float_list=tf.train.FloatList(value=[float(training_set[i].shape[1])])),
                'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
           }))
        # 将信息写入指定路径
        writer.write(example.SerializeToString())

        cnt += 1

    writer.close()   
    
# 指定你想要生成tfrecord名称，图片文件夹路径，含有图片信息的txt文件

records_path = '/lustre/lwork/csun001/DevData.tfrecords'
data_path = '/lustre/lwork/csun001/Dev'

create_record(records_path, data_path)