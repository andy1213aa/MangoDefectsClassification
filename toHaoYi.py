import csv
import numpy as np
import operator
import os
import tensorflow as tf
from V2.model import AE
from PIL import Image
import tensorflow.keras as keras
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# 開啟 CSV 檔案
with open('dev.csv', newline='', encoding='utf-8-sig') as csvfile:
    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)
    sortedlist = sorted(rows, key=operator.itemgetter(0))
    #print(sortedlist[-2])
    label_1 = []
    label_2 = []
    label_3 = []
    label_4 = []
    label_5 = []
    # 以迴圈輸出每一列
    for row in sortedlist:
        #print(row[0])
        if "不良-著色不佳" in row:
            label_1.append(1)
        else:
            label_1.append(0)
        if "不良-炭疽病" in row:
            label_2.append(1)
        else:
            label_2.append(0)
        if "不良-乳汁吸附" in row:
            label_3.append(1)
        else:
            label_3.append(0)
        if "不良-黑斑病" in row:
            label_4.append(1)
        else:
            label_4.append(0)
        if "不良-機械傷害" in row:
            label_5.append(1)
        else:
            label_5.append(0)

    label_1 = np.array(label_1)
    label_2 = np.array(label_2)
    label_3 = np.array(label_3)
    label_4 = np.array(label_4)
    label_5 = np.array(label_5)
    #print(label_1.shape)

##################################
allFile = os.listdir("Dev")
allFile.sort()

count = 0
batchSize = 128
# vector = []
#print(allFile)
inputs = tf.keras.Input(shape=(256, 256, 3), name='img')

modelAE = AE(32) #32 is the hyperparameter of filter size
modelAE.load_weights("/lustre/lwork/hyyang003/ImageRecognition/V2/trained_ckpt")
modelAE.call(inputs) 

m = keras.models.Model(inputs=inputs, outputs = modelAE.layers[-1].output)

en = keras.models.Model(inputs=inputs, outputs = m.layers[6].output)
del m

def generateData(tfRecordDir):
    def _parse_function(example_proto):
        features = tf.io.parse_single_example(
            example_proto,
            features={
                "ID": tf.io.FixedLenFeature([], tf.float32),
                "width": tf.io.FixedLenFeature([], tf.float32),
                "height": tf.io.FixedLenFeature([], tf.float32),
                'data_raw': tf.io.FixedLenFeature([], tf.string)
            }
        )
        ID = features['ID']
        width = features['width']
        height = features['height']
        data = features['data_raw']
        data = tf.io.decode_raw(data, tf.uint8)
        data = tf.reshape(data, [width, height, 3])
        data = tf.cast(data, tf.float32)
        data = tf.image.resize(data, [256, 256], method='bicubic')
        data = (data - 127.5)/127.5
        return data

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = tf.data.TFRecordDataset(tfRecordDir)
    data = data.map(_parse_function, num_parallel_calls=AUTOTUNE)

    testing_batch = data.batch(batchSize, drop_remainder = True)
    testing_batch = testing_batch.prefetch(buffer_size = AUTOTUNE)
    return testing_batch

tfRecordDir = '/home/csun001/finalProject/DevData.tfrecords'   
train_data = generateData(tfRecordDir)

vector = np.zeros((1, 4*4*512))
for _, batch in enumerate(train_data):
    tmpOutput = en(batch)
    tmpOutput = tf.reshape(tmpOutput, [batchSize, -1])
    vector = np.concatenate((vector, tmpOutput), axis=0)
vector = vector[1:]
# for f in allFile:
#     f = "Dev/" + f 
#     img = Image.open(f)
#     img = np.array(img)
#     img = tf.cast(img, tf.float32)
#     img = tf.image.resize(img, [256, 256], method='bicubic')
#     img = tf.reshape(img, [1, 256, 256, 3])
#     img = (img - 127.5)/127.5


#     code = en(img).numpy()
#     code = code.reshape([4*4*512])
#     vector.append(code)
#     count +=1
#     if count > 3:
#         break 
# vector = np.array(vector)
print(vector)
###########################3
def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def cal_perplexity(dist, idx=0, beta=1.0):
    '''计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    这里的perp仅计算了熵，方便计算
    '''
    prob = np.exp(-dist * beta)
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    if sum_prob < 1e-12:
        prob = np.maximum(prob, 1e-12)
        perp = -12
    else:
        perp = np.log(sum_prob) + beta * np.sum(dist * prob) / sum_prob
        prob /= sum_prob

    return perp, prob

def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''二分搜索寻找beta,并计算pairwise的prob
    '''

    # 初始化参数
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    dist[dist < 0] = 0
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # 取log，方便后续计算
    base_perp = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))

        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # 更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # 记录prob值
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    #每个点对其他点的条件概率分布pi\j
    return pair_prob

def tsne(x, no_dims=2, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1

    (n, d) = x.shape

    # 动量
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    # 随机初始化Y
    y = np.random.randn(n, no_dims)
    # dy梯度
    dy = np.zeros((n, no_dims))
    # iy是什么
    iy = np.zeros((n, no_dims))

    gains = np.ones((n, no_dims))

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)   #pij
    # early exaggeration
    # pi\j，提前夸大
    print ("T-SNE DURING:%s" % time.clock())
    P = P * 4
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)   #qij
        Q = np.maximum(Q, 1e-12)    #X与Y逐位比较取其大者

        # Compute gradient
        # np.tile(A,N) 重复数组AN次 [1],5 [1,1,1,1,1]
        # pij-qij
        PQ = P - Q
        # 梯度dy
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        # 迭代
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function\
        if (iter + 1) % 100 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)
            if (iter+1) != 100:
                ratio = C/oldC
                print("ratio ", ratio)
                if ratio >= 0.95:
                    break
            oldC = C
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y

data_2d = tsne(vector, 2)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c = label_5[:count])
plt.show()
plt.savefig('foo.png')

#############################
#LDA

def lda(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''

    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #within_class scatter matrix
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #between_class scatter matrix
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw)*SB
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim