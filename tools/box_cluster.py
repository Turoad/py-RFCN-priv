#! /usr/bin/env python
#################################################################################
#     File Name           :     box_cluster.py
#     Created By          :     Kuoliang Wu
#     Creation Date       :     [2017-10-03 09:23]
#     Last Modified       :     [2018-03-08 11:43]
#     Description         :
#################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
import cPickle

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

def iouDistance(x, y):
    I = np.minimum(x[0],y[0])*np.minimum(x[1],y[1])
    U = float(x[0])*x[1] + float(y[0])*y[1] - I
    IoU = float(I)/float(U)
    return 1-IoU

def iouDistance_3D(x, y):
    I = np.minimum(x[0],y[0])*np.minimum(x[1],y[1])*np.minimum(x[2],y[2])
    U = float(x[0])*x[1]*x[2] + float(y[0])*y[1]*y[2] - I
    IoU = float(I)/float(U)
    return 1-IoU

def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist  = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = iouDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)

    print 'Congratulations, cluster complete!'
    print centroids
    print clusterAssment
    return centroids, clusterAssment

# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "Sorry! Your k is too large! please contact Zouxy"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)

    plt.show()
    plt.savefig('kmeans.png')

#def cluster(roidb_cache, cluster_num):

roidb_cache = '../data/cache/trainval_gt_roidb.pkl'
cluster_num = 9

with open(roidb_cache, 'rb') as fid:
    roidb = cPickle.load(fid)
boxes = []
for i in range(len(roidb)):
    boxes.append(roidb[i]['boxes'])

boxes = np.concatenate(boxes)

box_size = np.concatenate([boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]])
box_size = np.reshape(box_size, (boxes.shape[0], 2), 'F')

centroids, clusterAssment = kmeans(box_size, cluster_num)

file_name = roidb_cache.split('/')[-1][:-4]+'_clusters'
# print file_name
np.save(file_name, centroids)
