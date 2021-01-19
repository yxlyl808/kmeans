from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def kmeans(dataset, k):
    m = shape(dataset)[0]
    n = shape(dataset)[1]
    clusterAssment = mat(zeros((m, 2)))
    centroids = mat(zeros((k, n)))
    for i in range(n):
        a = min(dataset[:, i])
        b = float(max(dataset[:, i]) - a)
        centroids[:, i] = a + b * random.rand(k, 1)
    flag = True
    count = 0
    while flag:
        flag = False
        count = count+1
        for i in range(m):
            juli = float('inf')
            suoyin = -1
            for j in range(k):
                p = sqrt(sum(power(centroids[j, :] - dataset[i, :], 2))) #欧氏距离计算
                if p < juli:
                    juli = p
                    suoyin = j
            if clusterAssment[i, 0] != suoyin:
                flag = True
            clusterAssment[i, :] = suoyin, juli**2
        for c in range(k):
            centroids[c, :] = mean(dataset[nonzero(clusterAssment[:, 0].A == c)[0]], axis=0)
    print('次数：', count)
    return clusterAssment, centroids

def showCluster(dataSet, k, clusterAssment, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = []
    for c in range(k):
        data.append(dataSet[nonzero(clusterAssment[:, 0].A == c)[0]])
    for d, c, m in zip(range(k), ['r', 'g', 'b'], ['^', 'o', '*']):
        ax.scatter(data[d][:, 0].tolist(), data[d][:, 1].tolist(), s=80, c=c, marker=m)
    ax.scatter(centroids[:, 0].tolist(), centroids[:, 1].tolist(), s=1000, c='black', marker='+', alpha=1)
    plt.show()

iris = load_iris()
d1, d2 = [], []
for i in iris.data:
    d1.append(i[0:2])
    d2.append(i[2:4])
for i in d1, d2:
    dataset = mat(i)
    k = 3
    clusterAssment, centroids = kmeans(dataset, k)
    showCluster(dataset, k, clusterAssment, centroids)
