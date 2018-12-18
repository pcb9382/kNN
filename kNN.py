"""
机器学习实战之kNN
姓名：彭传波
日期：2018.12.13
"""

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
#-*-coding:utf-8-*-

"""
将特征点画出为散点图
"""
def plotScatter(datingDataMat,datingLabels):
    """
    根据特征矩阵画散点图
    :param datingDataMat:特征矩阵
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']                    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False                      # 用来正常显示负号
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    type1_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    labelNum=len(datingLabels)
    for i in range(labelNum):                                       # 1000组数据，i循环1000次
        Num=datingLabels[i]
        if datingLabels[i] == 1:                                    # 根据标签进行数据分类,注意标签此时是字符串
            type1_x.append(datingDataMat[i][0])                     # 取的是样本数据的第一列特征和第二列特征
            type1_y.append(datingDataMat[i][1])

        if datingLabels[i] == 2:
            type2_x.append(datingDataMat[i][0])
            type2_y.append(datingDataMat[i][1])

        if datingLabels[i] == 3:
            type3_x.append(datingDataMat[i][0])
            type3_y.append(datingDataMat[i][1])

    plt.scatter(type1_x, type1_y, s=20, c='r', label='不喜欢')
    plt.scatter(type2_x, type2_y, s=40, c='b', label='魅力一般')
    plt.scatter(type3_x, type3_y, s=60, c='k', label='极具魅力')
    plt.legend()
    plt.xlabel("每年获取飞行常客里程数", fontsize=14)                   #设置x轴的标签以及标签的大小
    plt.ylabel("玩视频游戏所耗时间百分比",fontsize=14)                  #设置y轴的标签以及标签的大小
    plt.show()

"""
将特征数据归一化
"""
def autoNorm(dataSet):
    minVals=dataSet.min(0)                             #当参数为0时，min()函数返回每一列的最小值,
    maxVals=dataSet.max(0)                             #当参数为1时，min()函数返回每一行的最小值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]                                 #得到矩阵dataSet的行数
    normDataSet=dataSet-tile(minVals,(m,1))            #将normDataSet复制为m*3的矩阵
    normDataSet=normDataSet/tile(ranges,(m,1))         #矩阵相除，使得矩阵归一化
    return normDataSet,ranges,minVals

"""
将txt中的数据读入到numpy矩阵中
"""
def file2matrix(filename):
    """
    将txt文件转化成矩阵形式
    :param filename:文本的文件路径
    :return:
    """
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)                                 #得到文件行数
    returnMat=zeros((numberOfLines,3))                               #返回创建的Numpy的矩阵
    classLabelVector=[]                                            #创建一个空列表
    index=0
    for line in arrayOLines:
        line=line.strip()                                          #截取掉所有的回车符
        listFromLine=line.split('\t')                              #使用'\t'将上一步得到的整行数据分割成一个数据列表
        returnMat[index,:]=listFromLine[0:3]                       #选取一行数据中的前三个元素，放置到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))             #将一行的最后一个标签元素放置到标签矩阵中
        index+=1
    return returnMat,classLabelVector

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels

"""
kNN分类器
"""
def classify0(inX,dataSet,labels,k):
    """
    实施kNN算法
    :param inX: 用于分类的输入向量
    :param dataSet:输入的训练样本集
    :param labels:标签向量
    :param k:用于选择最邻近数目
    :return:
    """
    """
    计算距离
    """
    dataSetSize=dataSet.shape[0]                      #shape[1]表示第一维的长度，shape[0]表示第二维的长度，如果是group为（4,2）
    diffMat1=tile(inX,(dataSetSize,1))                #tile表示将给定的inX按照(dataSetSize,1)的方式复制
    diffMat=diffMat1-dataSet
    sqDiffMat=diffMat**2                              #平方
    sqDistances=sqDiffMat.sum(axis=1)                 #是压缩列,即将每一行的元素相加,将矩阵压缩为一列
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()            #将矩阵a按照axis排序，并返回排序后的下标

    """
    选择距离最小的k个点
    """
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

"""
分类针对约会网站的测试代码
"""
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt') #读取txt中的特征向量以及标签
    normMat,ranges,minvals=autoNorm(datingDataMat)              #将特征向量归一化
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range (numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)       #给定前10%用作测试，后90%用作训练
        print('The classifierResult came back with :%d, The real answer is: %d'\
              %(classifierResult,datingLabels[i]))

        if(classifierResult!=datingLabels[i]):
            errorCount+=1

        print('the total error rate is %f'%(errorCount/float(numTestVecs)))

"""
约会网站预测函数
"""
def classifyPerson():
    result=['not at all','in small doses','in large doses']
    percentTat=float(input("percentage of time spend video game?"))#input函数允许用户输入文本命令并返回用户所输入的命令
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTat,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",result[classifierResult-1])

"""
-------------------------------手写识别系统---------------------------

"""

"""
将图像转换为测试向量
"""
def img2vector(filename):
    returnVec=zeros((1,1024))                      #将图像32*32的二进制图像矩阵转换为1*1024的向量
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVec[0,32*i+j]=int(lineStr[j])
    return returnVec

"""
手写数字识别系统的测试代码
"""
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')     #获取目录内容
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):                             #从文件中解析出分类数字,并将图像进行向量转换
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)

    #对测试集进行向量转换以及错误率的计算
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumStr))
        if classifierResult!=classNumStr:
            errorCount+=1.0

    print('\nthe total number of error is:%d'%errorCount)
    print("\nthe total error rate is:%f"%(errorCount/float(mTest)))




def main():

    handwritingClassTest()                              #手写系统的识别
    #classifyPerson()                                   #约会网站的配对效果
    #datingClassTest()                                  #分类针对约会网站的测试代码
    #group,labels=file2matrix('datingTestSet2.txt')     #读txt文件的测试代码
    #autoNorm(group)                                    #特征相邻归一化的测试代码
    #classifyResoult=classify0([0,0],group,labels,3)    #分类结果的测试代码
    #plotScatter(group,labels)                          #散点图代码

if __name__ == '__main__':
    main()