# -*- coding: utf-8 -*-
# BP回归,bp就是多层感知器
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import savemat
# In[] 加载数据

np.random.seed(0)
xlsfile=pd.read_excel('BP2008.xlsx').iloc[0:,2:]# 前两列的日期不作为特征之一
data=np.array(xlsfile)
# 拟用前一时期的7个特征与预测时期的aqi作为输入，预测时期的 6 种空气污染物浓度作为输出
in_=data[0:-1,:]
out_=data[1:,0:]
#划分数据，一共1740个样本 选择1571个样本作为训练集 24个测试集
n=range(in_.shape[0])
m=108110 #
train_data = in_[n[0:m],]
test_data = in_[n[m:],]
train_label = out_[n[0:m],]
test_label = out_[n[m:],]
# 归一化
ss_X=MinMaxScaler(feature_range=(0,1)).fit(train_data)
ss_y=MinMaxScaler(feature_range=(0,1)).fit(train_label)
train_data = ss_X.transform(train_data)
train_label = ss_y.transform(train_label)

test_data = ss_X.transform(test_data)
test_label = ss_y.transform(test_label)


clf = MLPRegressor(max_iter=50,hidden_layer_sizes=(27, 27))
clf.fit(train_data,train_label)
test_pred=clf.predict(test_data)
test_pred[test_pred<0]=0
# In[] 画出测试集的值

# 对测试结果进行反归一化
test_pred = ss_y.inverse_transform(test_pred.reshape(-1,1))
test_label = ss_y.inverse_transform(test_label.reshape(-1,1))




test_label=test_label.reshape(-1,1)
test_pred=test_pred.reshape(-1,1)
plt.figure()
plt.plot(test_label,c='r', label='true')
plt.plot(test_pred,c='b',label='predict')
title='bp'
plt.title(title)
plt.xlabel('hours')
plt.ylabel('Power load volume')
plt.legend()
# plt.show()

savemat('D:\建模\结果/bp_result.mat',{'true':test_label,'pred':test_pred})
# In[]计算各种指标
# mape
test_mape=np.mean(np.abs((test_pred-test_label)/test_label))
# rmse
test_rmse=np.sqrt(np.mean(np.square(test_pred-test_label)))
# mae
test_mae=np.mean(np.abs(test_pred-test_label))
# R2
test_r2=r2_score(test_label,test_pred)

print('MLP测试集的mape:',test_mape,' rmse:',test_rmse,' mae:',test_mae,' R2:',test_r2)
test_pred = pd.DataFrame(test_pred)

test_pred.to_csv("D:\建模\结果\BP.csv")