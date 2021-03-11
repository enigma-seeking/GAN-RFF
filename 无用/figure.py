import numpy as np
from math import pi
import matplotlib.pyplot as plt

#rDATA = np.load("rDATA.npy")
#fDATA = np.load("fDATA.npy")
#retA = list(set(rDATA).intersection(set(fDATA)))
rDATA_random_point = np.load("IIrDATA_rand_point.npy")
fDATA_random_point = np.load("IIfDATA_rand_point.npy")

_,_ ,_,wr= np.unique(rDATA_random_point,return_counts=True,return_index=True,return_inverse=True)
_,_ ,_,wf= np.unique(fDATA_random_point,return_counts=True,return_index=True,return_inverse=True)
#print(len(retA)/400000)
#plt.plot(rDATA_rand_point)
#plt.show()

print(wr)

print(wf)

#求均值
arr_meanr = np.mean(wr)
#求方差
arr_varr = np.var(wr)
#求标准差
arr_stdr = np.std(wr, ddof=1)
print("平均值为：%f" % arr_meanr)
print("方差为：%f" % arr_varr)
print("标准差为:%f" % arr_stdr)

#求均值
arr_meanf = np.mean(wf)
#求方差
arr_varf = np.var(wf)
#求标准差
arr_stdf = np.std(wf, ddof=1)
print("平均值为：%f" % arr_meanf)
print("方差为：%f" % arr_varf)
print("标准差为:%f" % arr_stdf)