#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'zq'

##修改几何上前两列（前81个点）参数和坐标（编号）的对应关系
##修改真实参数
from pandas import Series
import numpy as np


##修改样本参数,传入的para为一维数组
def para_keys_modify(para):
    content=np.array(para)
    content=np.float64(content)
    content=Series(content)
    content1=content.copy()
    for i in range(1,41):
        content1[i*2+1]=content[i]
    content1[1]=content[41]
    for i in range(42,82):
        content1[(i-41)*2]=content[i]
    content1=content1.values
    return content1


#def truepara_key_modify():
#    root_directory=r'E:\EnRML_Gas_Modelling\true_obs'
#    with open(r'E:\EnRML_Gas_Modelling\true_obs\para_true.txt','r') as f:
#        content=f.readlines()
#
#    content=np.array(content)
#    content=np.float64(content)
#    content=Series(content)
#    content1=content.copy()
#    for i in range(1,41):
#        content1[i*2+1]=content[i]
#    content1[1]=content[41]
#
#    for i in range(42,82):
#        content1[(i-41)*2]=content[i]
#
#    para_distribution_map(content1,1681,root_directory)
#    np.savetxt(r'E:\EnRML_Gas_Modelling\true_obs\para_true.txt',content1)