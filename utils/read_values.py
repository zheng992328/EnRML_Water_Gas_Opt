#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'zq'

import pandas as pd
import numpy as np
import os

##定义函数，读取观测点处对应的预测值
def read_obs(Nod_num,obs_Num,i,filename_result,current_directory):
    gas_file='gas_{0}'.format(i)
    args_obs=os.path.join(current_directory,gas_file,filename_result)
    with open(args_obs,'r') as f:
        Nod_inf=f.readlines()
    last=Nod_inf[-(1600+Nod_num):-1600]  #1600是elements的个数
    for i in xrange(Nod_num):
        last[i]=last[i].split()
    last=pd.DataFrame(last)
    obs_prediction=last.ix[obs_Num,[4]]
    obs_prediction=obs_prediction.values
    obs_prediction=np.float64(obs_prediction)
    obs_p1=obs_prediction[:,0]
    return obs_p1


###读取所有的p1值
#def read_p(Nod_num,i,filename_result):
#    gas_file='gas_{0}'.format(i)
#    p_result=os.path.join(current_directory,gas_file,filename_result)
#    with open(p_result,'r') as f:
#        Nod_inf=f.readlines()
#    last=Nod_inf[-(1600+Nod_num):-1600]
#    head=[None]*Nod_num
#    for i in xrange(len(last)):
#        line=last[i].split()
#        head[i]=line[3:4]
#
#    head=np.array(head)
#    head=np.float64(head)
#    return head


if __name__=='__main__':
    current_directory=os.getcwd()
    root_directory=os.path.dirname(current_directory)
#    root_directory=os.path.join(parent_directory,'true_obs')
    time_step=20
    Nod_num=1681
    N=100
    varR=80
    obs_Num=[213,221,229,237,623,631,639,647,1033,1041,1049,1057,1443,1451,1459,1467]
    obs_num=len(obs_Num)
    sigma=0.6
    ki_mean=2.9e-15
    deltax=1
    deltay=0.5
    dx=0.05
    dy=0.025
    x=2
    y=1
    m=int(y/dy+1)
    n=int(x/dx+1)
    filename_KI='water_gas_KI.direct'
#    finename_Pressure='water_gas_PRESSURE1.direct'
    filename_result='water_gas_domain_quad.tec'
    yy=read_obs(Nod_num,obs_Num,12,filename_result,root_directory)
    
    
    
    
    
    
    
    
    
    
    
    
    
    