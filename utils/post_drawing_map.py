# -*- coding: utf-8 -*-

##用来绘制t=4,8,12,16,20时刻的参数场分布图，用来和真实参数场对比
import os
import pandas as pd
Nod_num=1681
import numpy as np
from para_distribution_map import para_distribution_map

current_directory=os.getcwd()
root_directory=os.path.dirname(current_directory)
root_directory_true_obs=os.path.join(root_directory,'true_obs')
#将真实参数场写入parameter_distribution.tec文件
filename='para_true.txt'
args_para_true=os.path.join(root_directory,filename)
para_true=pd.read_csv(args_para_true,sep=' ',names=[1])
para_true=para_true.values
para_true=np.squeeze(para_true)
para_distribution_map(para_true,Nod_num,root_directory_true_obs)

#将更新后的参数写入parameter_distribution.tec文件
#filename='t_20_ave.txt'
#filepath=os.path.join(root_directory,filename)
#content=pd.read_csv(filepath,sep=' ',names=[1])
#content=content.values
#para=content[:,0]
#para_distribution_map(para,Nod_num,root_directory_true_obs)
