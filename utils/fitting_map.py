# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from write_values import write_ki
import numpy as np
import subprocess,multiprocessing
import time
import os
##提取一个点（0.75,0.4）处的20个时间步的观测值
current_directory=os.getcwd()
root_directory=os.path.dirname(current_directory)
path_for_add='true_obs/obs_20.txt'
point_args=os.path.join(root_directory,path_for_add)
obs_values=pd.read_csv(point_args,sep=' ',names=[i for i in range(4)])
line=[5+16*i for i in range(20)]
values_for_map=obs_values.ix[line,[3]]
values_for_map=values_for_map.values


# ##一个文件夹提取出一根线(0.75,0.4)的20个时间步的变化图）
def get_obs_updated(i):
    line_num_updated=[3918+3284*ii for ii in range(20)]
    filename='gas_{0}\gas_domain_AIR_FLOW_quad.tec'.format(i)
    args_result=os.path.join(root_directory,filename)
    with open(args_result,'r') as f:
        content=f.readlines()
    for i in xrange(len(content)):
        content[i]=content[i].split()
    content=np.array(content)
    ff=lambda x:np.float64(x)
    obs_updated=content[line_num_updated]
    obs_updated=map(ff,obs_updated)
    obs_up=np.zeros((20,4))
    for m in range(20):
        for n in range(4):
            obs_up[m,n]=obs_updated[m][n]
    result=obs_up[:,3]
    return result


N=100
Nod_num=1681
args_para_updated=os.path.join(root_directory,'t_20.txt')
args_para_initial=os.path.join(root_directory,'para_initial.txt')
para=pd.read_csv(args_para_updated,sep=' ',names=[i for i in range(100)])
para_init=pd.read_csv(args_para_initial,sep=' ',names=[i for i in range(100)])
##将初始参数带进去，计算得到结果

def runexe(i,root_directory):
    path_1=root_directory
    path_2='gas_{0}/ogs'.format(i)
    path_3='gas_{0}/water_gas'.format(i)
    args_exe=os.path.join(path_1,path_2)
    args_para=os.path.join(path_1,path_3)
    pp=subprocess.Popen((args_exe,args_para),stdout=subprocess.PIPE)
    pp.communicate()
    pp.wait()
for i in xrange(N):
    para_initial=para_init.ix[:,[i]]
    para_initial=para_initial.values
    para_initial=para_initial.reshape(1681,)
    para_initial=np.exp(para_initial)
    write_ki(1681,para_initial,i)


jobs=[]
for i in xrange(N):
    p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()
    


values_initial=np.zeros((20,N))
for i in xrange(N):
    obs_initial=get_obs_updated(i)
    for m in xrange(len(obs_initial)):
        values_initial[m,i]=obs_initial[m]


##将更新后的参数带进去，计算得到结果
for i in xrange(N):
    para_updated=para.ix[:,[i]]
    para_updated=para_updated.values
    para_updated=para_updated.reshape(1681,)
    para_updated=np.exp(para_updated)
    write_ki(1681,para_updated,i)

jobs=[]
for i in xrange(N):
    p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
    jobs.append(p)
    p.start()
for j in jobs:
    j.join()

values_updated=np.zeros((20,N))
for i in xrange(N):
    obs_update=get_obs_updated(i)
    for m in xrange(len(obs_update)):
        values_updated[m,i]=obs_update[m]
#
# x=[i+1 for i in range(20)]
# for k in range(N):
#     plt.plot(x,values_initial[:,k],'b')
#     plt.plot(x,values_updated[:,k],'y')
#     plt.scatter(x,values_for_map,color='r')
# plt.show()

# args_1=r'E:\EnRML_Gas_Modelling\values_initial.txt'
# args_2=r'E:\EnRML_Gas_Modelling\values_updated.txt'
# args_3=r'E:\EnRML_Gas_Modelling\values_for_map.txt'
# values_initial=pd.read_csv(args_1,sep=' ',names=[i for i in range(100)])
# values_updated=pd.read_csv(args_2,sep=' ',names=[i for i in range(100)])
# values_for_map=pd.read_csv(args_3,names=[1])



# values_initial=values_initial.values
# values_updated=values_updated.values
# values_for_map=values_for_map.values
x=[i+1 for i in range(20)]
N=100
for k in range(N):
    plt.plot(x,values_initial[:,k],'b',alpha=0.4)     ##不知道怎样给N条线一个统一的图例
    plt.plot(x,values_updated[:,k],'y',alpha=0.8)
    if k==N-1:
        plt.plot(x,values_initial[:,k],'b',alpha=0.4,label='initial')     ##不知道怎样给N条线一个统一的图例
        plt.plot(x,values_updated[:,k],'y',alpha=0.8,label='updated')
plt.scatter(x,values_for_map,color='r',marker='x',s=60,label='observation')
plt.legend(loc='best')
plt.xlabel('time step')
plt.ylabel('Pressure(Pa)')
plt.xlim(0,20)
plt.show()
#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)




