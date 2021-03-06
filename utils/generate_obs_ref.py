#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'zq'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.generateL import generateL
from utils.para_key_mod import para_keys_modify
from utils.para_distribution_map import para_distribution_map
from utils.write_values import write_ki


def generate_para_true(sigma,deltax,deltay,dx,dy,m,n,ki_mean):
    L1=generateL(sigma,deltax,deltay,dx,dy,m,n)
    ran=np.random.normal(0,1,size=Nod_num)
    ran=ran.reshape(Nod_num,1)
    lnpara=np.dot(L1,ran)
    mean=np.log(ki_mean)
    lnpara=lnpara+mean
    para=np.array(lnpara)
    para=para.reshape(Nod_num,)
    return para

def write_KI(Nod_num,para,root_directory,filename_KI):
    para=np.exp(para)
    value_list={}   #产生要添加的ki序列
    for i in xrange(Nod_num):
        value_list[i]=para[i]
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])  #使用join进行字符串的拼接效率更高
        value_list_to_str.append(var)
    file_name=filename_KI
    args=os.path.join(root_directory,'true_obs',file_name)
    with open(args,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')

def write_BC(x,filename_bc,current_directory):
    x1,x2,x3=x
    args_path=os.path.join(current_directory,'true_obs',filename_bc)
    with open(args_path,'r') as  f:
        content=f.readlines()
        line_modify_1=content[9]
        line_modify_2=content[18]
        line_modify_3=content[27]
        modify_1=line_modify_1.split()
        modify_2=line_modify_2.split()
        modify_3=line_modify_3.split()
        modify_1[1]=str(x1)
        line_modify_1=' '.join(modify_1[i] for i in range(len(modify_1)))+'\n'
        modify_2[1]=str(x2)
        line_modify_2=' '.join(modify_2[i] for i in range(len(modify_2)))+'\n'
        modify_3[1]=str(x3)
        line_modify_3=' '.join(modify_3[i] for i in range(len(modify_3)))+'\n'
        content[9]=line_modify_1
        content[18]=line_modify_2
        content[27]=line_modify_3
    with open(args_path,'w') as f:
        for line in content:
            f.write(line)


def write_p1(Nod_num,root_directory,filename_Pressure):
    value_list={}   #产生要添加的p1序列
    for i in range(Nod_num):
        value_list[i]=101325
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])
        value_list_to_str.append(var)
    file_name=filename_Pressure
    args=os.path.join(root_directory,file_name)
    with open(args,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')


##提取所有时间步的观测点处的预测值，加噪声之后作为观测值
def read_trueobs(t,obs_Num,Nod_num,root_directory,filename_result):
    domain=filename_result
    args_domain=os.path.join(root_directory,domain)
    with open(args_domain,'r') as f:
        content=f.readlines()
    x=np.empty((1,5))   #准备提取tec文件中每个时间步输出的前5列数据（x,y,z,p1,p2）
    for i in range(t+1):
        content1=content[3284*i:3284*i+Nod_num+3]  #跳过第0个时间步，第一个时间步从第3284行开始，854会随网格划分格点及单元个数而改变
        content1=content1[3:]              #踢掉tec前面3行头数据
        for i in range(len(content1)):
            content1[i]=content1[i].split()
        content1=pd.DataFrame(content1)
        content_1=content1.ix[obs_Num,[0,1,2,3,4]]
        value=content_1.values
        x=np.vstack((x,value))

    x=x[1+len(obs_Num):]
    x=np.array(x)
    x=np.float64(x)
    file_name='obs_%d.txt' % t
    args=os.path.join(root_directory,file_name)
    np.savetxt(args,x)


##对观测值进行误差扰动，产生观测值样本空间，由于返回字典，故不进行txt存储，在主程序中直接调用该函数即可
def generate_obs(time_step,obs_num,varR,N,root_directory):
    obs_name='obs_{0}.txt'.format(time_step)
    args_obs_20=os.path.join(root_directory,obs_name)
    with open(args_obs_20,'r') as f:
        content=f.readlines()

    for i in xrange(len(content)):
        content[i]=content[i].split()
    obs=pd.DataFrame(content)
    obs=obs[[3,4]]
    obs=np.array(obs.values)
    obs=np.float64(obs)
    obs_p1=np.zeros([time_step,obs_num])
    obs_p2=np.zeros([time_step,obs_num])
    for i in range(len(obs)):
        li_num=(i)/obs_num
        col_num=(i)%obs_num
        line=obs[i]
        obs_p1[li_num,col_num]=line[0]
        obs_p2[li_num,col_num]=line[1]

    # # #在原始输出的基础上，进行样本之间的误差扰动，最终输出3维字典，第一维键为时间步，第二维键为观测点序号，第三维为键值，为20个样本的观测值
    obs_Pressure1={}
    obs_Pressure2={}
    for x in xrange(1,time_step+1):
        tobs1={}
        tobs2={}
        for y in range(1,obs_num+1):
            obs_value1={}
            obs_value2={}
            for z in xrange(N):
                obs_value1[z]=obs_p1[x-1,y-1]+np.sqrt(varR)*np.random.standard_normal()
                obs_value2[z]=obs_p2[x-1,y-1]+np.sqrt(varR)*np.random.standard_normal()
            tobs1[y]=obs_value1
            tobs2[y]=obs_value2
        obs_Pressure1[x]=tobs1
        obs_Pressure2[x]=tobs2
    return obs_Pressure1,obs_Pressure2
    
    
##查看一个点的压力值随时间的变化
def one_point_p(root_directory,filename_result):
    num=[315+3284*i for i in range(40)]
    domain=filename_result
    args_domain=os.path.join(root_directory,domain)
    with open(args_domain,'r') as f:
        content=f.readlines()
    for i in xrange(len(content)):
        content[i]=content[i].split()
    content=pd.DataFrame(content)
    point=content.ix[num,[0,1,2,3]]
    point=point.values
    return point


def generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR):      ##产生后面一个时间步的p2观测值样本 （4.13）
    domain=filename_result
    args_domain=os.path.join(root_directory,domain)
    with open(args_domain,'r') as f:
        content=f.readlines()
    content1=content[3284:3284+Nod_num+3]  #跳过第0个时间步，第一个时间步从第3284行开始，854会随网格划分格点及单元个数而改变
    content1=content1[3:]              #踢掉tec前面3行头数据
    for i in xrange(len(content1)):
        content1[i]=content1[i].split()
    content1=pd.DataFrame(content1)
    content_1=content1.ix[obs_Num,[0,1,2,3,4]]
    value=content_1.values
    value=np.array(value)
    value=np.float64(value)
    pressure2=value[:,-1]
    pressure2_sample=np.zeros((len(obs_Num),N))
    for i in xrange(len(obs_Num)):
        for j in xrange(N):
            pressure2_sample[i,j]=pressure2[i]+np.sqrt(varR)*np.random.standard_normal()
    return pressure2_sample
    







if __name__=='__main__':
    current_directory=os.getcwd()
    parent_directory=os.path.dirname(current_directory)
    root_directory=os.path.join(parent_directory,'true_obs')
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
#    
#    para_true=generate_para_true(sigma,deltax,deltay,dx,dy,m,n,ki_mean,root_directory)
#    para_true_modified=para_keys_modify(para_true)
#    np.savetxt(os.path.join(root_directory,'para_true.txt'),para_true_modified)
#    para_distribution_map(para_true_modified,Nod_num,root_directory)
#    write_KI(Nod_num,para_true,root_directory,filename_KI)
#    read_trueobs(20,obs_Num,Nod_num,root_directory,filename_result)
    y=generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR)
