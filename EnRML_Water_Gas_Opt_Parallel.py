#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
#from utils.generate_obs_ref import generate_obs
from utils.values_init import para_init
from utils.para_key_mod import para_keys_modify
from utils.write_values import write_ki
from utils.read_values import read_obs
from utils.time_modify import time_modify
from utils.calculate_q_well import profit_calcualtion_1,profit_calcualtion_2,profit_calcualtion_3
#from utils.write_values import write_bc
from utils.generate_obs import write_KI,generate_real_time_obs,generate_para_true,rfd_MODIIFY,time_MODIFY
from utils.modify_rfd import modify_rfd_for_g,modify_rfd_for_inverse
import os
import subprocess,multiprocessing



time_step=20
Nod_num=1681
N=64
varR=80
obs_Num=[213,221,229,237,623,631,639,647,1033,1041,1049,1057,1443,1451,1459,1467]
obs_num=len(obs_Num)
sigma=0.6
ki_mean=2.9e-15
deltax=1
deltay=0.52
dx=0.05
dy=0.025
x=2
y=1
m=int(y/dy+1)
n=int(x/dx+1)
root_directory=os.getcwd()
root_directory_true_obs=os.path.join(root_directory,'true_obs')
para_for_ogs='water_gas'
filename_result='water_gas_domain_quad.tec'
filename_KI='water_gas_Ki.direct'
filename_time='water_gas.tim'
filename_bc='water_gas.bc'
filename_rfd='water_gas.rfd'
filename_v1='water_gas_ply_PLY_2_t1.tec'
filename_v2='water_gas_ply_PLY_3_t2.tec'
filename_v3='water_gas_ply_PLY_4_t3.tec'
#filename_Pressure='gas_PRESSURE1.direct'
beta=0.3
eps1=1e-4
eps2=1e-3
pressure_low=90000
pressure_high=92000
pressure_mid=(pressure_low+pressure_high)/2
well_num=3
area_well=2   #井的集气面积认为是2，井深0.6m，底部直径0.8m
delta_t=50  #模拟时间步长是500s
x_ini=np.random.uniform(pressure_low,pressure_high,well_num)      #初始的x向量从均匀分布取出
#x_ini=np.array([pressure_mid]*3)
x_opt=np.zeros((len(x_ini),time_step))   #记录初始控制变量和每一个时间步优化之后的控制变量
#x_true=np.random.uniform(pressure_low,pressure_high,well_num) 
x_true=np.random.uniform(pressure_low,pressure_high,well_num)      #初始的x向量从均匀分布取出
h=open('log.txt','w')

#gas_cost=()    #0肯定是不合实际的
gas_price=16   ##0.16d/m3
gas_cost_ref=2.0e-10 ##采气负压的参考价格
alpha=0.3

##产生真实参数,产生第一个时间步的观测值样本，从始至终真实参数不变。变的是控制变量和观测值
para_true=generate_para_true(sigma,deltax,deltay,dx,dy,m,n,ki_mean,Nod_num)
para_true=para_keys_modify(para_true)
np.savetxt('para_true.txt',para_true)
write_KI(Nod_num,para_true,root_directory,filename_KI)
x_opt[:,0]=x_true ##将初始的控制变量记入x_opt第一列，第一个时间步的反演也是基于该值
time_MODIFY(1,delta_t,filename_time,root_directory)
rfd_MODIIFY(1,x_opt,root_directory,filename_rfd)
args_true_obs_exe=os.path.join(root_directory,'true_obs','ogs')
args_true_obs=os.path.join(root_directory,'true_obs')
args_true_obs_para=os.path.join(root_directory,'true_obs','water_gas')
pt=subprocess.Popen((args_true_obs_exe,args_true_obs_para),stdout=subprocess.PIPE)
pt.communicate()
pt.wait()


#第一个时间步的观测值
y_obs_p1=generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR,root_directory)  #得到第一个时间步的观测值
y_obs_p1_ave=np.mean(y_obs_p1,axis=1)

#产生初始参数
para_initial=para_init(sigma,deltax,deltay,dx,dy,m,n,Nod_num,N,ki_mean)
np.savetxt('para_initial.txt',para_initial)
para_initial_ave=np.mean(para_initial,axis=1)
for i in xrange(N):
    para_initial[:,i]=para_keys_modify(para_initial[:,i])   ##得到初始参数之后就将其转换成可以代入模型的排列形式

##获取误差矩阵
y_obs_p1_error=np.zeros_like(y_obs_p1)
para_error=np.zeros_like(para_initial)
for i in xrange(N):
    y_obs_p1_error[:,i]=y_obs_p1[:,i]-y_obs_p1_ave[:]
    para_error[:,i]=para_initial[:,i]-para_initial_ave[:]

Cd=np.dot(y_obs_p1_error,y_obs_p1_error.T)/(N-1)
Cm=np.dot(para_error,para_error.T)/(N-1)
mpr=para_initial
m1=para_initial
Imax=0

y_obs_prediction=np.zeros_like(y_obs_p1)
y_obs_prediction_error=np.zeros_like(y_obs_prediction)
parY=np.zeros((Nod_num,time_step))    #记录每一个时间步反演之后的参数
#production_initial=np.zeros(time_step)   #记录每个时间步在优化之前的初始产量
#production_updated=np.zeros(time_step)   #记录每个时间步在优化之后的更新产量
def runexe(i,root_directory):
    path_1=root_directory
    path_2='gas_{0}/ogs'.format(i)
    path_3='gas_{0}/water_gas'.format(i)
    args_exe=os.path.join(path_1,path_2)
    para=os.path.join(path_1,path_3)
    pp=subprocess.Popen((args_exe,para),stdout=subprocess.PIPE)
    pp.communicate()
    pp.wait()
while True:
    print time.ctime()
    print 'the {0} inverse iteration'.format(Imax)
    h.write('1_time_inverse_{0}_iteration'.format(Imax)+'\n')
    mat_to_arr=lambda x:np.array(x)
    m1,y_obs_prediction,y_obs_p1=map(mat_to_arr,[m1,y_obs_prediction,y_obs_p1])
    m1_average=np.mean(m1,axis=1)
    m1_error=np.zeros_like(m1)
    for i in xrange(N):
        m1_error[:,i]=m1[:,i]-m1_average[:]
      
    ks=np.exp(m1)
    for i in xrange(N):       
        ki_modify=ks[:,i]
#        ki_modify=para_keys_modify(ki_modify)
        write_ki(Nod_num,ki_modify,i,filename_KI,root_directory) 
        #改变.rfd的值，从x_opt读入，x_opt当前有几列，就计算到第几个时间步。但是第一个时间步的计算，原文件中肯定都设置好了，所以不用特地写入
        modify_rfd_for_inverse(1,i,x_opt,root_directory,filename_rfd)
        
#    ppservers=()
#    job_server=pp.Server(2,ppservers=ppservers)
#    Ne=tuple(range(N))
#    jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#    for i,job in jobs:
#        job()
        
#    for i in xrange(N):
#        runexe(i,root_directory)
#        print i
#        h.write('1_inverse_{0}_1'.format(i)+'\n')
    jobs=[]
    for i in xrange(N):
        p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
       
    
    time.sleep(3)
    for k in xrange(N):
        Obs_p1=read_obs(Nod_num,obs_Num,k,filename_result,root_directory)
        for ii in xrange(obs_num):
            y_obs_prediction[ii][k]=Obs_p1[ii]  #组成观测点处的预测值矩阵
          

    y_obs_prediction_average=np.mean(y_obs_prediction,axis=1)
    for i in xrange(N):
        y_obs_prediction_error[:,i]=y_obs_prediction[:,i]-y_obs_prediction_average[:]
    
    wn_M=np.linalg.pinv(m1_error)
    G=np.dot(y_obs_prediction_error,wn_M)
    arr_to_mar=lambda x: np.matrix(x)
    mpr,m1,Cm,G,Cd,y_obs_prediction,y_obs_p1=map(arr_to_mar,[mpr,m1,Cm,G,Cd,y_obs_prediction,y_obs_p1])
    m2=beta*mpr+(1-beta)*m1-beta*Cm*(G.T)*(np.linalg.inv(Cd+G*Cm*G.T))*(y_obs_prediction-y_obs_p1-G*(m1-mpr))
    S1=np.trace(((y_obs_prediction-y_obs_p1).T)*np.linalg.inv(Cd)*(y_obs_prediction-y_obs_p1))
    m2=np.array(m2)
    y_obs_prediction=np.array(y_obs_prediction)
       
    ks2=np.exp(m2)
    for i in xrange(N):
        ki_modify=ks2[:,i]
#        ki_modify=para_keys_modify(ki_modify)
        write_ki(Nod_num,ki_modify,i,filename_KI,root_directory)  #para为初始参数，样本之间存在随机扰动
        
#    ppservers=()
#    job_server=pp.Server(2,ppservers=ppservers)
#    Ne=tuple(range(N))
#    jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#    for i,job in jobs:
#        job()
        
#    for i in xrange(N):
#        runexe(i,root_directory)
#        print i
#        h.write('1_inverse_{0}_2'.format(i)+'\n')
    jobs=[]
    for i in xrange(N):
        p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
 
    time.sleep(3)
    for k in xrange(N):
        Obs_p1=read_obs(Nod_num,obs_Num,k,filename_result,root_directory)
        for ii in xrange(obs_num):
            y_obs_prediction[ii][k]=Obs_p1[ii]  #组成观测点处的预测值矩阵
    y_obs_prediction=np.matrix(y_obs_prediction)
 
    S2=np.trace(((y_obs_prediction-y_obs_p1).T)*np.linalg.inv(Cd)*(y_obs_prediction-y_obs_p1))
    print 'max(m2-m1):',np.max(np.abs(m2-m1))
    print 'S2-S1:',S2-S1
    h.write('max(m2-m1):'+str(np.max(np.abs(m2-m1)))+'\n')
    h.write('S2-S1:'+str(S2-S1)+'\n')
    
    if (np.max(np.abs(m2-m1))<eps1 or S2-S1<eps2*S1 or Imax>6):
        break
    print 'next iteration'
    if S2<S1:
        m1=m2
        beta=2*beta
        if beta>1:
            beta=1
    else:
        beta=0.5*beta
    Imax+=1
np.savetxt('t_1.txt',m2)  #更新后的参数m2是要写进文件的，为优化服务的
parY[:,0]=np.average(m2,axis=1)
np.savetxt('t_1_ave.txt',parY[:,0])
#x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)   #计算优化前的累积产量
#x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v2,root_directory)
#x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v3,root_directory)
#production_initial[0]=y1+y2+y3
#np.savetxt('production_initial.txt',production_initial)
##计算g部分
Itera=0
while True:
    print 'the {0} opt iteration'.format(Itera)
    h.write('1_time_opt_{0}_iteration'.format(Itera)+'\n')
    x_mat=np.zeros((len(x_ini),N))
    for i in xrange(len(x_ini)):
        for j in xrange(N):
            x_mat[i,j]=x_ini[i]+np.random.normal(0,0.001*x_ini[i])    #扰动偏差是当前压力的0.1%
    
    x_mat_ave=np.average(x_mat,axis=1)
    x_mat_error=np.zeros_like(x_mat)
    for i in xrange(N):
        x_mat_error[:,i]=x_mat[:,i]-x_mat_ave[:]
    
    for i in xrange(N):     #从当前时刻计算到最后，算这个时间段的g作为优化目标
        time_modify(1,i,delta_t,filename_time,root_directory)
        ##改时间的同时得把边界条件也改了，从当前时刻到最后都用一样的边界。
        modify_rfd_for_g(i,x_mat[:,i],root_directory,filename_rfd)
        
    jobs=[]
    for i in xrange(N):
        p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
        
#    ppservers=()
#    job_server=pp.Server(2,ppservers=ppservers)
#    Ne=tuple(range(N))
#    jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#    for i,job in jobs:
#        job()
        
    time.sleep(3)
    Cx=np.dot(x_mat_error,x_mat_error.T)/(N-1)   #Cx是这么计算的吗？Cx是对角线上服从高斯分布，其余地方都为0，大小是多少呢？
    g_x_mat=np.zeros(N)
    for i in xrange(N):   #第一个时间步向后计算g，总共19个时间步，那么从1开始，19结束，时间步为19
        gas_cost1=(101325-x_mat[0,i])*gas_cost_ref
        gas_cost2=(101325-x_mat[1,i])*gas_cost_ref
        gas_cost3=(101325-x_mat[2,i])*gas_cost_ref
#        gas_cost1=0
#        gas_cost2=0
#        gas_cost3=0
        x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost1,filename_v1,root_directory)
        x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost2,filename_v2,root_directory)
        x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost3,filename_v3,root_directory)
        g_x_mat[i]=x1+x2+x3
        
    g_x_mat_ave=np.average(g_x_mat)
    g_x_mat_error=np.zeros_like(g_x_mat)
    for i in xrange(N):
        g_x_mat_error[i]=g_x_mat[i]-g_x_mat_ave
    
    C_x_gx=np.zeros_like((x_mat_error))
    for i in xrange(N):
        C_x_gx[:,i]=x_mat_error[:,i]*g_x_mat_error[i]
    
    C_x_gx=np.sum(C_x_gx,axis=1)/(N-1)
    ##是对每个样本进行迭代计算，然后每个样本都有一个更新值，还是样本的作用只是用来求解G，然后用原来的均值去迭代更新，更新后还是得到均值
    ##姑且采用每个样本都单独进行更新,但是这样每个样本的增加值一样了。
    inner_itera=0
#    alpha_increase_number=0
    while True:  ##(是吗？)这个循环可以用来看看alpha是否合适，不合适（新的g小于老的g）的话就增加alpha
        h.write('1_time_opt_{0}_inneriteration_{1}'.format(Itera,inner_itera)+'\n')
        delta_x=np.dot(Cx,C_x_gx)/alpha
        np.savetxt('delta_x_{0}.txt'.format(Itera),delta_x)
        x_1=np.zeros_like(x_mat_ave)
        x_1=delta_x+x_mat_ave      #这里x更新均值，得到更新后的均值，在此基础上再一次加噪声变成样本
        x_1_mat=np.zeros_like(x_mat)
        for i in xrange(len(x_1)):
            for j in xrange(N):
                x_1_mat[i,j]=x_1[i]+np.random.normal(0,0.001*x_1[i])  
            
        if np.min(x_1_mat)<0:
            break
        x_1_mat[x_1_mat<pressure_low]=pressure_low
        x_1_mat[x_1_mat>pressure_high]=pressure_high
        ##接下来要将x_1_mat代入模型的边界条件，首先改写写入.bc文件的函数
        for i in xrange(N):
            modify_rfd_for_g(i,x_1_mat[:,i],root_directory,filename_rfd)
#        ppservers=()
#        job_server=pp.Server(2,ppservers=ppservers)
#        Ne=tuple(range(N))
#        jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#        for i,job in jobs:
#            job()
#        for i in xrange(N):
#            runexe(i,root_directory)
#            time.sleep(1)
#            print 'inner_itera_{0}_{1}'.format(inner_itera,i)
#            h.write('{0}_opt_inneritera_{1}_{2}'.format(1,inner_itera,i)+'\n')
        jobs=[]
        for i in xrange(N):
            p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        time.sleep(3)    
        for i in xrange(N):
            gas_cost1=(101325-x_1_mat[0,i])*gas_cost_ref
            gas_cost2=(101325-x_1_mat[1,i])*gas_cost_ref
            gas_cost3=(101325-x_1_mat[2,i])*gas_cost_ref
#            gas_cost1=0
#            gas_cost2=0
#            gas_cost3=0
            x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost1,filename_v1,root_directory)
            x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost2,filename_v2,root_directory)
            x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost3,filename_v3,root_directory)
            g_x_mat[i]=x1+x2+x3
#        production_updated[0]=y1+y2+y3
#        np.savetxt('production_updated.txt',production_updated)
        g_x_mat_ave_1=np.average(g_x_mat)
        x_1_mat_ave=np.average(x_1_mat,axis=1)
        print g_x_mat_ave_1,g_x_mat_ave
        h.write('g_x_mat_ave_1:'+str(g_x_mat_ave_1)+'g_x_mat_ave:'+str(g_x_mat_ave)+'\n')
        if g_x_mat_ave_1>g_x_mat_ave:
            x_ini=x_1_mat_ave
            break
        elif inner_itera>2:
            break
        else:
            alpha=1.5*alpha
#            alpha_increase_number+=1
        inner_itera+=1
    
    delta_g=(g_x_mat_ave_1-g_x_mat_ave)/g_x_mat_ave
    x_opt[:,1]=x_1_mat_ave
    np.savetxt('x_1_{0}.txt'.format(Itera),x_1_mat)
    np.savetxt('x_1_{0}_ave.txt'.format(Itera),x_1_mat_ave)
    
    #计算到当前时间步的累积产量
#    for i in xrange(N):
#        time_modify(1,i,delta_t,filename_time,root_directory)
#        modify_rfd_for_g(i,x_opt[:,0],root_directory,filename_rfd)   #此处任然可用该函数，虽然他定义到时间段末尾（1000），但是在之前用time_modufy修改了计算时间，所以还是按照计算时间来
#    for i in xrange(N):
#        runexe(i,root_directory)
#        time.sleep(1)
#        h.write('1_opt_cal_cum_g_{0}'.format(i)+'\n')
#        
#    while True:
#        back_process=os.popen('tasklist |findstr /i ogs.exe').readlines()
#        if back_process==[]:
#            break
#    for i in xrange(N):
#        x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)
#        x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v2,root_directory)
#        x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost,filename_v3,root_directory)
#    production_updated[0]=y1+y2+y3
    
    x_change_rate=np.zeros_like(x_1_mat_ave)
    for i in xrange(len(x_1_mat_ave)):
        x_change_rate[i]=(x_1_mat_ave[i]-x_mat_ave[i])/x_mat_ave[i]   
    x_change_rate_ave=np.average(x_change_rate)    # 这里取的是控制变量向量中每个元素对应的变化率的平均
    print delta_g,x_change_rate_ave
    h.write('delta_g:'+str(delta_g)+'x_change_rate_ave:'+str(x_change_rate_ave)+'\n')
    if (0<delta_g<0.0005 and x_change_rate_ave<0.01):
        break
#    x_ini=x_1_mat_ave   #将迭代之后的x_1_mat样本上取平均之后再赋给x_ini,再进行循环。
    Itera=Itera+1
    if Itera>3:
        break
    
    
print 'the first opt is over'
#
####这里得到了当前时间步的更新参数，要用它去计算目标函数
for t in range(2,21):
    print time.ctime()
    print 'the {0} time step'.format(t)
    h.write('the {0} time step'.format(t)+'\n')
    mpr=m2 #上一步带出来的反演参数是m2
    x_ini=x_opt[:,t-1]  #第二个时间步之后，控制变量的初始值设为上一步的优化值，而每一次退出优化带出来的是x_1_mat_ave,保存于x_opt[:t-1]
    
    ##将t-1时刻得到的优化控制变量作用于真实系统（true_obs），得到t时刻的观测值样本(和enrml结合，让他从初始时刻计算到t时刻)
    time_MODIFY(t,delta_t,filename_time,root_directory)
    rfd_MODIIFY(t,x_opt,root_directory,filename_rfd)   ##上一个时间步优化后的控制变量作用于系统，作用一个时间步
    pt=subprocess.Popen((args_true_obs_exe,args_true_obs_para),stdout=subprocess.PIPE)
    pt.communicate()
    pt.wait()
    
    y_obs_p1=generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR,root_directory)
    y_obs_p1_ave=np.mean(y_obs_p1,axis=1)
    y_obs_prediction=np.zeros_like(y_obs_p1)
    for i in xrange(N):
        y_obs_p1_error[:,i]=y_obs_p1[:,i]-y_obs_p1_ave[:]
        para_error[:,i]=mpr[:,i]-parY[:,t-2]
    Cd=np.dot(y_obs_p1_error,y_obs_p1_error.T)/(N-1)
    Cm=np.dot(para_error,para_error.T)/(N-1)
    m1=mpr
    
    Imax=0
    
    while True:
        print '{0}_time_inverse_{1}_iteration'.format(t,Imax)
        h.write('{0}_time_inverse_{1}_iteration'.format(t,Imax)+'\n')
        mat_to_arr=lambda x:np.array(x)
        m1,y_obs_prediction,y_obs_p1=map(mat_to_arr,[m1,y_obs_prediction,y_obs_p1])
        m1_average=np.mean(m1,axis=1)
        for i in xrange(N):
            m1_error[:,i]=m1[:,i]-m1_average[:]
            
        ks=np.exp(m1)
        
        for i in xrange(N):
            time_modify(t,i,delta_t,filename_time,root_directory)
            ki_modify=ks[:,i]
#            ki_modify=para_keys_modify(ki_modify)
            write_ki(Nod_num,ki_modify,i,filename_KI,root_directory)  #para为初始参数，样本之间存在随机扰动
            modify_rfd_for_inverse(t,i,x_opt,root_directory,filename_rfd)   #把x_opt数组下前t列的写到rfd文件中
            
#        ppservers=()
#        job_server=pp.Server(2,ppservers=ppservers)
#        Ne=tuple(range(N))
#        jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#        for i,job in jobs:
#            job()
#        for i in xrange(N):
#            runexe(i,root_directory)
#            print '{0}_{1}_1'.format(t,i)
#            h.write('{0}_inverse_{1}_1'.format(t,i)+'\n')
#            time.sleep(1)
        jobs=[]
        for i in xrange(N):
            p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        
         
        time.sleep(3)
        y_obs_prediction=np.array(y_obs_prediction)
        for k in xrange(N):
            Obs_p1=read_obs(Nod_num,obs_Num,k,filename_result,root_directory)
            for ii in xrange(obs_num):
                y_obs_prediction[ii][k]=Obs_p1[ii]  
       
         
        
        y_obs_prediction_average=np.mean(y_obs_prediction,axis=1)
        for i in xrange(N):
            y_obs_prediction_error[:,i]=y_obs_prediction[:,i]-y_obs_prediction_average[:]
        
        wn_M=np.linalg.pinv(m1_error)
        G=np.dot(y_obs_prediction_error,wn_M)
        arr_to_mar=lambda x: np.matrix(x)
        mpr,m1,Cm,G,Cd,y_obs_prediction,y_obs_p1=map(arr_to_mar,[mpr,m1,Cm,G,Cd,y_obs_prediction,y_obs_p1])
        m2=beta*mpr+(1-beta)*m1-beta*Cm*(G.T)*(np.linalg.inv(Cd+G*Cm*G.T))*(y_obs_prediction-y_obs_p1-G*(m1-mpr))
        S1=np.trace(((y_obs_prediction-y_obs_p1).T)*np.linalg.inv(Cd)*(y_obs_prediction-y_obs_p1))
        m2=np.array(m2)
        y_obs_prediction=np.array(y_obs_prediction)
        
        
        ks2=np.exp(m2)
        for i in xrange(N):
            ki_modify=ks2[:,i]
#            ki_modify=para_keys_modify(ki_modify)
            write_ki(Nod_num,ki_modify,i,filename_KI,root_directory)  #para为初始参数，样本之间存在随机扰动

#        ppservers=()
#        job_server=pp.Server(2,ppservers=ppservers)
#        Ne=tuple(range(N))
#        jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#        for i,job in jobs:
#            job()
#        for i in xrange(N):
#            runexe(i,root_directory)
#            time.sleep(1)
#            print '{0}_{1}_2'.format(t,i)
#            h.write('{0}_inverse_{1}_2'.format(t,i)+'\n')
        jobs=[]
        for i in xrange(N):
            p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        
    
        time.sleep(3)
        for k in xrange(N):
            Obs_p1=read_obs(Nod_num,obs_Num,k,filename_result,root_directory)
            for ii in xrange(obs_num):
                y_obs_prediction[ii][k]=Obs_p1[ii]  #组成观测点处的预测值矩阵
#        np.savetxt('y_obs_prediction_{0}_2.txt'.format(t),y_obs_prediction)
        y_obs_prediction=np.matrix(y_obs_prediction)
     
        S2=np.trace(((y_obs_prediction-y_obs_p1).T)*np.linalg.inv(Cd)*(y_obs_prediction-y_obs_p1))
        print 'max(m2-m1):',np.max(np.abs(m2-m1))
        print 'S2-S1:',S2-S1
        h.write('max(m2-m1):'+str(np.max(np.abs(m2-m1)))+'\n')
        h.write('S2-S1:'+str(S2-S1)+'\n')
        if (np.max(np.abs(m2-m1))<eps1 or S2-S1<eps2*S1 or Imax>6):
            break
        
        if S2<S1:
            m1=m2
            beta=2*beta
            if beta>1:
                beta=1
        else:
            beta=0.5*beta  
        Imax+=1
    np.savetxt('t_{0}.txt'.format(t),m2)
    parY[:,t-1]=np.average(m2,axis=1)
    np.savetxt('t_{0}_ave.txt'.format(t),parY[:,t-1])
    np.savetxt('parY.txt',parY)
#    x1,y1=profit_calcualtion_1(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)
#    x2,y2=profit_calcualtion_2(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v2,root_directory)
#    x3,y3=profit_calcualtion_3(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v3,root_directory)
#    production_initial[t-1]=y1+y2+y3
#    np.savetxt('production_initial.txt',production_initial)
#    
#    ##计算g部分,优化貌似只能到time_step-1个时间步，因为要留出一个时间步作为未来产量的预测去进行优化比较
    Itera=0
    while True:
        if t==20:
            break
        print 'Itera=',Itera
        h.write('{0}_time_opt_{1}_iteration'.format(t,Itera)+'\n')
        x_mat=np.zeros((len(x_ini),N))
        for i in xrange(len(x_ini)):
            for j in xrange(N):
                x_mat[i,j]=x_ini[i]+np.random.normal(0,0.001*x_ini[i])    #扰动偏差是当前压力的10%
        
        x_mat_ave=np.average(x_mat,axis=1)
        x_mat_error=np.zeros_like(x_mat)
        for i in xrange(N):
            x_mat_error[:,i]=x_mat[:,i]-x_mat_ave[:]
        
        for i in xrange(N):     #从当前时刻计算到最后，算这个时间段的g作为优化目标
            time_modify(1,i,delta_t,filename_time,root_directory)   
            modify_rfd_for_g(i,x_mat[:,i],root_directory,filename_rfd)
            
#        ppservers=()
#        job_server=pp.Server(2,ppservers=ppservers)
#        Ne=tuple(range(N))
#        jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#        for i,job in jobs:
#            job()
            
        jobs=[]
        for i in xrange(N):
            p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        
        time.sleep(3)
        Cx=np.dot(x_mat_error,x_mat_error.T)/(N-1)   #Cx是这么计算的吗？Cx是对角线上服从高斯分布，其余地方都为0，大小是多少呢？
        g_x_mat=np.zeros(N)
        for i in xrange(N):
            gas_cost1=(101325-x_mat[0,i])*gas_cost_ref
            gas_cost2=(101325-x_mat[1,i])*gas_cost_ref
            gas_cost3=(101325-x_mat[2,i])*gas_cost_ref
            x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost1,filename_v1,root_directory)
            x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost2,filename_v2,root_directory)
            x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost3,filename_v3,root_directory)
            g_x_mat[i]=x1+x2+x3
        
        g_x_mat_ave=np.average(g_x_mat)
        g_x_mat_error=np.zeros_like(g_x_mat)
        for i in xrange(N):
            g_x_mat_error[i]=g_x_mat[i]-g_x_mat_ave
        
        C_x_gx=np.zeros_like((x_mat_error))
        for i in range(N):
            C_x_gx[:,i]=x_mat_error[:,i]*g_x_mat_error[i]
        
        C_x_gx=np.sum(C_x_gx,axis=1)/(N-1)
        
        inner_itera=0
        while True:   ##在对比g(xl+1)和g(xl)的过程中也有一个小循环，只有前者大于后者才会退出,否则增大alpha，再去求解新的xl+1
            h.write('{0}_time_opt_{1}_inneriteration_{2}'.format(t,Itera,inner_itera)+'\n')
            delta_x=np.dot(Cx,C_x_gx)/alpha
#            np.savetxt('delta_x_{0}_{1}.txt'.format(t,Itera),delta_x)
            x_1=np.zeros_like(x_mat_ave)
            x_1=delta_x+x_mat_ave      #这里x更新均值，得到更新后的均值，在此基础上再一次加噪声变成样本
            x_1_mat=np.zeros_like(x_mat)
            for i in xrange(len(x_1)):
                for j in xrange(N):
                    x_1_mat[i,j]=x_1[i]+np.random.normal(0,0.001*x_1[i])  
                
            if np.min(x_1_mat)<0:
                break
            x_1_mat[x_1_mat<pressure_low]=pressure_low
            x_1_mat[x_1_mat>pressure_high]=pressure_high
            ##接下来要将x_1代入模型的边界条件，首先改写写入.bc文件的函数
            
            for i in xrange(N):
                modify_rfd_for_g(i,x_1_mat[:,i],root_directory,filename_rfd)
           
#            ppservers=()
#            job_server=pp.Server(2,ppservers=ppservers)
#            Ne=tuple(range(N))
#            jobs=[(i,job_server.submit(runexe,(i,root_directory),(),('win32api',))) for i in Ne]
#            for i,job in jobs:
#                job()
#            for i in xrange(N):
#                runexe(i,root_directory)
#                print '{0}_{1}_inner_itera'.format(t,i)
#                h.write('{0}_opt_inneritera_{1}_{2}'.format(t,inner_itera,i)+'\n')
#                time.sleep(1)
            jobs=[]
            for i in xrange(N):
                p=multiprocessing.Process(target=runexe,args=(i,root_directory,))
                jobs.append(p)
                p.start()
            for j in jobs:
                j.join()
            
            time.sleep(10) 
            for i in range(N):
                gas_cost1=(101325-x_1_mat[0,i])*gas_cost_ref
                gas_cost2=(101325-x_1_mat[1,i])*gas_cost_ref
                gas_cost3=(101325-x_1_mat[2,i])*gas_cost_ref
                x1,y1=profit_calcualtion_1(1,1,1,i,area_well,delta_t,gas_price,gas_cost1,filename_v1,root_directory)
                x2,y2=profit_calcualtion_2(1,1,1,i,area_well,delta_t,gas_price,gas_cost2,filename_v2,root_directory)
                x3,y3=profit_calcualtion_3(1,1,1,i,area_well,delta_t,gas_price,gas_cost3,filename_v3,root_directory)
                g_x_mat[i]=x1+x2+x3
            g_x_mat_ave_1=np.average(g_x_mat)
            x_1_mat_ave=np.average(x_1_mat,axis=1)
            print g_x_mat_ave_1,g_x_mat_ave
            h.write('g_x_mat_ave_1:'+str(g_x_mat_ave_1)+'g_x_mat_ave:'+str(g_x_mat_ave)+'\n')
            if g_x_mat_ave_1>g_x_mat_ave:
                x_ini=x_1_mat_ave
                break
            elif inner_itera>2:
                break
            else:
                alpha=1.5*alpha
            inner_itera+=1
        
        delta_g=(g_x_mat_ave_1-g_x_mat_ave)/g_x_mat_ave
        np.savetxt('x_{0}_{1}.txt'.format(t,Itera),x_1_mat)
        np.savetxt('x_{0}_{1}_ave.txt'.format(t,Itera),x_1_mat_ave)
        x_opt[:,t]=x_1_mat_ave
        np.savetxt('x_opt.txt',x_opt)
        x_change_rate=np.zeros_like(x_1_mat_ave)
        
#        #计算到当前时间步的累积产量
#        for i in xrange(N):
#            time_modify(t,i,delta_t,filename_time,root_directory)
#            modify_rfd_for_inverse(t,i,x_opt)
#            runexe(i,root_directory)
#            time.sleep(1)
#            h.write('{0}_opt_cal_cum_g_{1}'.format(t,i)+'\n')
#            
#        while True:
#            back_process=os.popen('tasklist |findstr /i ogs.exe').readlines()
#            if back_process==[]:
#                break
#        for i in xrange(N):
#            x1,y1=profit_calcualtion_1(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)
#            x2,y2=profit_calcualtion_2(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v2,root_directory)
#            x3,y3=profit_calcualtion_3(1,t,t,i,area_well,delta_t,gas_price,gas_cost,filename_v3,root_directory)
#        production_updated[t-1]=y1+y2+y3
        
        for i in xrange(len(x_1_mat_ave)):
            x_change_rate[i]=(x_1_mat_ave[i]-x_mat_ave[i])/x_mat_ave[i]   
        x_change_rate_ave=np.average(x_change_rate)    # 这里取的是控制变量向量中每个元素对应的变化率的平均
        print delta_g,x_change_rate_ave
        h.write('delta_g:'+str(delta_g)+'x_change_rate_ave:'+str(x_change_rate_ave)+'\n')
        if (0<delta_g<0.0005 and x_change_rate_ave<0.01):
            break
#        x_ini=x_1_mat_ave   #将迭代之后的x_1样本上取平均之后再赋给x_ini,再进行循环。
        Itera=Itera+1
        if Itera>3:
            break
h.close()
print time.ctime()
