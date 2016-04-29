# -*- coding: utf-8 -*-

##构造[x,g(x)]矩阵
import numpy as np
import os
from calculate_q_well import profit_calcualtion_1,profit_calcualtion_2,profit_calcualtion_3

def profit_calculation(t,i,area_well,delta_t,gas_price,gas_cost,filename_v1,filename_v2,filename_v3,root_directory):
    x1,y1=profit_calcualtion_1(t,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)
    x2,y2=profit_calcualtion_2(t,i,area_well,delta_t,gas_price,gas_cost,filename_v2,root_directory)
    x3,y3=profit_calcualtion_3(t,i,area_well,delta_t,gas_price,gas_cost,filename_v3,root_directory)
    profit_sum=x1+x2+x3
    return profit_sum,y1,y2,y3
 
if __name__=='__main__': 
    N=100
    pressure_low=97500
    pressure_high=98500
    well_num=3
    x_ini=np.random.uniform(pressure_low,pressure_high,well_num)
    filename_v1='water_gas_ply_PLY_2_t1.tec'
    filename_v2='water_gas_ply_PLY_3_t2.tec'
    filename_v3='water_gas_ply_PLY_4_t3.tec'
    gas_cost=0    #0肯定是不合实际的
    gas_price=0.16   ##0.16d/m3
    alpha=0.3
    Itera=0
    current_directory=os.getcwd()
    root_directory=os.path.dirname(current_directory)
    #gas_0中的流速计算出来的通量对应到opt_mat[3,0],以此类推
    #计算gas_0中的通量
    t=1
    i=7
    area_well=2
    delta_t=500
    profit,y1,y2,y3=profit_calculation(t,i,area_well,delta_t,gas_price,gas_cost,filename_v1,filename_v2,filename_v3,root_directory)
    
    
    
#    while True:
#        print 'Itera',Itera
#        x=np.zeros((len(x_ini),N))
#        for i in range(len(x_ini)):
#            for j in range(N):
#                x[i,j]=x_ini[i]+np.random.normal(0,3)
#        
#        x_mat=x
#        x_mat_ave=np.average(x_mat,axis=1)
#        x_mat_error=np.zeros_like(x_mat)
#        for i in xrange(N):
#            x_mat_error[:,i]=x_mat[:,i]-x_mat_ave[:]
#        
#        Cx=np.dot(x_mat_error,x_mat_error.T)/(N-1)   #Cx不是这么计算的，Cx是对角线上服从高斯分布，其余地方都为0，大小是多少呢？
#        g_x_mat=np.zeros(N)
#        for i in range(N):
#            g_x_mat[i]=profit_calcualtion(i)
#            
#        g_x_mat_ave=np.average(g_x_mat)
#        g_x_mat_error=np.zeros_like(g_x_mat)
#        for i in xrange(N):
#            g_x_mat_error[i]=g_x_mat[i]-g_x_mat_ave
#        
#        C_x_gx=np.zeros_like((x_mat_error))
#        for i in range(N):
#            C_x_gx[:,i]=x_mat_error[:,i]*g_x_mat_error[i]
#        
#        C_x_gx=np.sum(C_x_gx,axis=1)/(N-1)
#        ##是对每个样本进行迭代计算，然后每个样本都有一个更新值，还是样本的作用只是用来求解G，然后用原来的均值去迭代更新，更新后还是得到均值
#        ##姑且采用每个样本都单独进行更新
#        delta_x=np.dot(Cx,C_x_gx)/alpha
#        x_1=np.zeros_like(x)
#        for i in range(N):
#            x_1[:,i]=delta_x[:]+x[:,i]
#        
#        ##接下来要将x_1代入模型的边界条件，首先改写写入.bc文件的函数
#        from write_values import write_bc
#        for i in range(N):
#            x=x_1[:,i]
#            write_bc(i,x)
#        
#        
#        for i in range(N):
#            args_exe=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}\ogs.exe'.format(i)
#            args=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}'.format(i)
#            win32api.ShellExecute(0,'open',args_exe,'water_gas',args,0)
#            time.sleep(3)
#            print '{0}_{1}'.format(Itera,i)
#            
#        for i in range(N):
#            g_x_mat[i]=profit_calcualtion(i)
#        g_x_mat_ave_1=np.average(g_x_mat)
#        delta_g=(g_x_mat_ave_1-g_x_mat_ave)/g_x_mat_ave
#        if delta_g<0.0001:
#            break
#        x_ini=x_1
#        Itera=Itera+1



