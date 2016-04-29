# -*- coding: utf-8 -*-

##实现了从t_start计算到t_end时间步的累积产量和净现值
import numpy as np
import os
def profit_calcualtion_1(t_start,t_end,time_step,i,area_well,delta_t,gas_price,gas_cost,filename_v1,current_directory):
    '''
    由于只向后预测一个时间步，gas_cost只有一个值，因此由外部函数根据pressure定义，传进来即可
    '''
    gas_file='gas_{0}'.format(i)
    args_path_1=os.path.join(current_directory,gas_file,filename_v1)
#    args_path_1=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}\water_gas_ply_PLY_2_t1.tec'.format(i)   
    with open(args_path_1,'r') as f:
        content=f.readlines()
    v_sum_well_1=np.zeros(t_end-t_start+1)
    for x in range(t_start,t_end+1):
        if x==time_step:
            content_1=content[-25:]
        else:
            m=time_step+1-x
            content_1=content[-(m*25+m-1):-(26*(m-1))]
        for i in range(len(content_1)):
            content_1[i]=content_1[i].split()
        content_1=np.array(content_1)
        content_1=np.float64(content_1)
        v_1=np.zeros(len(content_1))
        for i in range(len(content_1)): 
            v_1[i]=np.sqrt(np.power(content_1[i,1],2)+np.power(content_1[i,2],2))
        v_sum_1=sum(v_1)
        v_sum_well_1[x-t_start]=v_sum_1
    
    q_sum_1=(sum(v_sum_well_1))*area_well*delta_t
    profit_1=q_sum_1*gas_price-gas_cost  
    return profit_1,q_sum_1

def profit_calcualtion_2(t_start,t_end,time_step,i,area_well,delta_t,gas_price,gas_cost,filename_v2,current_directory): 
    gas_file='gas_{0}'.format(i)
    args_path_2=os.path.join(current_directory,gas_file,filename_v2)
#    args_path_2=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}\water_gas_ply_PLY_3_t2.tec'.format(i)   
    with open(args_path_2,'r') as f:
        content2=f.readlines()
    
    v_sum_well_2=np.zeros(t_end-t_start+1)
    for x in range(t_start,t_end+1):
        if x==time_step:
            content_2=content2[-25:]
        else: 
            m=time_step+1-x
            content_2=content2[-(m*25+m-1):-(26*(m-1))]
        for i in range(len(content_2)):
            content_2[i]=content_2[i].split()
        content_2=np.array(content_2)
        content_2=np.float64(content_2)
        v_2=np.zeros(len(content_2))
        for i in range(len(content_2)): 
            v_2[i]=np.sqrt(np.power(content_2[i,1],2)+np.power(content_2[i,2],2))
        v_sum_2=sum(v_2)
        v_sum_well_2[x-t_start]=v_sum_2
    
    q_sum_2=(sum(v_sum_well_2))*area_well*delta_t
    profit_2=q_sum_2*gas_price-gas_cost
    return profit_2,q_sum_2

def profit_calcualtion_3(t_start,t_end,time_step,i,area_well,delta_t,gas_price,gas_cost,filename_v3,current_directory): 
    gas_file='gas_{0}'.format(i)
    args_path_3=os.path.join(current_directory,gas_file,filename_v3)
#    args_path_3=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}\water_gas_ply_PLY_4_t3.tec'.format(i)  
    with open(args_path_3,'r') as f:
        content3=f.readlines()
    v_sum_well_3=np.zeros(t_end-t_start+1)
    for x in range(t_start,t_end+1):
        if x==time_step:   #如果只要取最后一个时间步的
            content_3=content3[-25:]
        else:  
            m=time_step+1-x
            content_3=content3[-(m*25+m-1):-(26*(m-1))]
        for i in range(len(content_3)):
            content_3[i]=content_3[i].split()
        content_3=np.array(content_3)
        content_3=np.float64(content_3)
        v_3=np.zeros(len(content_3))
        for i in range(len(content_3)): 
            v_3[i]=np.sqrt(np.power(content_3[i,1],2)+np.power(content_3[i,2],2))
        v_sum_3=sum(v_3)
        v_sum_well_3[x-t_start]=v_sum_3
    q_sum_3=(sum(v_sum_well_3))*area_well*delta_t
    profit_3=q_sum_3*gas_price-gas_cost
    return profit_3,q_sum_3


if __name__=='__main__': 
    time_step=12
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
    t_start=1
    t_end=12
    
    i=5
    area_well=2
    delta_t=500
#    y1,y2=profit_calcualtion_1(t_start,t_end,time_step-1,i,area_well,delta_t,gas_price,gas_cost,filename_v1,root_directory)
    gas_file='gas_{0}'.format(i)
    args_path_1=os.path.join(root_directory,gas_file,filename_v1)
#    args_path_1=r'E:\EnRML_Gas_Modelling\water_gas\water_gas_line_3_19\gas_{0}\water_gas_ply_PLY_2_t1.tec'.format(i)   
    with open(args_path_1,'r') as f:
        content=f.readlines()
    v_sum_well_1=np.zeros(t_end-t_start+1)
    for x in range(t_start,t_end+1):
        if x==time_step:
            content_1=content[-25:]
        else:
            m=time_step+1-x
            content_1=content[-(m*25+m-1):-(26*(m-1))]
        for i in range(len(content_1)):
            content_1[i]=content_1[i].split()
        content_1=np.array(content_1)
        content_1=np.float64(content_1)
        print content_1
        
        
        v_1=np.zeros(len(content_1))
        for i in range(len(content_1)): 
            v_1[i]=np.sqrt(np.power(content_1[i,1],2)+np.power(content_1[i,2],2))
        v_sum_1=sum(v_1)
        v_sum_well_1[x-t_start]=v_sum_1 
    q_sum_1=(sum(v_sum_well_1))*area_well*delta_t
    profit_1=q_sum_1*gas_price-gas_cost
 
    
