#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'zq'

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
class post_calculation:
    '''
    t is the current time-step
    time_step is the total timesteps
    Nod_num is the number of nodes
    '''
    def __init__(self,t,time_step,Nod_num):
        self.t=t
        self.time_step=time_step
        self.Nod_num=Nod_num

    def spread_calculation(self):
        a=np.zeros((self.t))
        current_directory=os.getcwd()
        parent_directory=os.path.dirname(current_directory)
        for i in range(1,self.t+1):
            args=os.path.join(parent_directory,'t_{0}.txt'.format(i))
            f=open(args,'r')
            content=f.readlines()
            for line in range(len(content)):
                content[line]=content[line].split()
                pass
            content=np.array(content)
            content=np.float64(content)
            var=np.var(content,axis=1)
            std=np.sqrt(np.mean(var,axis=0))
            a[i-1]=std
            pass
        x=[x+1 for x in range(self.t)]
        print a
        plt.plot(x,a,'r')
        plt.title('spread')
        plt.show()


    def rmse_calculation(self):
        current_directory=os.getcwd()
        parent_directory=os.path.dirname(current_directory)
#        root_directory=os.path.join(parent_directory,'true_obs')
        para_args=os.path.join(root_directory,'para_true.txt')
        with open(para_args,'r') as f:
            content=f.readlines()
        for i in range(len(content)):
            content[i]=content[i].split()

        content=np.array(content)
        content=np.float64(content)
        content=content[:,0]
        content=np.array(content)
        para_true=content
  

        
        parY_args=os.path.join(parent_directory,'parY.txt')
        with open(parY_args,'r') as f:
            content_parY=f.readlines()
        for i in range(len(content_parY)):
            content_parY[i]=content_parY[i].split()
        content_parY=np.array(content_parY)
        content_parY=np.float64(content_parY)
        parY=content_parY

        rmse=np.zeros((self.time_step,1))
        for i in range(self.t):
            par_Y=parY[:,i]
#            print par_Y.shape
            rmse[i,0]=math.sqrt(sum((par_Y-para_true)**2)/(self.Nod_num-1))
        X=[i+1 for i in range(self.t)]
        Y=rmse[0:self.t,0]
        plt.plot(X,Y,'r')
        plt.title('rmse')
        plt.show()
        return rmse
def production_comp_map(time,initial_name,updated_name,root_directory):
    args_initial=os.path.join(root_directory,initial_name)
    args_updated=os.path.join(root_directory,updated_name)
    production_initial=pd.read_csv(args_initial,sep=' ',names=[1])
    production_updated=pd.read_csv(args_updated,sep=' ',names=[1])
    production_initial=np.squeeze(production_initial.values)
#    production_initial=production_initial[:19]
#    production_updated=production_updated[:19]
    production_updated=np.squeeze(production_updated.values)
    x=np.arange(time)
    plt.plot(x,production_initial,label='initial production')
    plt.plot(x,production_updated,label='updated production')
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('production(m3)')
    plt.show()

#优化后的气压随时间分布图
def opt_pressure_map(root_diretory,filename):
    args=os.path.join(root_directory,filename)
    x_opt=pd.read_csv(args,sep=' ',names=[i for i in range(20)])
    x_opt=x_opt.T
    x_opt.columns=['well 1','well 2','well 3']
    x_opt.plot()
    plt.xlabel('time step')
    plt.ylabel('pressure(Pa)')
    plt.legend(loc=1)
    plt.show()
    
    

if __name__=='__main__':
    time=20
    current_directory=os.getcwd()
    root_directory=os.path.dirname(current_directory)
    initial_name='production_initial.txt'
    updated_name='production_updated.txt'
    pressure_name='x_opt.txt'
#    production_comp_map(time-1,initial_name,updated_name,root_directory)
    a=post_calculation(time,20,1681)
    a.spread_calculation()
    print a.rmse_calculation()
    opt_pressure_map(root_directory,pressure_name)
