# -*- coding: utf-8 -*-

import os
import numpy as np
##计算未来所有时间步的g的时候，采用的控制变量为当前的控制变量，并且自始至终不变
def modify_rfd_for_g(i,pressure_control,root_directory,filename_rfd):
    '''
    pressure_control是控制变量，为一个一维数组
    water_gas.rfd文件基于gas_OO中的文件，时间恒定为0-1000，这样总能计算完tim中定义的时间
    '''
    dirname='gas_{0}'.format(i)
    args=os.path.join(root_directory,dirname,filename_rfd)
    args_parent_rfd=os.path.join(root_directory,'gas_00',filename_rfd)
    with open(args_parent_rfd,'r') as f:
        content=f.readlines()
    
    content_1=content[6:8]
    content_2=content[9:11]
    content_3=content[12:14]
    for i in range(len(content_1)):
        content_1[i]=content_1[i].split()
    content_1=np.array(content_1)
    content_1=np.float64(content_1)
    content_1[:,1]=pressure_control[0]
    content_1_str=[]
    for i in range(len(content_1)):
        var=' '.join(str(x) for x in content_1[i])+'\n'
        content_1_str.append(var)
    content[6:8]=content_1_str
    
    for i in range(len(content_2)):
        content_2[i]=content_2[i].split()
    content_2=np.array(content_2)
    content_2=np.float64(content_2)
    content_2[:,1]=pressure_control[1]
    content_2_str=[]
    for i in range(len(content_2)):
        var=' '.join(str(x) for x in content_2[i])+'\n'
        content_2_str.append(var)
    content[9:11]=content_2_str
    
    for i in range(len(content_3)):
        content_3[i]=content_3[i].split()
    content_3=np.array(content_3)
    content_3=np.float64(content_3)
    content_3[:,1]=pressure_control[2]
    content_3_str=[]
    for i in range(len(content_3)):
        var=' '.join(str(x) for x in content_3[i])+'\n'
        content_3_str.append(var)
    content[12:14]=content_3_str
    
    with open(args,'w') as f:
        for line in content:
            f.write(line)



def modify_rfd_for_inverse(t,i,pressure_control,root_directory,filename_rfd):   #t是指当前时间步，pressure_control
    '''
    t是指当前时间步\n
    pressure_control是x_opt数组，取前t列
    基于主目录下的water_gas.rfd，时间基础是0,50，并依次在此基础上扩增，以计算到当前时刻
    '''
    args_parent_rfd=os.path.join(root_directory,filename_rfd)
    dirname='gas_{0}'.format(i)
    args=os.path.join(root_directory,dirname,filename_rfd)
    with open(args_parent_rfd,'r') as f:
        content=f.readlines()

    content_1=content[6:8]
    content_2=content[9:11]
    content_3=content[12:14]
    for i in range(len(content_1)):
        content_1[i]=content_1[i].split()
    content_1=np.array(content_1)
    content_1=np.float64(content_1)
    content_1[:,1]=pressure_control[0,0]
    content_1_str=[]
    for i in range(len(content_1)):
        var=' '.join(str(x) for x in content_1[i])+'\n'
        content_1_str.append(var)
    content_1_str_add=[]
    for tt in range(2,t+1):
        str1_1=' '.join([str((tt-1)*50+1),str(pressure_control[0,tt-1])])+'\n'
        content_1_str_add.append(str1_1)
        str1_2=' '.join([str(tt*50),str(pressure_control[0,tt-1])])+'\n'
        content_1_str_add.append(str1_2)
    content[6:8]=content_1_str+content_1_str_add
    
    
    for i in range(len(content_2)):
        content_2[i]=content_2[i].split()
    content_2=np.array(content_2)
    content_2=np.float64(content_2)
    content_2[:,1]=pressure_control[1,0]
    content_2_str=[]
    for i in range(len(content_2)):
        var=' '.join(str(x) for x in content_2[i])+'\n'
        content_2_str.append(var)
    content_2_str_add=[]
    for tt in range(2,t+1):
        str2_1=' '.join([str((tt-1)*50+1),str(pressure_control[1,tt-1])])+'\n'
        content_2_str_add.append(str2_1)
        str2_2=' '.join([str(tt*50),str(pressure_control[1,tt-1])])+'\n'
        content_2_str_add.append(str2_2)
    content[9+len(content_1_str_add):11+len(content_1_str_add)]=content_2_str+content_2_str_add
    
    for i in range(len(content_3)):
        content_3[i]=content_3[i].split()
    content_3=np.array(content_3)
    content_3=np.float64(content_3)
    content_3[:,1]=pressure_control[2,0]
    content_3_str=[]
    for i in range(len(content_3)):
        var=' '.join(str(x) for x in content_3[i])+'\n'
        content_3_str.append(var)
    content_3_str_add=[]
    for tt in range(2,t+1):
        str3_1=' '.join([str((tt-1)*50+1),str(pressure_control[2,tt-1])])+'\n'
        content_3_str_add.append(str3_1)
        str3_2=' '.join([str(tt*50),str(pressure_control[2,tt-1])])+'\n'
        content_3_str_add.append(str3_2)
    content[12+2*len(content_1_str_add):14+2*len(content_1_str_add)]=content_3_str+content_3_str_add
    
    with open(args,'w') as f:
        for line in content:
            f.write(line)

if __name__=='__main__':
    i=12
    pressure_control=np.array([98000,98123,84666])
    current_directory=os.getcwd()
    root_directory=os.path.dirname(current_directory)
    filename_rfd='water_gas.rfd'
#    modify_rfd_for_g(i,pressure_control,root_directory,filename_rfd)
#    from time_modify import time_modify
#    time_modify(12,12,50,'water_gas.tim',root_directory)
    x_opt=np.zeros((3,5))
    x_opt[:,0]=2000
    x_opt[:,1]=2000
    x_opt[:,2]=3000
    x_opt[:,3]=5000
    t=1
    modify_rfd_for_inverse(t,i,x_opt,root_directory,filename_rfd)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    