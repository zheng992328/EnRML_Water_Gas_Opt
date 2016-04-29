# -*- coding: utf-8 -*-



import numpy as np
import os
from generateL import generateL
import pandas as pd
def generate_para_true(sigma,deltax,deltay,dx,dy,m,n,ki_mean,Nod_num):
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
    args=os.path.join(root_directory,'true_obs',filename_KI)
    with open(args,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')

#def write_BC(x,filename_bc,current_directory):
#    x1,x2,x3=x
#    args_path=os.path.join(current_directory,'true_obs',filename_bc)
#    with open(args_path,'r') as  f:
#        content=f.readlines()
#        line_modify_1=content[9]
#        line_modify_2=content[18]
#        line_modify_3=content[27]
#        modify_1=line_modify_1.split()
#        modify_2=line_modify_2.split()
#        modify_3=line_modify_3.split()
#        modify_1[1]=str(x1)
#        line_modify_1=' '.join(modify_1[i] for i in range(len(modify_1)))+'\n'
#        modify_2[1]=str(x2)
#        line_modify_2=' '.join(modify_2[i] for i in range(len(modify_2)))+'\n'
#        modify_3[1]=str(x3)
#        line_modify_3=' '.join(modify_3[i] for i in range(len(modify_3)))+'\n'
#        content[9]=line_modify_1
#        content[18]=line_modify_2
#        content[27]=line_modify_3
#    with open(args_path,'w') as f:
#        for line in content:
#            f.write(line)


def generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR,root_directory):      ##产生后面一个时间步的p2观测值样本 （4.13）
    domain=filename_result
    args_domain=os.path.join(root_directory,'true_obs',domain)
    with open(args_domain,'r') as f:
        content=f.readlines()
    content1=content[-(1600+Nod_num):-1600]               #踢掉tec前面3行头数据
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


def time_MODIFY(t,delta_t,filename_time,current_directory):
    gas_file='true_obs'
    args_tim=os.path.join(current_directory,gas_file,filename_time)
    with open(args_tim,'r') as f:
        content=f.readlines()
        line_modify_1=content[5]
        line_modify_2=content[7]
        line_modify_3=content[14]
        line_modify_4=content[16]
        modify_1=line_modify_1.split()
        modify_1[0]='{0}'.format(t)
        line_modify_1=' '.join(modify_1[i] for i in range(len(modify_1)))+'\n'
        modify_2=line_modify_2.split()
        modify_2[0]='{0}'.format(t*delta_t)
        line_modify_2=' '.join(modify_2[i] for i in range(len(modify_2)))+'\n'
        modify_3=line_modify_3.split()
        modify_3[0]='{0}'.format(t)
        line_modify_3=' '.join(modify_3[i] for i in range(len(modify_3)))+'\n'
        modify_4=line_modify_4.split()
        modify_4[0]='{0}'.format(t*delta_t)
        line_modify_4=' '.join(modify_4[i] for i in range(len(modify_4)))+'\n'
        content[5]=line_modify_1
        content[7]=line_modify_2
        content[14]=line_modify_3
        content[16]=line_modify_4
    with open(args_tim,'w') as f:
        for line in content:
            f.write(line)

#def rfd_MODIIFY(pressure_control,root_directory,filename_rfd):   #pressure_control
#    dirname='true_obs'
#    args=os.path.join(root_directory,dirname,filename_rfd)
#    args_parent_rfd=os.path.join(root_directory,'gas_00',filename_rfd)
#    with open(args_parent_rfd,'r') as f:
#        content=f.readlines()
#    
#    content_1=content[6:8]
#    content_2=content[9:11]
#    content_3=content[12:14]
#    for i in range(len(content_1)):
#        content_1[i]=content_1[i].split()
#    content_1=np.array(content_1)
#    content_1=np.float64(content_1)
#    content_1[:,1]=pressure_control[0]
#    content_1_str=[]
#    for i in range(len(content_1)):
#        var=' '.join(str(x) for x in content_1[i])+'\n'
#        content_1_str.append(var)
#    content[6:8]=content_1_str
#    
#    for i in range(len(content_2)):
#        content_2[i]=content_2[i].split()
#    content_2=np.array(content_2)
#    content_2=np.float64(content_2)
#    content_2[:,1]=pressure_control[1]
#    content_2_str=[]
#    for i in range(len(content_2)):
#        var=' '.join(str(x) for x in content_2[i])+'\n'
#        content_2_str.append(var)
#    content[9:11]=content_2_str
#    
#    for i in range(len(content_3)):
#        content_3[i]=content_3[i].split()
#    content_3=np.array(content_3)
#    content_3=np.float64(content_3)
#    content_3[:,1]=pressure_control[2]
#    content_3_str=[]
#    for i in range(len(content_3)):
#        var=' '.join(str(x) for x in content_3[i])+'\n'
#        content_3_str.append(var)
#    content[12:14]=content_3_str
#    
#    with open(args,'w') as f:
#        for line in content:
#            f.write(line)

##从初始计算到t时刻
def rfd_MODIIFY(t,pressure_control,root_directory,filename_rfd):   #pressure_control
    dirname='true_obs'
    args_parent_rfd=os.path.join(root_directory,filename_rfd)
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
    delta_t=50
    x=2
    y=1
    m=int(y/dy+1)
    n=int(x/dx+1)
    filename_KI='water_gas_KI.direct'
    filename_time='water_gas.tim'
    filename_rfd='water_gas.rfd'
#    finename_Pressure='water_gas_PRESSURE1.direct'
    filename_result='water_gas_domain_quad.tec'
#    para_true=generate_para_true(sigma,deltax,deltay,dx,dy,m,n,ki_mean,root_directory)
#    para_true_modified=para_keys_modify(para_true)
#    np.savetxt(os.path.join(root_directory,'para_true.txt'),para_true_modified)
#    para_distribution_map(para_true_modified,Nod_num,root_directory)
#    write_KI(Nod_num,para_true,root_directory,filename_KI)
#    read_trueobs(20,obs_Num,Nod_num,root_directory,filename_result)
    y=generate_real_time_obs(filename_result,obs_Num,Nod_num,N,varR)
    
    
    
    