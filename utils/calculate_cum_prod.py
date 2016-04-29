# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from calculate_q_cum import profit_calcualtion_1,profit_calcualtion_2,profit_calcualtion_3
current_directory=os.getcwd()
root_directory=os.path.dirname(current_directory)
filename_rfd='water_gas.rfd'
args_x_opt=os.path.join(root_directory,'x_opt.txt')
args_exe=os.path.join(root_directory,'cumulative_production','ogs')
args_para=os.path.join(root_directory,'cumulative_production','water_gas')
args=os.path.join(root_directory,'cumulative_production')
filename_v1='water_gas_ply_PLY_2_t1.tec'
filename_v2='water_gas_ply_PLY_3_t2.tec'
filename_v3='water_gas_ply_PLY_4_t3.tec'
filename_KI='water_gas_Ki.direct'
t=20
delta_t=50
Nod_num=1681
area_well=2
gas_price=16
x_opt=pd.read_csv(args_x_opt,sep=' ',names=[i for i in range(20)])
x_opt=x_opt.values
x_ini=x_opt[:,0]
production_initial=np.zeros((3,t))
production_updated=np.zeros_like(production_initial)

#将真实参数场写入ki.direct文件
args_para_true=os.path.join(root_directory,'para_true.txt')
para_true=pd.read_csv(args_para_true,sep=' ',names=[1])
para_true=para_true.values
para_true=np.squeeze(para_true)

##如果将真实参数场换成最后反演下来的结果呢
#args_para_true=r'E:\water_gas_line_opt\t_20_ave.txt'
#para_true=pd.read_csv(args_para_true,sep=' ',names=[1])
#para_true=para_true.values
#para_true=np.squeeze(para_true)



para=np.exp(para_true)
value_list={}   #产生要添加的ki序列
for i in xrange(Nod_num):
    value_list[i]=para[i]
value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
for i,j in value_list.iteritems():
    var=' '.join([str(i),str(j),'\n'])  #使用join进行字符串的拼接效率更高
    value_list_to_str.append(var)
args_KI=os.path.join(root_directory,'cumulative_production',filename_KI)
with open(args_KI,'w') as f:
    for line in value_list_to_str:
        f.write(line)
        pass
    f.write('#STOP')





#计算优化前的控制变量作用于系统整个生产期的累积产量
args_rfd=os.path.join(root_directory,'cumulative_production','water_gas.rfd')
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
content_1[:,1]=x_ini[0]
content_1_str=[]
for i in range(len(content_1)):
    var=' '.join(str(x) for x in content_1[i])+'\n'
    content_1_str.append(var)
content[6:8]=content_1_str

for i in range(len(content_2)):
    content_2[i]=content_2[i].split()
content_2=np.array(content_2)
content_2=np.float64(content_2)
content_2[:,1]=x_ini[1]
content_2_str=[]
for i in range(len(content_2)):
    var=' '.join(str(x) for x in content_2[i])+'\n'
    content_2_str.append(var)
content[9:11]=content_2_str

for i in range(len(content_3)):
    content_3[i]=content_3[i].split()
content_3=np.array(content_3)
content_3=np.float64(content_3)
content_3[:,1]=x_ini[2]
content_3_str=[]
for i in range(len(content_3)):
    var=' '.join(str(x) for x in content_3[i])+'\n'
    content_3_str.append(var)
content[12:14]=content_3_str
    
with open(args_rfd,'w') as f:
    for line in content:
        f.write(line)
p=subprocess.Popen((args_exe,args_para),stdout=subprocess.PIPE)
p.communicate()
p.wait()
x_ini_1=np.array([x_ini[0]]*t)
x_ini_2=np.array([x_ini[1]]*t)
x_ini_3=np.array([x_ini[2]]*t)
x1=profit_calcualtion_1(1,t,t,area_well,delta_t,gas_price,filename_v1,root_directory,x_ini_1)
x2=profit_calcualtion_2(1,t,t,area_well,delta_t,gas_price,filename_v2,root_directory,x_ini_2)
x3=profit_calcualtion_3(1,t,t,area_well,delta_t,gas_price,filename_v3,root_directory,x_ini_3)
production_initial[0,:]=x1.cumsum()
production_initial[1,:]=x2.cumsum()
production_initial[2,:]=x3.cumsum()
production_initial_sum=production_initial.sum(axis=0)
production_initial_sum=production_initial_sum*area_well*delta_t


##计算优化后的production
args_parent_rfd=os.path.join(root_directory,filename_rfd)
with open(args_parent_rfd,'r') as f:
    content=f.readlines()

content_1=content[6:8]
content_2=content[9:11]
content_3=content[12:14]
for i in range(len(content_1)):
    content_1[i]=content_1[i].split()
content_1=np.array(content_1)
content_1=np.float64(content_1)
content_1[:,1]=x_opt[0,0]
content_1_str=[]
for i in range(len(content_1)):
    var=' '.join(str(x) for x in content_1[i])+'\n'
    content_1_str.append(var)
content_1_str_add=[]
for tt in range(2,t+1):
    str1_1=' '.join([str((tt-1)*50+1),str(x_opt[0,tt-1])])+'\n'
    content_1_str_add.append(str1_1)
    str1_2=' '.join([str(tt*50),str(x_opt[0,tt-1])])+'\n'
    content_1_str_add.append(str1_2)
content[6:8]=content_1_str+content_1_str_add


for i in range(len(content_2)):
    content_2[i]=content_2[i].split()
content_2=np.array(content_2)
content_2=np.float64(content_2)
content_2[:,1]=x_opt[1,0]
content_2_str=[]
for i in range(len(content_2)):
    var=' '.join(str(x) for x in content_2[i])+'\n'
    content_2_str.append(var)
content_2_str_add=[]
for tt in range(2,t+1):
    str2_1=' '.join([str((tt-1)*50+1),str(x_opt[1,tt-1])])+'\n'
    content_2_str_add.append(str2_1)
    str2_2=' '.join([str(tt*50),str(x_opt[1,tt-1])])+'\n'
    content_2_str_add.append(str2_2)
content[9+len(content_1_str_add):11+len(content_1_str_add)]=content_2_str+content_2_str_add

for i in range(len(content_3)):
    content_3[i]=content_3[i].split()
content_3=np.array(content_3)
content_3=np.float64(content_3)
content_3[:,1]=x_opt[2,0]
content_3_str=[]
for i in range(len(content_3)):
    var=' '.join(str(x) for x in content_3[i])+'\n'
    content_3_str.append(var)
content_3_str_add=[]
for tt in range(2,t+1):
    str3_1=' '.join([str((tt-1)*50+1),str(x_opt[2,tt-1])])+'\n'
    content_3_str_add.append(str3_1)
    str3_2=' '.join([str(tt*50),str(x_opt[2,tt-1])])+'\n'
    content_3_str_add.append(str3_2)
content[12+2*len(content_1_str_add):14+2*len(content_1_str_add)]=content_3_str+content_3_str_add

with open(args_rfd,'w') as f:
    for line in content:
        f.write(line)

p=subprocess.Popen((args_exe,args_para),stdout=subprocess.PIPE)
p.communicate()
p.wait()

x_up_1=x_opt[0,:]
x_up_2=x_opt[1,:]
x_up_3=x_opt[2,:]
x1_1=profit_calcualtion_1(1,t,t,area_well,delta_t,gas_price,filename_v1,root_directory,x_up_1)
x2_1=profit_calcualtion_2(1,t,t,area_well,delta_t,gas_price,filename_v2,root_directory,x_up_2)
x3_1=profit_calcualtion_3(1,t,t,area_well,delta_t,gas_price,filename_v3,root_directory,x_up_3)
production_updated[0,:]=x1_1.cumsum()
production_updated[1,:]=x2_1.cumsum()
production_updated[2,:]=x3_1.cumsum()
production_updated_sum=production_updated.sum(axis=0)
production_updated_sum=production_updated_sum*delta_t*area_well
##作图
X=np.arange(1,t+1)
plt.plot(X,production_initial_sum,'b',label='initial')
plt.plot(X,production_updated_sum,'r',label='optimized')
plt.legend(loc='best')
plt.xlabel('time step')
plt.ylabel('Net present value($)')
plt.show()


