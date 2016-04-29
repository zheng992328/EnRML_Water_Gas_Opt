#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'zq'

import os

##之前的write_ki是修改的每一行的第二个数，现在改变思路，直接生成要输入的值，一下子覆盖进去
def write_ki(Nod_num,para,i,filename_KI,current_directory):
    value_list={}   #产生要添加的ki序列
    gas_file='gas_{0}'.format(i)
    args_ki=os.path.join(current_directory,gas_file,filename_KI)
    for i in xrange(Nod_num):
        value_list[i]=para[i]
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])  #使用join进行字符串的拼接效率更高
        value_list_to_str.append(var)
    with open(args_ki,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')


# 定义函数，将更新之后的p1的值写入direct文件
def write_p1(Nod_num,p1,i,filename_Pressure,current_directory):
    value_list={}   #产生要添加的p1序列
    gas_file='gas_{0}'.format(i)
    args_p1=os.path.join(current_directory,gas_file,filename_Pressure)
    for i in xrange(Nod_num):
        value_list[i]=p1[i]
    value_list_to_str=[]    #将序列转换成可以写入文件的字符串格式
    for i,j in value_list.iteritems():
        var=' '.join([str(i),str(j),'\n'])
        value_list_to_str.append(var)
    with open(args_p1,'w') as f:
        for line in value_list_to_str:
            f.write(line)
            pass
        f.write('#STOP')


def write_bc(i,x,filename_bc,current_directory):
    x1,x2,x3=x
    gas_file='gas_{0}'.format(i)
    args_path=os.path.join(current_directory,gas_file,filename_bc)
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













