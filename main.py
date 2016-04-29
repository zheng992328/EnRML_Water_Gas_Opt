import os
import subprocess
import re
import shutil
import time
for m in range(10):
	m=0
	print 'the {0} time calculation'.format(m)
	print time.ctime()
	p1=subprocess.Popen(("./utils/generate_files.py"),shell=True)
	p1.wait()
	print 'start calculation'
	p2=subprocess.Popen("./EnRML_Water_Gas_Opt_Parallel.py",shell=True,stderr=subprocess.PIPE)
	p2.communicate()
	p2.wait()
	print 'asss'
	if os.path.exists('./time_{0}'.format(m))==False:
		os.makedirs('time_{0}'.format(m))
	filelist=os.listdir(os.getcwd())
	for i in filelist:
		pattern=re.search('t_[0-9].txt',i)
		if pattern:
			newfile=pattern.group()
			shutil.move(newfile,'./time_{0}'.format(m))
		pattern1=re.search('t_[0-9]_ave.txt',i)
		if pattern1:
			newfile1=pattern1.group()
			shutil.move(newfile1,'./time_{0}'.format(m))
		pattern2=re.search('t_[0-9][0-9].txt',i)
		if pattern2:
			newfile2=pattern2.group()
			shutil.move(newfile2,'./time_{0}'.format(m))
		pattern3=re.search('t_[0-9][0-9]_ave.txt',i)
		if pattern3:
			newfile3=pattern3.group()
			shutil.move(newfile3,'./time_{0}'.format(m))
	shutil.move('./x_opt.txt','./time_{0}'.format(m))
	shutil.move('./para_true.txt','./time_{0}'.format(m))
	shutil.move('./log.txt','./time_{0}'.format(m))
	shutil.move('./parY.txt','./time_{0}'.format(m))
