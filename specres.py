import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
mypath = '/Users/turnertopping/XRhrexSoft/gait_optim/python/SR/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
for filla in onlyfiles:
	if filla[-4:] != '.dyl':
		onlyfiles.remove(filla)
onlyfiles.remove('Parameters.py')

print onlyfiles
frontz = '/Users/turnertopping/XRhrexSoft/gait_optim/python/SR/'
#b = 'xrhex_log_20150707_155457.dyl'
#c=a+b
xmass = 8.106
g = 9.8
execfile('/Users/turnertopping/XRhrexSoft/xrhex_software/scripts/dyLogParser.py')
sres=[]
vels=[]
pavava=[]
for fil3 in onlyfiles:
	#print fil3
	p = LogParser(frontz+fil3)

	kstart,kend = 0,0
	for i in range(1,len(p.april.pose.d[1,:])):
		klast = p.april.pose.d[1,i-1]
		if (p.april.pose.d[1,i] - klast > 0):
			kend = i+1

	a = list(p.april.pose.d[1,:])
	a.reverse()
	for i in range(1,len(a)):
		klast = a[i-1]
		kcur = a[i]
		if kcur-klast != 0:
			kstart = len(a)-i

	t = p.april.pose.t[kstart:kend]
	y = p.april.pose.d[1,kstart:kend]
	#print t
	#print y
	#print kstart,kend
	#plot(p.april.pose.t[:],p.april.pose.d[1,:])
	#plot(t,y)

	x = p.april.pose.d[0,kstart:kend]
	delx = x[-1]-x[0]
	dely = y[-1]-y[0]
	E = p.battery.energy_used.d[kstart:kend]
	delE = E[-1]-E[0]
	delt = t[-1]-t[0]
	#sres = dE*3600/sqrt(dx**2+dy**2)
	#plot(t,E)
	#print dE
	#print sres
	V = p.battery.voltage.d[kstart:kend]
	I = p.battery.current.d[kstart:kend]
	P = V*I
	#plot(t,P)
	#Pavg = mean(P)
	Pavg = delE*3600
	dis = sqrt(delx**2+dely**2)
	avvel = dis/delt
	sres.append(Pavg/(avvel*xmass*g))
	pavava.append(Pavg)
	vels.append(avvel)
	#plot(t,V)
	#plot(t,I)
#print sres
#n, bins, patches = plt.hist(sres, 50, normed=1, facecolor='green', alpha=0.75)
pavava = array(pavava)
hist(array(vels))
plt.show()
plt.figure()
hist(pavava)
plt.show()
plt.figure()
hist(array(sres))
plt.show()
print sres