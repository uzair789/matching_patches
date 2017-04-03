import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from numpy import linalg as LA
from numpy.linalg import inv

f = open('X_new.txt','r')
data = []
for line in f:
	a =  line.split(" ")
	a.pop()
	data.append(a)

data = np.array(data,dtype = np.float)
#print data

gmm = GaussianMixture(n_components = 3,covariance_type = 'spherical').fit(data)
pred = gmm.predict(data)


for i in xrange(len(pred)):
	print pred[i]


cluster0 = []
cluster1 = []
cluster2 = []

gmm.means_ = np.matrix(gmm.means_)
means01 = gmm.means_[:,0:2]
means23 = gmm.means_[:,2:4]
means34 = gmm.means_[:,3:]

#print means01
#print means23
#print means34


def data_(means_c):
	min_ = np.argmin(means_c,axis = 0)
	max_ = np.argmax(means_c,axis = 0)
	
	min_ = min_[0,0]
	max_ = max_[0,0]

	#print min_
	#print max_
	

	for j in xrange(3):
		if (j != min_ and j != max_):
			med_ = j
			break

	#print '******'	
	#print min_
	#print med_
	#print max_

	for i in xrange(len(pred)):
		if  pred[i] == min_:
			cluster0.append(int(i))
		if pred[i] == med_:
			cluster1.append(int(i))
		if pred[i] == max_:
			cluster2.append(int(i))

	data_cluster0 = data[cluster0,:]
	data_cluster1 = data[cluster1,:]
	data_cluster2 = data[cluster2,:]
	return data_cluster0,data_cluster1,data_cluster2


data_cluster0, data_cluster1,data_cluster2 = data_(means01)


print 'means'
print gmm.means_
#print 'covarianves'
#print gmm.covariances_



fig = plt.figure(1)
plt.subplot(311)
plt.plot(data_cluster0[:,1],data_cluster0[:,0],'bo')
plt.plot(data_cluster1[:,1],data_cluster1[:,0],'go')
plt.plot(data_cluster2[:,1],data_cluster2[:,0],'ro')
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gaussian data x1 x2')
#plt.draw()
#plt.show()


data_cluster0, data_cluster1,data_cluster2 = data_(means23)


#fig2 = plt.figure(2)
plt.subplot(312)
plt.plot(data_cluster0[:,3],data_cluster0[:,2],'bo')
plt.plot(data_cluster1[:,3],data_cluster1[:,2],'go')
plt.plot(data_cluster2[:,3],data_cluster2[:,2],'ro')
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.xlabel('x3')
plt.ylabel('x4')
plt.title('Gaussian data x3 x4')
#plt.draw()
#plt.show()

data_cluster0, data_cluster1,data_cluster2 = data_(means34)


#fig3 = plt.figure(3)
plt.subplot(313)
plt.plot(data_cluster0[:,4],data_cluster0[:,3],'bo')
plt.plot(data_cluster1[:,4],data_cluster1[:,3],'go')
plt.plot(data_cluster2[:,4],data_cluster2[:,3],'ro')
plt.xlim((-10,10))
plt.ylim((-10,10))

plt.xlabel('x4')
plt.ylabel('x5')
plt.title('Gaussian data x5 x4')
#plt.draw()
plt.show()





##############################################
covariances = np.array(gmm.covariances_)
weights = np.array(gmm.weights_)

print covariances
print weights
py = weights
P = np.zeros([3000,3])
u = np.zeros([3,5])
u_old = np.zeros([3,5])
#initialize u
u[0,:]=data[10,:]
u[1,:]=data[100,:]
u[2,:]=data[500,:]
thresh = [0,0,0]
threshold = 0.001
u_old[:,:] = u[:,:]

print 'initializing u_old'
print u_old
print 'intial u'
print u
count = 0
while(1):
	for i in xrange(3):
		for j in xrange(3000):
			tmp = np.dot((data[j,:] - u[i,:]),inv(-2*covariances[i]*np.eye(5)))
			ans = np.dot(tmp,np.transpose(data[j,:] - u[i,:]))
			P[j,i] = np.exp(ans)*py[i]


	
	for i in xrange(3):
		summ_n = np.array([0,0,0,0,0],dtype = np.float)
		summ_d = 0
		for j in xrange(3000):
			summ_n = summ_n + P[j,i]*data[j,:]
			summ_d = summ_d + P[j,i]
			#print summ_n

		print i
		print 'summ_n nad summ_d'
		print (summ_n)
		print float(summ_d)
		print "%%%%%%%%%%%%"
		print u[i,:]
		u[i,:] = (summ_n)*(1/float(summ_d))
		print u[i,:]
		
		print "%%%%%%%%%%%"
		print '**'
		print u[i,:]
		print u_old[i,:]
		print '****'
		thresh[i] = LA.norm(u[i,:] - u_old[i,:],2)
		print "###"
		print thresh	
	
	print 'count = ' + str(count) + '|' + str(thresh[0]) + " " + str(thresh[1]) + " " + str(thresh[2])
	u_old[:,:] = u[:,:]
	count = count + 1
	if ((thresh[0] < threshold) and (thresh[1] < threshold) and (thresh[2] < threshold)):
		print "CONVERGED!!!!!"
		break


print u
print gmm.means_
print gmm.covariances_
