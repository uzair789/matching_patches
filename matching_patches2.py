#!/usr/bin/python


"""
This script reads two images and finds matching patches using the depth map of the 
first image and the camera orientations of both the images.
"""

import argparse
import sys
import os
from PIL import Image
import numpy
import random
"""
#ROS default
focalLength_X = 525
focalLength_Y = 525
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0
"""
"""
#freiburg1
focalLength_X = 517.3
focalLength_Y = 516.5
centerX = 318.6
centerY = 255.3
scalingFactor = 5000.0
"""

#freiburg2
focalLength_X = 520.9
focalLength_Y = 521.0
centerX = 325.1
centerY = 249.7
scalingFactor = 5000.0


""" freiburg 3
focalLength_X = 535.4
focalLength_Y = 539.2
centerX = 320.1
centerY = 247.6
scalingFactor = 5000.0
"""

def generate_XYZcam(rgb,depth,patch_location):#,ply_file):
    """
    Generate a 3D point cloud in camera coordinate frane from a color and a depth image.
    
    Input:
    rgb_patch -- patch of color image
    depth_patch -- patch of depth image
    patch_location -- tuple (left,upper,right,lower)
    
    Output:
    dictionary : points[(u,v)] = [X,Y,Z] at each v,u location	
    
    """
   
    XYZ_cam = {}
    #coordinates of the patch
    left_limit = patch_location[0] 
    right_limit = patch_location[2]
    upper_limit = patch_location[1]
    lower_limit = patch_location[3]
    height = (lower_limit - upper_limit) #including the starting element
    width = (right_limit-left_limit) #including the starting element
    print ("Total selected pixels = %d"%(height*width))

    XYZ_cam_mat = numpy.zeros([4,width*height]) 
    print "XYZ_cam1_mat.shape"	
    print XYZ_cam_mat.shape
    count = 0   
    v = upper_limit
    
    while(v<lower_limit):
        u = left_limit
        while(u<right_limit):
            #color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
	    #print "ZZZZZZZZZZZZZZZZ"
	    #print Z		
            if Z==0:
                u=u+1
                continue
            X = (u - centerX) * Z / focalLength_X
            Y = (v - centerY) * Z / focalLength_Y
            XYZ_cam[(u,v)] = [X,Y,Z]         
            #print(("%f %f %f\n"%(X,Y,Z)))
            XYZ_cam_mat[:,count]=[X,Y,Z,1]
            #print (u,v)
            u=u+1#(width-1)
            count = count+1
        v=v+1#(height-1)
    print "length"
    print len(XYZ_cam)
    print "count"
    print count	
    return XYZ_cam_mat[:,0:count] # ignoring the zeros in the matrix because matsize is width*height


def generate_Transform4x4(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    _EPS = numpy.finfo(float).eps * 4.0
    #print ("EPS = %f"%_EPS)
    t = l[1:4]
    q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
	  

def find_quaternion(rgb_name):
    """
    Find the quaterion for the corresponding rgb image.

    Input :
	rgb_name -- name of the rgb image (format : rgb/'timestamp'.png)	
    Output : 
	l -- quaternion with timestamp (format : timestamp tx ty tz qx qy qz qw)	

    """

    f1 = open(generated_data_path+'rgb_groundtruth.txt')
    for line in f1:
	a=line.split(" ")     
	if a[1] == rgb_name:
		l = a[2:10]
                return l
                
    return None	



def find_depthMap(rgb_name):
    """
    Find the depthmap for the corresponding rgb image.

    Input :
	rgb_name -- name of the rgb image (format : rgb/'timestamp'.png  (string) )	
    Output : 
	depMap -- depth Map (format : depth/'timestamp'.png (string)  )	
    """

    f1 = open(generated_data_path+'rgb_depth.txt')
    for line in f1:
	a=line.split(" ")
	if a[1] == rgb_name:
		depMap = a[3]
		a=depMap.split("\n") #theres a \n in the end causing problems. Hence removing it.
		return a[0]		
                
    return None
    	
def norm_check(image1,image2):
	"""
	Computes a check on the ksi norm between the images

	G_w2 = G_w1 * G_12
	inv(G_w1) * G_w2 = G_12
	ksi = log(G_12)
	norm(ksi) --> gives the amount of change between G_w1 and G_w2

	Input :
		image1 -- name of reference image (format : rgb/'timestamp'.png  (string))
		image2 -- name of current image (format : rgb/'timestamp'.png  (string)) 

	Output : boolean (True of False)
		True if (norm > threshold)
	"""
	thresh = 0.02

	l1 = find_quaternion(image1)
	l2 = find_quaternion(image2)

	G_w1 = generate_Transform4x4(l1)
	G_w2 = generate_Transform4x4(l2)

	inv_G_w1 = numpy.linalg.inv(G_w1)

	G_12 = numpy.dot(inv_G_w1 , G_w2)

	print 'G_w1'
	print G_w1
	print 'G_w2'
	print G_w2
	print 'inv_G_w1'
	print inv_G_w1
	print 'G_12'
	print G_12

	ksi = numpy.log(G_12)

	print "generated ksi"
	print ksi
	#downhat ksi

	nrm = numpy.linalg.norm(ksi,2)

	print "thresh"
	print thresh
	print "norm of ksi"
	print nrm	

	if nrm > thresh:
		print "Transforamtion too large between frames!!!!"
		return True
	else:
		return False



def generateXYZ_world1(file_rgb1,start_X,start_Y):

    b1 = file_rgb1.split("/")   
    rgb_name1 = b1[5]+"/"+b1[6]
    print rgb_name1    
    im_rgb1 = Image.open(file_rgb1).convert('L')
    im_rgb_image1 = im_rgb1.load()
    
    depImage = find_depthMap(rgb_name1)    
    if depImage == None:
	print "Depth Image not found"
	return	
    
    path = dataset_path + dataset_sequence + depImage
    im_depth = Image.open(path)
    
    #create patches in the rgb image and also select the corresponding patch in the depth image
    left = start_X#410
    upper = start_Y#275
    right = left+patchSize_X#149
    lower = upper+patchSize_Y#369#325
    patch_location = (left,upper,right,lower)
    
    #visualizing the selected pixels in image1
    i=left
    while(i<right):
	j=upper
	while(j<lower):
                #if(i==left or i == right or j == upper or j == lower):
		 #  im_rgb_image1[i,j] = (255,0,0)
 		j=j+1
        i=i+1
    #im_rgb1.show()
    patch_rgb = im_rgb1.crop(patch_location)
    #print "dubugggggggggggg"
    #print patch_rgb.size		
    #patch_rgb.show()
    #print im_rgb1
    #print im_depth
    #print patch_location
    
    #generate the XYZ cordinates for the Image1
    XYZ_cam1 = generate_XYZcam(im_rgb1,im_depth,patch_location)
    print "XYZ_cam1.shape"
    print XYZ_cam1.shape
    if XYZ_cam1.shape[1] < 3:
	return	

    #Compute the 4x4 transform for the rgb image
	#--- first find the quaternion for the images (l represents the quaternion)
    l1 = find_quaternion(rgb_name1)
    
    if l1 == None:
	print "quaternion for l1 or l2 not found"
	return 

    #generate the 4x4 transform matrix for image1
    g1 = generate_Transform4x4(l1)
    g_inv1=numpy.linalg.inv(g1)


    #Transform XYZ_cam1 to XYZ_world1
    XYZ_world1 = numpy.dot(g1,XYZ_cam1)
    print "XYZ_world1.shape"
    print XYZ_world1.shape
    return XYZ_world1 



def generateXYZ_cam2(file_rgb2,XYZ_world1):

    b2 = file_rgb2.split("/")   
    rgb_name2 = b2[5]+"/"+b2[6]    
    im_rgb2 = Image.open(file_rgb2).convert('L')
    im_rgb_image2 = im_rgb2.load()
    l2 = find_quaternion(rgb_name2)
    
    if l2==None:
	print "quaternion for image2 not found"
	return 0	    
    
    #generate the 4x4 transform matrix for image2
    g2 = generate_Transform4x4(l2)
    g_inv2=numpy.linalg.inv(g2)

    #Map the XYZ_world1 to XYZ_im2
    K = numpy.zeros([3,3])
    K[0,0]=focalLength_X
    K[0,2]=centerX
    K[1,1]=focalLength_Y
    K[1,2]=centerY
    K[2,2]=1	
    Rt = g_inv2[0:3,:]

    #XYZ_im2 = K * [R t] * [X;Y;Z;1]
    #XYZ_im2 = [x y z]
    XYZ_cam2 = numpy.dot(Rt,XYZ_world1)
    print 'XYZ_cam2.shape'
    print XYZ_cam2.shape
    XYZ_im2 = numpy.dot(K,XYZ_cam2)
    
    print "XYZ_im2.shape"
    print XYZ_im2.shape
    #switching from homogenous to cartesian cordinates
    #x = x/z ; y = y/z  
    xyz_c = numpy.zeros([2,XYZ_im2.shape[1]])	
    for i in xrange(XYZ_im2.shape[1]):
	[xx,yy,zz] = XYZ_im2[:,i]
	x_c=xx/zz
	y_c=yy/zz 
        #xyz_c[:,i]=[(x_c),(y_c)]
        if (x_c > 640 or y_c > 480 or x_c < 0 or y_c < 0):
	        print 'corresponding patch is out of bounds'
		return 0
  	else:
		xyz_c[:,i] = [x_c,y_c]

    left = xyz_c[0,0]
    upper = xyz_c[1,0]
    right = xyz_c[0,XYZ_im2.shape[1]-1]
    lower = xyz_c[1,XYZ_im2.shape[1]-1]


    #print "*********"
    #print left
    #print right
    #print upper
    #print lower

    # doing this to maintain a constant crop size because usually some locations are absent
    mph = int((left+right)/2)
    mpv = int((upper+lower)/2)
    
    if(patchSize_X % 2 == 0):
	left = mph- (patchSize_X/2 ) #49
        right = (mph) + (patchSize_X/2  ) #49
    else:
 	left = mph - int(patchSize_X/2)	
	right = mph + int(patchSize_X/2) +1

    if(patchSize_Y % 2 == 0):
	upper = mpv - (patchSize_Y/2 )#49
        lower = (mpv) + (patchSize_Y/2  )#49
    else:
	upper = mpv - int(patchSize_Y/2)#49
        lower = mpv + int(patchSize_Y/2) + 1#49
    
    print 'size crop2'
    size = (right-left)*(lower-upper)
    print size
    match_location = (int(left),int(upper),int(right),int(lower))
    print "match location"
    print match_location

    #keeping the bounding box within the image dimensions
    if (match_location[0]<0 or match_location[2] > 640 or match_location[1] < 0 or match_location[3] > 480):
	print "matching image crop after bounding box fix is out of image!"
	return 0		


    #im_rgb2.show()
    patch_rgb2 = im_rgb2.crop(match_location)


    print "real size crop2"
    print patch_rgb2.size

    #patch_rgb2.show()
    path = generated_data_path + str(folder)+'/'+ str(folder) + b2[6]
    patch_rgb2.save(path,'PNG')
    f1.write('/mnt/data/uzair/'+path+" "+str(folder)+"\n")
    return 1





#if __name__ == '__main__':
def matching_patches(dataset_sequence_ip,generated_data_path_ip,num_folders,randomP = True):
    #parser = argparse.ArgumentParser(description = ' This script generates the synthetic matching and non-matching pairs  ')
    #parser.add_argument('-rP', type = bool, help='if set as :True selects random patches; ',default =  True)
    #args = parser.parse_args()
    #randomP = args.rP
    print randomP
    #if (randomP == True):
    #	num_folders = input("please input the number of folders : example 10,20...etc : ")
    #	num_folders = int(num_folders)

    global dataset_path
    global generated_data_path
    global dataset_sequence

    generated_data_path = generated_data_path_ip#'dummy_test/'#'generated_dataset100x100_slidingWindow/'	
    dataset_path = '/mnt/data/tum_rgbd_slam/'
    dataset_sequence = dataset_sequence_ip#'rgbd_dataset_freiburg2_rpy/'	
   	
   

    #generate a random image
    f2 = open(dataset_path+dataset_sequence+'/rgb.txt','r')
    imageList1 = list(f2)
    noLines = len(imageList1)		
    
    global patchSize_X
    global patchSize_Y
    patchSize_X = 100
    patchSize_Y = 100
    #num_folders = 40	
    off_x = 0
    off_y = 0	     
    global folder
    folder = 1
    os.mkdir(generated_data_path+str(folder))
	
    global f1
    f1 = open(generated_data_path+'dataset.txt','w')	
    master_loop = True	

    #for i in xrange(num_folders):
    while(master_loop):	
	#generate a random image
    		
    	idx = numpy.random.random_integers(3,noLines-1)# avoiding the top three lines with '#"
	print idx
    	a1 = imageList1[idx].split(" ")
	
	image1 = a1[1].rstrip()
        print '***'
	print image1	

    	file_rgb1 = dataset_path + dataset_sequence + image1 #'rgb/1311867718.641710.png' 
    	im_rgb1_vis = Image.open(file_rgb1)
    	im_rgb_vis_image1 = im_rgb1_vis.load()



	#os.mkdir(generated_data_path+str(folder))
	
    	start_X = off_x 
    	start_Y = off_y

	#visualizing the crop on image1
	left = start_X#410
        upper = start_Y#275
        right = left+patchSize_X#149
        lower = upper+patchSize_Y#369#325
	ii=left
	while(ii<right):
		jj=upper
		while(jj<lower):
		   im_rgb_vis_image1[ii,jj] = (255-2*folder,2*folder,folder)
		   #print('ii = %d, jj = %d , right = %d , lower = %d' %(ii,jj,right,lower))	
 		   jj=jj+1
                ii=ii+1
      	
	#if (start_X >= 0 and start_X <= 640-patchSize_X and start_Y >= 0 and start_Y <= 480-patchSize_Y):
        	#pass
	#else:
		#print ("Current starting point (x,y) : %d %d "%(startX,startY))
		#print 'change the starting location or patchSize - Pixels are out of bounds'
		#sys.exit()
    	
	#print "000000000"
	#print file_rgb1
	#print start_X
	#print start_Y

	XYZ_world1 = generateXYZ_world1(file_rgb1,start_X,start_Y)
	if (XYZ_world1 == None): #because of Z being 0
		off_x = random.randrange(0,640-patchSize_X+1)
		off_y = random.randrange(0,480-patchSize_Y+1)
		continue	
		
    	#file_rgb2 = 'rgbd_dataset_freiburg2_rpy/rgb/1311867733.645698.png'
    	count = 0
    	f = open(dataset_path + dataset_sequence + 'rgb.txt')
	summ = 0
    	for line in f:
		if line[0] == '#':
			continue
        	count = count + 1
		#if count == 20:
		#	break

		print('processing folder % d | count %d | start (%d,%d)'%(folder,count,start_X,start_Y))
		a = line.split(" ")
		b=a[1].split("\n")
		file_rgb2 = dataset_path + dataset_sequence + b[0] 	

		##norm check to check if the transformation between the images is greater than a threshold
		#if (norm_check(image1,b[0]) ):
		#	break
		
		#print file_rgb2	
		flag = generateXYZ_cam2(file_rgb2,XYZ_world1)
		summ = summ + flag
				
	if (summ >  0):# to prevent empty folders
		folder = folder + 1
		os.mkdir(generated_data_path+str(folder))
	
	

	if (randomP == True):
		off_x = random.randrange(0,640-patchSize_X+1)   #-1 and +1 to include the lower and upper bound...range(lower,upper) doesn include lower nad upper
    		off_y = random.randrange(0,480-patchSize_Y+1)
		if (folder == num_folders + 1):
			master_loop = False

	#performing full sliding window  
	if (randomP == False):
		off_x = off_x + patchSize_X-10#random.randrange(-50-1,640-patchSize_X-50+1)   #-1 and +1 to include the lower and upper bound...range(lower,upper) doesn include lower nad upper
    		if off_x > 640-patchSize_X:
			off_x = 0
			off_y = off_y + patchSize_Y-10#random.randrange(-270-1,480-patchSize_Y-270+1)
			if off_y > 480-patchSize_Y:
				print('whole image covered!!')
				master_loop = False
	 		
    	
    #im_rgb1_vis.show()
    im_rgb1_vis.save(generated_data_path+"patches_all.png",'PNG')
    os.rmdir(generated_data_path+str(folder))#removing the last empty folder	

	
    
    











    
