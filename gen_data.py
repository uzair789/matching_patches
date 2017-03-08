#!/usr/bin/python
import os

path = 'generated_dataset/'

f1 = open(path+'dataset.txt','r')
f3 = open(path+'matching_pairs.txt','w')
f4 = open(path+'nonmatching_pairs.txt','w')

"""
seperating the matching and the non-matching pairs
 - input : dataset.txt
 - outputs : mathing_pairs_shuf.txt and nonmatching_pairs_shuf.txt
            (The txt files are shuffled)
"""
count = 0
linesMatching = 0
linesNonmatching = 0
i = 1
for line in f1:
	a=line.split(" ")
	a[1] = a[1].rstrip()
	f2 = open(path+'dataset.txt','r')
	j = 1
	for line2 in f2:
		if i > j:
			j = j+1
		else:
			count = count + 1
			print count
			b = line2.split(" ")
			b[1] = b[1].rstrip()
			if a[1] == b[1]:
				f3.write(a[0]+" "+b[0]+" "+str(1)+"\n")
				linesMatching = linesMatching + 1			
			else:
				f4.write(a[0]+" "+b[0]+" "+str(0)+"\n")
				linesNonmatching = linesNonmatching + 1
				j = j+1
	i = i+1                     

f1.close()
f2.close()
f3.close()
f4.close()


""" 
  shuffling the matching_pairs.txt --> matching_pairs_shuf.txt
  shuffling the nonmatching_pairs.txt --> nonmatching_pairs_shuf.txt

"""
os.system('shuf '+path+'matching_pairs.txt -o '+path+'matching_pairs_shuf.txt')
os.system('shuf '+path+'nonmatching_pairs.txt -o '+path+'nonmatching_pairs_shuf.txt')

"""
  Merging the matching_pairs_shuf.txt and the nonmatching_pairs_shuf.txt into a txt file with alternate matching
  and non matching pairs.
  
   input : matching_pairs_shuf.txt and nonmatching_pairs_shuf.txt
   output : train1.txt, train2.txt, val1.txt, val2.txt
"""
f1 = open(path+'matching_pairs_shuf.txt','r')
f2 = open(path+'nonmatching_pairs_shuf.txt','r')
f3 = open(path+'train1.txt','w')
f4 = open(path+'train2.txt','w')
f5 = open(path+'val1.txt','w')
f6 = open(path+'val2.txt','w')
f7 = open(path+'test1.txt','w')
f8 = open(path+'test2.txt','w')
totalLines = linesMatching + linesNonmatching
print totalLines
trainData = int(.7 * totalLines)
valData = int(.2 * totalLines)
testData = int(.1 * totalLines)
testLowerBound = trainData + valData
testUpperBound = trainData + valData + testData
valLowerBound = trainData
valUpperBound = trainData + valData 




count = 0
for line in f1:
	a=line.split(" ")
	print a
	print count
	if(count > testLowerBound and count <= testUpperBound):
		#write test data
		f7.write(a[0]+" "+a[2])
		f8.write(a[1]+" "+a[2])

	if(count > valLowerBound and count <= valUpperBound):
		#write val data
		f5.write(a[0]+" "+a[2])
		f6.write(a[1]+" "+a[2])
	else:
		#write train data
		f3.write(a[0]+" "+a[2])
		f4.write(a[1]+" "+a[2])
		
	for line2 in f2:
		b = line2.split(" ")
		if(count > testLowerBound and count <= testUpperBound):
			#write test data
			f7.write(b[0]+" "+b[2])
			f8.write(b[1]+" "+b[2])

		if(count > valLowerBound and count <= valUpperBound):
			#write val data
			f5.write(b[0]+" "+b[2])
			f6.write(b[1]+" "+b[2])
		else:
			#write train data
			f3.write(b[0]+" "+b[2])
			f4.write(b[1]+" "+b[2])
	
		count = count + 2
		break

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()






		
