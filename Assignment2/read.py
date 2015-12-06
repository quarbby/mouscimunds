from __future__ import division
import scipy.io
from scipy.io import wavfile
import os
import math
from math import sqrt
import numpy

def RMS(list):
	return numpy.sqrt(numpy.mean(numpy.square(list)))

def PAR(list, rms):
	#return max([abs(i) for i in list])/rms
        return max(numpy.absolute(list))/rms 

def ZCR(list1,n):
	list2=list(list1)
	list1.pop()
	list2.pop(0)
	prod = [1 if (a*b<0) else 0 for a,b in zip(list1,list2)]
	prodsum = sum(prod)
	return prodsum/n
        '''
        list1 = list(list1)
        zcr_sum = len(scipy.where(list1[:-1]*list1[1:]<0)[0])
        return zcr_sum*1.0/(len(list1)-1)
        '''

def zcr(list1):
    zcr_sum = len(scipy.where(list1[:-1]*list1[1:]<0)[0])
    return zcr_sum*1.0/(len(list1)-1)

def MAD(list,m):
	#return numpy.median([abs(i-m) for i in list])
        med = numpy.median(list)
        abs_val = numpy.abs(med-list)
        return numpy.median(abs_val)

fp = open("ground truth.txt")
fp1 = open("output.csv", "w")
while True:
	line = fp.readline()
	#print line
	if not line : break
	line = str(line).split('	')
	fp1.write(line[0])
	fp1.write(",")
	test = scipy.io.wavfile.read(os.path.join(os.getcwd(), line[0]))
	arr = test[1]

	n=len(arr)
	#arr = [x/32768 for x in arr]
        arr = arr/32768.0

	rms = RMS(arr)
	fp1.write(str('{:.6f}'.format(round(rms,6))))
	fp1.write(",")


	fp1.write(str('{:.6f}'.format(round(PAR(arr,rms),6))))
	fp1.write(",")

	#fp1.write(str('{:.6f}'.format(round(ZCR(arr,n-1),6))))
	fp1.write(str('{:.6f}'.format(round(zcr(arr),6))))
	fp1.write(",")

	fp1.write(str('{:.6f}'.format(round(MAD(arr,numpy.median(arr)),6))))
	fp1.write("\n")


