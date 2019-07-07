import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt

#eye cascading xml file
eye_cascade = cv2.CascadeClassifier("Train/third-party/frontalEyes35x16.xml")
#reading glassess image
glasses = cv2.imread("Train/glasses.png",cv2.IMREAD_UNCHANGED)
glasses = cv2.cvtColor(glasses,cv2.COLOR_BGR2RGBA)

#eye cascading xml file
nose_cascade = cv2.CascadeClassifier("Train/third-party/Nose18x15.xml")
#reading glassess image
mustache = cv2.imread("Train/mustache.png",cv2.IMREAD_UNCHANGED)
mustache = cv2.cvtColor(mustache,cv2.COLOR_BGR2RGBA)


#eye cascading xml file
face_cascade = cv2.CascadeClassifier("Train/haarcascade_frontalface_alt.xml")
#reading glassess image
chain = cv2.imread("Train/Gold-Chain.png",cv2.IMREAD_UNCHANGED)
#chain = cv2.cvtColor(chain,cv2.COLOR_BGR2RGBA)


mouth_cascade = cv2.CascadeClassifier("Train/Mouth25x15.1/Mouth.xml")
#reading cigar image
cigar = cv2.imread("Train/cigar.png",cv2.IMREAD_UNCHANGED)
#cigar = cv2.cvtColor(cigar,cv2.COLOR_BGR2RGBA)


def add_glasses(image):
	
	#reading glassess image
	
	eyes = eye_cascade.detectMultiScale(image,1.5,5)

	if(len(eyes)!=0):
		x = eyes[0][0]
		y = eyes[0][1]
		w = eyes[0][2]
		h = eyes[0][3]
	else:
		return image

	#resizing glasses
	new_glasses = cv2.resize(glasses ,dsize = (w,h))

	alpha = new_glasses[:,:,3]/255

	(X,Y) = alpha.shape
	#adding glasses
	for i in range(X):
		for j in range(Y):
			if(alpha[i,j]):
				image[y+i,x+j,:3] = new_glasses[i,j,:3]

	return image

def add_mustache(image):
	
	noses = nose_cascade.detectMultiScale(image,4,5)
	
	if(len(noses)!=0):
		x1 = noses[0][0]
		y1 = noses[0][1]
		w1 = noses[0][2]
		h1 = noses[0][3]
	else:	
		return image

	new_mustache = cv2.resize(mustache ,dsize = (w1,(h1//2)))

	alpha1 = new_mustache[:,:,3]/255

	(X1,Y1) = alpha1.shape

	for j in range(Y1):
		for i in range(X1):
			if(alpha1[i,j]):
				image[y1+i+(2*h1//4),x1+j,:3] = new_mustache[i,j,:3]

	return image

def add_chain(image):
	
	faces = face_cascade.detectMultiScale(image,1.3,5)
	if(len(faces)!=0):
		x2 = faces[0][0]
		y2 = faces[0][1]
		w2 = faces[0][2]
		h2 = faces[0][3]
	else:	
		return image

	new_chain = cv2.resize(chain,dsize=(w2,h2))

	alpha2 = new_chain[:,:,3]/255

	(X2,Y2) = alpha2.shape
	l =	image.shape[0]
	b = image.shape[1]

	for i in range(X2):
		for j in range(Y2):
			if(alpha2[j,i]):
				if(y2+h2+j<l and x2+i<b):
					image[y2+h2+j,x2+i,:3] = new_chain[j,i,:3]
	return image

def add_cigar(image):

	mouths = mouth_cascade.detectMultiScale(image,5,11)

	if(len(mouths)!=0):
		x = mouths[0][0]
		y = mouths[0][1]
		w = mouths[0][2]
		h = mouths[0][3]
		print(x,y,w,h)
	else:	
		return image


	new_cigar = cv2.resize(cigar,dsize=(w//2,h))
	alpha3 = new_cigar[:,:,3]/255
	(Y3,X3) = alpha3.shape
	l = image.shape[0]
	b = image.shape[1]

	for i in range(X3):
		for j in range(Y3):
			if(alpha3[j,i]):
				if(y+j<l and x+i<b):
					image[y+j,x+i,:3] = new_cigar[j,i,:3]

	return image

cam = cv2.VideoCapture(0)

while True:
	ret,frame = cam.read()
	if ret==False:
		print("Something Went Wrong!")
		continue

	key_pressed = cv2.waitKey(20) & 0xFF #Bitmasking to get last 8 bits
	if key_pressed==ord('q'): #ord-->ASCII Value(8 bit)
		break
	frame = add_glasses(frame)
	frame = add_mustache(frame)
	frame = add_chain(frame)
	frame = add_cigar(frame)
	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow("Video",frame)
	
cam.release()
cv2.destroyAllWindows()	
