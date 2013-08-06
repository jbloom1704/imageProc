''' Deep Hathi, Abhishek Mishra
    CSE 415, Autumn 2012
    Final Project

	The final client script that calls the pre-processing functions,
	segments the image and trains the individual features.

	'''


from preprocess import *
from backProp import *
import math
WIDTH = 200
HEIGHT = 200

wn = pmOpenImage(0,"ChessPieces.jpg")
grayscale(wn)
wn,edgest = canny_edge(wn)
wnarray = image_segment(wn)

heightList = [("ki",-1), ("q",-1),("b",-1), ("kn",-1), ("r",-1), ("p",-1)]

def findHeight(win):
	yList = []
	for x in range(pmGetImageWidth(win)):
		for y in range(pmGetImageHeight(win)):
			RGB = pmGetPixel(win, x, y)
			if RGB[0]!=0:
				yList.append(y)
	yList = set(yList)			
	#yList.sort()
	yList = sorted(yList)
	return yList[len(yList) - 1] - yList[0]



print len(wnarray)
for i in range(len(heightList)):
	piece, height = heightList[i]
	height = findHeight(wnarray[i])
	heightList[i] = (piece, height)
	print piece+" "+str(height)
	
	
	
pattern = [
			[[heightList[0][1]], [1]],	#king
			[[heightList[1][1]], [0]],	#queen
			[[heightList[2][1]], [1]],	#bishop
			[[heightList[3][1]], [0]],	#knight
			[[heightList[4][1]], [1]],	#rook	
			[[heightList[5][1]], [0]],	#pawn
		]	
		
		
createAndTrain(1,10,1,pattern, -0.2, 0.2, -2.0, 2.0, 1000,0.5)		
		
		
		



	

		
