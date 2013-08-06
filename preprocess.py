''' Deep Hathi, Abhishek Mishra
    CSE 415, Autumn 2012
    Final Project

    Image Pre-processing functions for feature extraction'''

from JythonPixelMathInterface import *
import math

WIDTH = 200
HEIGHT = 200

def imscale(wn):
   # scales individual window to consistent WIDTH-by-HEIGHT window
   wn2 = pmNewImage(0,"Scaled",WIDTH,HEIGHT,255,255,255)
   scaleW = 1.0*pmGetImageWidth(wn)/WIDTH
   scaleH = 1.0*pmGetImageHeight(wn)/HEIGHT
   print str(scaleW)+","+str(scaleH)
   for x in range(WIDTH):
      for y in range(HEIGHT):
         px = math.floor(x*scaleW)
         py = math.floor(y*scaleH)
         r,g,b = pmGetPixel(wn,int(px),int(py))
         pmSetPixel(wn2,x,y,r,g,b)
   return wn2

def grayscale(source):
   for x in range(pmGetImageWidth(source)):
      for y in range(pmGetImageHeight(source)):
         r,g,b = pmGetPixel(source,x,y)
         grayform = int(0.2125 * r) + int(0.7154 * g) + int(0.0721 * b)
         pmSetPixel(source,x,y,grayform,grayform,grayform)
         
def threshold(wn, theta):
   # Use Blue component for the threshold; constructs binary image
   pmSetSource1(wn)
   pmSetDestination(wn)
   pmSetFormula("If Blue1(x,y) > " + str(theta) + " then 0 else RGB(255,255,255)")
   pmCompute()

def erode(source, dest):
   pmSetSource1(source)
   pmSetDestination(dest)
   pmSetFormula("If {Blue1(x,y)=255} and " +\
                "{(Blue1(x-1,y)*Blue1(x+1,y)*Blue1(x,y-1)*Blue1(x,y+1))=0} " +\
                "then 0 else S1(x,y)")
   pmCompute()

def dilate(source, dest):
   pmSetSource1(source)
   pmSetDestination(dest)
   pmSetFormula("If {Blue1(x,y)=0} and " +\
                " {(Blue1(x-1,y)+Blue1(x+1,y)+Blue1(x,y-1)+Blue1(x,y+1))>100} " +\
                "then RGB(1,0,255) else RGB(0,0,Blue1(x,y)")
   pmCompute()

def canny_edge(wn):
   # uses a Canny edge detector, consisting of image smoothing, gradient
   # calculation, non-maximal suppression, and hysterisis
   # Should be more accurate than Sobel edge detector
   width = pmGetImageWidth(wn)
   height = pmGetImageHeight(wn)

   # apply Gaussian filtering
   # gaussianfilter(wn)

   # apply Sobel operator
   wn2, mag_ang = edges(wn,True)
   wn3 = pmNewImage(0,"Final edge",width,height,0,0,0)
   edgeSet = []
   
   # non-maximal suppression
   for x in range(2,width-2):
      for y in range(2,height-2):
         mag, angle = mag_ang[(x,y)]
         if (angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5):
            # 45 degrees (NW, SE)
            mag_ang[(x,y)] = [mag,45]
            magNW,a = mag_ang[(x+1,y+1)]
            magSE,b = mag_ang[(x-1,y-1)]
            if mag > magNW and mag > magSE: edgeSet.append((x,y))
         elif (angle > 67.5 and angle < 112.5) or (angle >= 247.5 and angle < 292.5):
            # 90 degrees (E,W)
            mag_ang[(x,y)] = [mag,90]
            magE,a = mag_ang[(x-1,y)]
            magW,b = mag_ang[(x+1,y)]
            if mag > magE and mag > magW: edgeSet.append((x,y))
         elif (angle >= 112.5 and angle < 157.5) or (angle >= 292.5 and angle < 337.5):
            # 135 degrees (NE, SW)
            mag_ang[(x,y)] = [mag,135]
            magNE, a = mag_ang[(x-1,y+1)]
            magSW, b = mag_ang[(x+1,y-1)]
            if mag > magNE and mag > magSW: edgeSet.append((x,y))
         else:
            # 0 degrees (N, S)
            mag_ang[(x,y)] = [mag,0]
            magN, a = mag_ang[(x,y+1)]
            magS, b = mag_ang[(x,y-1)]
            if mag > magN and mag > magS: edgeSet.append((x,y))

   for point in edgeSet:
      mag,ang = mag_ang[point]
      gray = int(0.2125 * mag) + int(0.7154 * mag) + int(0.0721 * mag)
      pmSetPixel(wn3,point[0],point[1],gray,gray,gray)

   # hysteresis
   wn4 = pmNewImage(0,"After hysteresis",width,height,0,0,0)
   output = hysteresis(edgeSet, 120, 180, mag_ang)
   for point in output:
      pmSetPixel(wn4, point[0],point[1],255,255,255)
   
   return wn4, edgeSet

def gaussianfilter(wn, sf=1):
   # create two masks to simulate matlab meshgrid function
   # use default sigma = 1 for now, so mask is 5x5
   # using scale factor with rounding to use integers values for convolution
   x_mask = []
   y_mask = []
   normScale = 0
   # for x-mask, the middle row is 0
   for x in range(0,5):
      temp = []
      for y in range(0,5):
         temp.append(sf*gaussian(abs(x-2),abs(y-2)))
      x_mask.append(temp)
      normScale += sum(temp)*765
   # construct y-mask as the transposition of x-mask
   y_mask = []
   for y in range(0,5):
      temp = []
      for x in range(0,5):
         temp.append(x_mask[x][y])
      y_mask.append(temp)

   # now apply masks in x, y directions by calculating the gradient magnitudes
   width = pmGetImageWidth(wn)
   height = pmGetImageHeight(wn)
   for x in range(2,width-2):
      for y in range(2,height-2):
         Gx = 0
         Gy = 0
         for mx in range(0,4):
            for my in range(0,4):
               r,g,b = pmGetPixel(wn,x-2+mx,y-2+my)
               Gx += (r+g+b)*x_mask[mx][my]
               Gy += (r+g+b)*y_mask[mx][my]
               
         # calculate magnitude and normalize
         mag = int(math.hypot(Gx,Gy)/(normScale)*255)
         gray = int(0.2125 * mag) + int(0.7154 * mag) + int(0.0721 * mag)
         pmSetPixel(wn,x,y,gray,gray,gray)
   return x_mask, y_mask

def gaussian(x,y, sigma=1):
   # calculate the guassian
   return 1/(2*math.pi)*math.exp(-1*(x**2+y**2)/(2.0*sigma**2))

def hysteresis(edgeSet, highT, lowT,mag_ang):
   # traverses points in the edge set and checks to see if they are valid
   # Valid if: magnitude of gradient >= high threshold
   #           low threshold <= magnitude < high threshold; recurse through
   #           neighboring edges to see if they are valid
   # Invalid if: magnitude < low threshold
   edgeSet = sorted(set(edgeSet))
   output = []
   visited = []
   for (x,y) in edgeSet:
      if (x,y) not in visited:
         visited.append((x,y))
         mag, ang = mag_ang[(x,y)]
         if mag >= lowT:
            # valid points for consideration
            if mag >= highT:
               # point is a valid edge
               output.append((x,y))
            else:
               # check pixels nearby to see if edge is valid
               output = follow_hysteresis((x,y),visited,output,highT,lowT,\
                                          mag_ang)
   return output

def follow_hysteresis(point, visited, output, highT, lowT, mag_ang):
   # recursive helper function for hysteresis
   magP, ang = mag_ang[point]
   x,y = point
   magSet = {}
   for mx in range(-1,2):
      for my in range(-1,2):
         if (x+mx,y+mx) not in visited:
            mag, ang = mag_ang[(x+mx,y+mx)]
            magSet[(x+mx,y+mx)] = mag

   for point in magSet:
      visited.append(point)
      if magSet[point] >= highT:
         # there is a valid edge next to (x,y) so (x,y) likely to be an edge
         # as well; point compared is also appended to the output set
         output.append((x,y))
         output.append(point)
         return output
      elif magSet[point] >= lowT:
         # recurse again
         return follow_hysteresis(point, visited, output,highT,lowT,mag_ang)
   return output
                    
def edges(wn, retMagAng=False):
   # use the Sobel edge detector
   mag_ang = {} # store magnitude, angle for each point (x,y)
   width = pmGetImageWidth(wn)
   height = pmGetImageHeight(wn)
   wn_edge = pmNewImage(0,"Edge Space", width, height, 0,0,0)
   xkernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
   ykernel = [[-1,-2,-1],[0,0,0],[1,2,1]]
   for x in range(1,width-1):
      for y in range(1,height-1):
         Gx = 0
         Gy = 0
         for grad in range(-1,2):
            RGB = [pmGetPixel(wn,x+grad,y), pmGetPixel(wn,x,y+grad),\
                   pmGetPixel(wn,x+grad,y+grad)]
            Gx += xkernel[grad+1][0]*sum(RGB[0]) +\
                  xkernel[0][grad+1]*sum(RGB[1]) +\
                  xkernel[grad+1][grad+1]*sum(RGB[2])
            Gy += ykernel[grad+1][0]*sum(RGB[0]) +\
                  ykernel[0][grad+1]*sum(RGB[1]) +\
                  ykernel[grad+1][grad+1]*sum(RGB[2])
         length = int(math.hypot(Gx,Gy)/4328*255)
         angle = math.atan2(Gy,Gx)*180/math.pi+180
         mag_ang[(x,y)] = (length,angle)
         r,g,b = pmGetPixel(wn,x,y)
         if length < r-3: length = 255   
         else:
            mag_ang[(x,y)] = [length - 80,angle]
            length = 0
         pmSetPixel(wn_edge, x, y, length, length, length)
   # return optional magnitude, angle map if retMagAng = True
   if retMagAng: return wn_edge, mag_ang
   return wn_edge

def image_segment(wn):
   # Given a binary image, return an array of window handles, where
   # each window contains one object from overall image
   wnarray = []
   edgeSet = []
   width = pmGetImageWidth(wn)
   height = pmGetImageHeight(wn)
   for x in range(width):
      for y in range(int(height/2)-5, int(height/2)+5):
         r,g,b = pmGetPixel(wn,x,y)
         if r != 0:
            edgeSet.append((x,y))
   edgeSet = sorted(set(edgeSet))

   # boundary sets
   left = []
   right = []
   onRight = False
   for (x,y) in edgeSet:
      if len(left) == 0:
         left.append((x,y))
         onRight = True
      else:
         (x1,y1) = left[-1]
         if x-x1 >= 20:
            if onRight:
               right.append((x,y))
               onRight = False
            else:
               (x1,y1) = right[-1]
               if x-x1 > 20:
                  left.append((x,y))
                  onRight = True

   # traverse left, right sets to find appropriate spots for segmentation
   count = 0
   start = 0
   for i in range(1,len(left)):
      if len(right) > i-1:
         (x,y) = left[i]
         (x1,y1) = right[i-1]
         dist = int((x - x1)/2)
         if dist > 0:
            print("Start: "+str(start)+"; End: "+str(min(x1+dist,width)))
            wnarray.append(imtranslate(wn,start,min(x1+dist,width)))
            start = x1+dist+1
   # get the last window handle (from start to width)
   if len(right) == len(left):
      wnarray.append(imtranslate(wn,start,width))
   return wnarray

def imtranslate(wn,start,end):
   # creates a separate window with a copy of the pixels from [start,end) from
   # the original window
   height = pmGetImageHeight(wn)
   scaleY = 1.0*HEIGHT/height
   wn1 = pmNewImage(0,"Separate pieces",end-start,HEIGHT,0,0,0)
   for x in range(start,end):
      for y in range(height):
         r,g,b = pmGetPixel(wn,x,y)
         pmSetPixel(wn1,x-start,int(y*scaleY),r,g,b)
   pmSetVisible(wn1,False)
   return wn1

def hough_transform(source, dThres=15, stepSize = 200, rhoSize = 200):
   # explore space with theta in [0,pi)
   dtheta = math.pi/stepSize
   width = pmGetImageWidth(source)
   height = pmGetImageHeight(source)
   temp = pmNewImage(0, "Hough Space", stepSize, rhoSize, 0,0,0)
   # calculate p for each x,y,theta
   rmax = math.hypot(stepSize, rhoSize)
   dr = rmax/(rhoSize/2)
   rholoc = {}
   maxRGB = 0
   accumulator = {}
   for x in range(width):
      for y in range(height):
         # check if the pixel should even be considered
         pixelRGB = pmGetPixel(source, x, y)
         if pixelRGB[0] != 0:
            for ntheta in range(stepSize):
               theta = dtheta*ntheta
               rho = x*math.cos(theta) + y*math.sin(theta)
               # identify location of rho, theta in hough space
               rho_i = rhoSize/2 + int(rho/dr + 0.5)
               rholoc[rho_i] = rho
               # update hough space
               pixelRGB = pmGetPixel(temp, ntheta, rho_i)
               if (ntheta, rho_i) not in accumulator:
                  accumulator[(ntheta, rho_i)] = 1
               else:
                  accumulator[(ntheta, rho_i)] += 1

   # show hough space
   for (ntheta, rho_i) in accumulator:
      val = accumulator[(ntheta, rho_i)]
      if val > 255: val = 255
      pmSetPixel(temp, ntheta, rho_i, val, val, val)
   # identify threshold
   maxThreshold = max(accumulator.values())
   if maxThreshold > 255: maxThreshold = 255
   # identify intersection points in hough space
   temp2 = pmCloneImage(temp)
   intersections = []
   threshold(temp2,maxThreshold - dThres)
   raw_intersect = peak_detect(temp,temp2,accumulator)
   for (ntheta, rho_i) in raw_intersect:
      intersections.append(((dtheta*ntheta),rholoc[rho_i]))
   
   # convert to slope, intercept form
   lines = []
   for (theta, rho) in intersections:
      if theta != 0.0:
         slope = round(-1.0/math.tan(theta),4)
         intercept = round(rho/math.sin(theta), 5)
         if (slope,intercept) not in lines:
            lines.append((slope,intercept))
   return accumulator, intersections, lines

def label_region(wn):
   # given a binary image, labels each pixel according to the region it is in
   label = 0
   point_label = {}
   for y in range(0,pmGetImageHeight(wn)):
      for x in range(0,pmGetImageWidth(wn)):
         thrPix = pmGetPixel(wn,x,y)
         if thrPix[0] != 0:
            point_label[(x,y)] = 0
         elif y == 0:
            if x == 0:
               label += 1
               point_label[(0,0)] = label
            else:
               thrPix0 = pmGetPixel(wn,x-1,0)
               if thrPix[0] == 0 and thrPix0[0] == 0:
                  # points are connected
                  point_label[(x,y)] = point_label[(x-1,y)]
               else:
                  label += 1
                  point_label[(x,y)] = label
         else:
            if x == 0:
               r,g,b = pmGetPixel(wn,x,y-1)
               if r == 0: point_label[(x,y)] = point_label[(x,y-1)]
               else:
                  label += 1
                  point_label[(x,y)] = label
            else:
               tu = pmGetPixel(wn,x,y-1)
               tl = pmGetPixel(wn,x-1,y)
               td = pmGetPixel(wn,x-1,y-1)
               # check all cases to assign labels
               if tu[0] != 0 and tl[0] != 0 and td[0] != 0:
                  label += 1
                  point_label[(x,y)] = label
               elif tu[0] == 0 and tl[0] != 0 and td[0] != 0:
                  point_label[(x,y)] = point_label[(x,y-1)]
               elif tl[0] == 0 and tu[0] != 0 and td[0] != 0:
                  point_label[(x,y)] = point_label[(x-1,y)]
               elif td[0] == 0 and tu[0] != 0 and tl[0] != 0:
                  point_label[(x,y)] = point_label[(x-1,y-1)]
               elif tu[0] == 0 and tl[0] == 0 and td[0] != 0:
                  lab2 = min(point_label[(x-1,y)], point_label[(x,y-1)])
                  point_label[(x-1,y)] = lab2
                  point_label[(x,y-1)] = lab2
                  point_label[(x,y)] = lab2
               elif tu[0] == 0 and tl[0] != 0 and td[0] == 0:
                  lab2 = min(point_label[(x-1,y-1)], point_label[(x,y-1)])
                  point_label[(x-1,y-1)] = lab2
                  point_label[(x,y-1)] = lab2
                  point_label[(x,y)] = lab2
               elif tu[0] != 0 and tl[0] == 0 and td[0] == 0:
                  lab2 = min(point_label[(x-1,y)], point_label[(x-1,y-1)])
                  point_label[(x-1,y)] = lab2
                  point_label[(x-1,y-1)] = lab2
                  point_label[(x,y)] = lab2
               elif tu[0] == 0 and tl[0] == 0 and td[0] == 0:
                  lab2 = min(point_label[(x-1,y)],\
                             point_label[(x-1,y-1)], point_label[(x,y-1)])
                  point_label[(x-1,y)] = lab2
                  point_label[(x,y-1)] = lab2
                  point_label[(x-1,y-1)] = lab2
                  point_label[(x,y)] = lab2

   # invert point_label so that each label points to a set of (x,y) coordinates
   label_points = {}
   for x in range(pmGetImageWidth(wn)):
      for y in range(pmGetImageHeight(wn)):
         if (x,y) in point_label:
             label = point_label[(x,y)]
             if label not in label_points: label_points[label] = [(x,y)]
             else: label_points[label].append((x,y))
   return label_points

def peak_detect(wn1, wn2, accumulator):
   # overlay thresholded binary image wn2 on top of wn1 to find similar regions
   # take maximum of accumulator in the regions for peak
   # the dimensions of wn1, wn2 have to be the same
   label_points = label_region(wn2)
   intersections = {}
      
   # for each label, find local maximum in wn1 using accumulator
   for label in label_points:
      maxVal = 0
      for point in label_points[label]:
         ntheta, rho_i = point
         if (ntheta, rho_i) in accumulator and\
            accumulator[(ntheta, rho_i)] > maxVal:
            maxVal = accumulator[(ntheta, rho_i)]
            intersections[label] = (ntheta, rho_i)
   return intersections.values()
