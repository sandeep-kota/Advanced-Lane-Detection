import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os



def main():
	# Input and output video paths
	out = cv2.VideoWriter('Data2.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1200,600))
	cap = cv2.VideoCapture('./Dataset/data_2/challenge_video.mp4')

	# HSV Color Threshold - Yellow 
	[Y_H_L, Y_S_L, Y_V_L] = [0,28,146]
	[Y_H_M, Y_S_M, Y_V_M] = [40,255,255]

	# HSV Color Threshold - White 
	[W_H_L, W_S_L, W_V_L] = [0,0,178]
	[W_H_M, W_S_M, W_V_M] = [255,255,255]
 	while True:
 		
 		# Read Image
 		ret ,img = cap.read()
 		if ret==False:
 			break
		h,  w = img.shape[:2]
		
		# Given Intrinsic camera parameters
		K = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
			 [  0.00000000e+00,   1.14818221e+03 ,  3.86046312e+02],
			 [  0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])

		# Given distortion camera parameters
		dist = np.array([ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,    2.20573263e-02])
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
		
		# Undistorted image
		dst = cv2.undistort(img,K,dist,None,newcameramtx)

		# 4 Points from image and ground truth for finding Homography
		h_pts_img = np.array([[396,640],[660,450],[705,450],[925,630]], dtype="float32")
		h_pts_gt = np.array([[32,512],[32,0],[192,0],[192,512]],dtype="float32")
		H,_ = cv2.findHomography(h_pts_img, h_pts_gt)
		
		# Bird's Eye View
		warped = cv2.warpPerspective(dst,H,(256,512))
		gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

		# HSV Threshold yellow and white lanes
		HSV = cv2.cvtColor(warped,cv2.COLOR_BGR2HSV)
		edges_l = cv2.inRange(HSV, (Y_H_L, Y_S_L, Y_V_L), (Y_H_M, Y_S_M, Y_V_M))
		edges_r = cv2.inRange(HSV, (W_H_L, W_S_L, W_V_L), (W_H_M, W_S_M, W_V_M))
		edges = edges_l + edges_r
		

		hist_along_x_left = np.sum(edges[:,:int(edges.shape[1]/2)]>0, axis=0)
		hist_along_x_right = np.sum(edges[:,int(edges.shape[1]/2):]>0, axis=0)

		## Uncomment to plot histogram
		# plt.figure("Thres")
		# plt.imshow(edges)
		# plt.plot(hist_along_x_left)
		# plt.plot(np.hstack([np.zeros_like(hist_along_x_left), hist_along_x_right]))
		# plt.show()

		# Extract maxima on left and right lanes
		left_lane = np.argmax(hist_along_x_left)
		right_lane = int(edges.shape[1]/2) + np.argmax(hist_along_x_right)

		# Remove Noise threshold
		if left_lane-15>0:
			edges[:, :left_lane-15] = 0
		edges[:, left_lane+15:int(edges.shape[1]/2)]=0
		edges[:, int(edges.shape[1]/2):right_lane-15]=0
		if right_lane+15<edges.shape[0]:
			edges[:, right_lane+15:]=0

		# Get left and right lane features
		left_lane_pts = np.where(edges[:, :left_lane+15]>0)
		right_lane_pts = np.where(edges[:, right_lane-15:]>0)
		right_lane_pts = (right_lane_pts[0], (right_lane -15 + right_lane_pts[1]))

		# Get best fit line equation for left and right lanes
		pl = np.polyfit(left_lane_pts[0],left_lane_pts[1],2)
		pr = np.polyfit(right_lane_pts[0],right_lane_pts[1],2)
		
		# Radius of curvature formula https://www.quora.com/How-can-I-find-the-curvature-of-parabola-y-ax-2-bx-c
		curvature =(10**10) * ((2*pl[0])/((1+((2*pl[0]*(-pl[2]/(2*pl[0])))+pl[1])**2)*np.sqrt(1+((2*pl[0]*(-pl[2]/(2*pl[0])))+pl[1])**2)))
		
		# Heuristic for the lane curvature
		if curvature>15:
			curve = "Straight"
		if (curvature>3 and curvature<15):
			curve = "Right"
		if (curvature<-4 and curvature>-10):
			curve = "Left"		

		# Find all points of the polyfit lines
		lane = np.zeros((warped.shape[0],warped.shape[1],2))
		for x in range(0,512):
			if (((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])) < 256) and (((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])) > 0):
				lane[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])),0] = 255
			if ((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])<256):
				lane[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])),1] = 255

		# Inverse Perspective Transform
		world_lane_left = cv2.warpPerspective(lane[:,:,0],np.linalg.inv(H),(w,h))
		world_lane_right = cv2.warpPerspective(lane[:,:,1],np.linalg.inv(H),(w,h))
		dst[world_lane_left>0] = [255,0,0]
		dst[world_lane_right>0] = [0,0,255]

		# Get points for FillPoly the lane 
		left = np.argwhere(world_lane_left>0)
		left = np.flip(left,axis=1)
		right = np.argwhere(world_lane_right>0)
		right = np.flip(right,axis=1)

		poly_lane = np.zeros((dst.shape[0],dst.shape[1],3),np.uint8 )
		poly_lane = cv2.fillPoly(poly_lane,[np.vstack((np.flip(left,axis=0),right))],[0,255,0])

		# Add Lane onto the Undistorted Image
		dst = cv2.addWeighted(dst,0.8,poly_lane,0.2,1)
		dst = cv2.putText(dst,curve,(500,150),1,cv2.FONT_HERSHEY_DUPLEX,(0,0,255),2,cv2.LINE_AA)
		cv2.imshow("Lane World",dst[60:660,45:1245])
		out.write(dst[60:660,45:1245])
		cv2.waitKey(1)
	print("Saving Video!")
	out.release()
	cap.release()
    

if __name__=="__main__":
	main()
	# trial()