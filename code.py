import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def main():

	# for i in 

	path = '/home/default/ENPM673/Project2/Dataset/data_1/data/'

	imgs = os.listdir(path)
	# imgs = ["0000000170.png"]
	for i in imgs:
		img = cv2.imread(path + str(i))
		print(path + str(i))
		# cv2.imshow("Original",img)

		# img = cv2.flip(img,1)

		h,  w = img.shape[:2]
		K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
						[0.000000e+00, 9.019653e+02, 2.242509e+02],
						 [0.000000e+00, 0.000000e+00, 1.000000e+00]])
		dist = np.array([ -3.639558e-01 ,1.788651e-01, 6.029694e-04 ,-3.922424e-04 ,-5.382460e-02])
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
		dst = cv2.undistort(img,K,dist,None,newcameramtx)
		# cv2.imshow("Undistorted",dst)

		h_pts_img = np.array([[364,417],[622,254],[707,254],[846,417]], dtype="float32")
		h_pts_gt = np.array([[32,512],[32,0],[192,0],[192,512]],dtype="float32")
		
		H,_ = cv2.findHomography(h_pts_img, h_pts_gt)
		# print("H Matrix: ",H)
		
		warped = cv2.warpPerspective(dst,H,(256,512))

		# cv2.imshow("Unidistorted :", dst)
		gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
		edges = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
		edges[gray>245] = 255

		# plt.figure("Thres")
		# plt.imshow(edges)
		# cv2.imshow("Thresh",edges)

		hist_along_x_left = np.sum(edges[:,:int(edges.shape[1]/2)]>0, axis=0)
		hist_along_x_right = np.sum(edges[:,int(edges.shape[1]/2):]>0, axis=0)

		# plt.plot(hist_along_x_left)
		# plt.plot(np.hstack([np.zeros_like(hist_along_x_left), hist_along_x_right]))
		# plt.show()

		left_lane = np.argmax(hist_along_x_left)
		right_lane = int(edges.shape[1]/2) + np.argmax(hist_along_x_right)
		print("my lanes:", left_lane, right_lane)

		if left_lane-15>0:
			edges[:, :left_lane-15] = 0

		edges[:, left_lane+15:int(edges.shape[1]/2)]=0
		edges[:, int(edges.shape[1]/2):right_lane-15]=0
		
		if right_lane+15<edges.shape[0]:
			edges[:, right_lane+15:]=0

		# plt.figure("New Edges")
		# plt.imshow(edges)
		# cv2.imshow("New Edges",edges)

		# cv2.waitKey(0)
		# sys.exit(0)

		left_lane_pts = np.where(edges[:, :left_lane+15]>0)
		right_lane_pts = np.where(edges[:, right_lane-15:]>0)
		right_lane_pts = (right_lane_pts[0], (right_lane -15 + right_lane_pts[1]))

		# print("Left :", left_lane_pts)
		# print("Right :", right_lane_pts)
		# plt.imshow(edges)
		# plt.show()
		
		# sys.exit(0)

		pl = np.polyfit(left_lane_pts[0],left_lane_pts[1],2)
		pr = np.polyfit(right_lane_pts[0],right_lane_pts[1],2)
		
		# Radius of curvature formula https://www.quora.com/How-can-I-find-the-curvature-of-parabola-y-ax-2-bx-c
		curvature =(10**11) * ((2*pr[0])/((1+((2*pr[0]*(-pr[2]/(2*pr[0])))+pr[1])**2)*np.sqrt(1+((2*pr[0]*(-pr[2]/(2*pr[0])))+pr[1])**2)))
		curve = "Straight"
		if (curvature>(0.6)):
			curve = "Right"
		if (curvature<-0.6):
			curve = "Left"		

		# print("Polyfit Left: ",pl)
		# print("Polyfit Right: ",pr)
		# print("Curvature :",curvature )
		# right_poly = []
		# left_poly = []
		lane = np.zeros((warped.shape[0],warped.shape[1],2))
		for x in range(0,512):
			# print("Value :",x,((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])),((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])))
			# warped[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2]))] = [255,0,0]
			# warped[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2]))] = [0,0,255]
			lane[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])),0] = 255
			lane[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])),1] = 255
			# right_poly.append([x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2]))])
			# left_poly.append([x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2]))])

		# right_poly = np.array(right_poly)
		# left_poly = np.array(left_poly)

		print("Curve :",curve)
		# plt.figure("Warped")
		# cv2.imshow("Warped",warped)
		# cv2.imshow("Lane l",lane[:,:,0])
		# cv2.imshow("Lane r",lane[:,:,1])
				
		world_lane_left = cv2.warpPerspective(lane[:,:,0],np.linalg.inv(H),(w,h))
		world_lane_right = cv2.warpPerspective(lane[:,:,1],np.linalg.inv(H),(w,h))

		dst[world_lane_left>0] = [255,0,0]
		dst[world_lane_right>0] = [0,0,255]
		print([world_lane_left>0,world_lane_right>0])
		# world_lane = cv2.warpPerspective(warped,np.linalg.inv(H),(w,h))
		# dst = cv2.fillPoly(dst,[world_lane_left>0,world_lane_right>0],[0,255,0])
		cv2.imshow("Lane World",dst)

		cv2.waitKey(0)


if __name__=="__main__":
	main()
	# trial()