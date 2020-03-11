import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def main():

	# for i in 

	path = '/home/default/ENPM673/Project2/Dataset/data_1/data/'

	# imgs = os.listdir(path)
	imgs = ["0000000288.png"]
	for i in imgs:
		img = cv2.imread(path + str(i))
		print(path + str(i))
		# cv2.imshow("Original",img)


		h,  w = img.shape[:2]
		K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
						[0.000000e+00, 9.019653e+02, 2.242509e+02],
						 [0.000000e+00, 0.000000e+00, 1.000000e+00]])
		dist = np.array([ -3.639558e-01 ,1.788651e-01, 6.029694e-04 ,-3.922424e-04 ,-5.382460e-02])
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
		dst = cv2.undistort(img,K,dist,None,newcameramtx)
		# cv2.imshow("Undistorted",dst)


		
		# h_pts_img = np.array([[225,512],[595,272],[707,272],[917,512]], dtype="float32")
		h_pts_img = np.array([[364,417],[622,254],[707,254],[846,417]], dtype="float32")
		h_pts_gt = np.array([[64,512],[64,0],[192,0],[192,512]],dtype="float32")
		
		H,_ = cv2.findHomography(h_pts_img, h_pts_gt)
		# print("H Matrix: ",H)
		
		warped = cv2.warpPerspective(dst,H,(256,512))

		# cv2.imshow("Unidistorted :", dst)
		gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

		plt.figure("Warped")
		plt.imshow(gray)
		# gray_warped = cv2.GaussianBlur(gray_warped,(5,5),0)
		# gray_warped = cv2.equalizeHist(gray_warped)
		edges = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
		edges[gray>245] = 255
		plt.figure("Thres")
		plt.imshow(edges)
		# sys.exit(0)
		# edges = cv2.Canny(gray_warped,0,300)
		# edges = cv2.Sobel(gray_warped,cv2.CV_64F,1,0,ksize = 3)
		# edges_y = cv2.Sobel(gray_warped,cv2.CV_64F,0,1,ksize=5)
		# edges_s = np.sqrt(edges_x**2 + edges_y**2)
		# edges = cv2.Sobel(gray_warped,cv2.CV_64F,0,1,ksize=7)
		
		hist_along_x_left = np.sum(edges[:,:int(edges.shape[1]/2)]>0, axis=0)
		hist_along_x_right = np.sum(edges[:,int(edges.shape[1]/2):]>0, axis=0)

		plt.plot(hist_along_x_left)
		plt.plot(np.hstack([np.zeros_like(hist_along_x_left), hist_along_x_right]))
		# plt.show()

		left_lane = np.argmax(hist_along_x_left)
		right_lane = int(edges.shape[1]/2) + np.argmax(hist_along_x_right)
		print("my lanes:", left_lane, right_lane)

		edges[:, :left_lane-15] = 0
		edges[:, left_lane+15:int(edges.shape[1]/2)]=0
		edges[:, int(edges.shape[1]/2):right_lane-15]=0
		edges[:, right_lane+15:]=0

		plt.figure("New Edges")
		plt.imshow(edges)
		# cv2.imshow("New Edges",edges)

		left_lane_pts = np.where(edges[:, :left_lane+15]>0)
		right_lane_pts = np.where(edges[:, right_lane-15:]>0)
		# print("Right :", right_lane_pts)
		right_lane_pts = (right_lane_pts[0], (right_lane -15 + right_lane_pts[1]))

		print("Left :", left_lane_pts)
		print("Right :", right_lane_pts)
		plt.imshow(edges)
		plt.show()
		
		sys.exit(0)

		# pl = np.polyfit(left_lane_pts[0],left_lane_pts[1],2)
		# pr = np.polyfit(right_lane_pts[0],right_lane_pts[1],2)
		# print("Polyfit Left: ",pl)
		# print("Polyfit Right: ",pr)

		for x in range(0,512):
			# print("Value :",x,((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])),((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])))
			warped[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2]))] = [255,0,0]
			warped[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2]))] = [0,0,255]

		# plt.figure("Warped")
		cv2.imshow("Warped",warped)
		cv2.waitKey(0)


if __name__=="__main__":
	main()
	# trial()