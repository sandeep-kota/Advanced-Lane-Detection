import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# def nothing(x):
# 	pass


# img = cv2.imread('/home/default/ENPM673/Project2/Dataset/data_1/data/0000000001.png',-1)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



# h,  w = img.shape[:2]
# K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
# 				[0.000000e+00, 9.019653e+02, 2.242509e+02],
# 				 [0.000000e+00, 0.000000e+00, 1.000000e+00]])

# dist = np.array([ -3.639558e-01 ,1.788651e-01, 6.029694e-04 ,-3.922424e-04 ,-5.382460e-02])
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
# print("New Camera Matrix: ",newcameramtx)
# print("ROI: ",roi)
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow("Image",dst)
# # dst = cv2.blur(dst,(10,10))
# dst = cv2.cvtColor(dst,cv2.COLOR_RGB2GRAY)

# p1 = [70,355]
# p2 = [900,355]
# p3 = [530,145]
# p4 = [645,145]
# pol = []
# pol.append(p1)
# pol.append(p2)
# pol.append(p4)
# pol.append(p3)
# pol = np.array(pol)
# mask = np.zeros((dst.shape[0],dst.shape[1]))
# mask = cv2.fillConvexPoly(mask,pol,255)
# print("mask.shape:", mask.shape)

# new = cv2.bitwise_and(dst.astype(np.uint8), mask.astype(np.uint8))
# cv2.imshow("ROI",new)

# # dst_pts = np.array([[200, bottom_px], [200, 0], [1000, 0], [1000, bottom_px]], np.float32)


# cv2.waitKey(0)


def main():

	# for i in 

	img = cv2.imread("/home/default/ENPM673/Project2/Dataset/data_1/data/0000000125.png")
	print(img.shape)
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
	
	left_lane = np.argmax(hist_along_x_left)
	right_lane = int(edges.shape[1]/2) + np.argmax(hist_along_x_right)
	print("my lanes:", left_lane, right_lane)

	edges[:, :left_lane-15] = 0
	edges[:, left_lane+15:int(edges.shape[1]/2)]=0
	edges[:, int(edges.shape[1]/2):right_lane-15]=0
	edges[:, right_lane+15:]=0

	plt.figure("New Edges")
	plt.imshow(edges)

	left_lane_pts = np.where(edges[:, :left_lane+15]>0)
	right_lane_pts = np.where(edges[:, right_lane-15:]>0)
	# print("Right :", right_lane_pts)
	right_lane_pts = (right_lane_pts[0], (int(edges.shape[1]/2) + 25 + right_lane_pts[1]))

	print("Left :", left_lane_pts)
	print("Right :", right_lane_pts)
	# plt.imshow(edges)
	# plt.show()

	pl = np.polyfit(left_lane_pts[0],left_lane_pts[1],2)
	pr = np.polyfit(right_lane_pts[0],right_lane_pts[1],2)
	print("Polyfit Left: ",pl)
	print("Polyfit Right: ",pr)

	for x in range(0,512):
		print("Value :",x,((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2])),((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2])))
		warped[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2]))] = [255,0,0]
		warped[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2]))] = [0,255,0]

	plt.figure("Warped")
	plt.imshow(warped)
	plt.show()
	sys.exit(0)


	# hist = np.zeros((edges.shape[1],1),np.uint8)

	# for i in range(0,hist.shape[0]):
	# 	count = 0
	# 	if (edges[i,:])
	# hist = cv2.calcHist([edges],[0],None,[256],[0,256])

	# hist = cv2.calcHist([edges],[0],None,[50],[np.min(edges),np.max(edges)])
	# hist,bins = np.histogram(edges.ravel(),50,[np.min(edges),np.max(edges)])
	# edges[edges < -10414.32] = 0
	# edges[edges > 9380.64] = 0

	# lanes = np.zeros((512,256))
	# lanes[edges_x < -100] = 255
	# plt.imshow(edges)
	# plt.plot(hist)
	# plt.imshow(lanes)
	# plt.imshow(edges_x,cmap="gray")
	# plt.show()
	# sys.exit(0)

	# cv2.imshow("B/W",gray_warped)
	# plt.imshow(edges)
	# plt.show()
	# sys.exit(0)
	# cv2.imshow("Edges",edges)
	# # cv2.imshow("Lanes",lanes)
	# # cv2.imshow("Edges_x",edges_x)
	# # cv2.imshow("Edges_y",edges_x)


	# left_pts = []
	# for i in range(0,edges.shape[0]):
	# 	for j in range(64,128):
	# 		if edges[i,j]>0:
	# 			left_pts.append([i,j])
	# left_pts = np.array(left_pts)
	# print("Left Points :",left_pts)

	# right_pts = []
	# for i in range(0,edges.shape[0]):
	# 	for j in range(128,192):
	# 		if edges[i,j]>0:
	# 			right_pts.append([i,j])
	# right_pts = np.array(right_pts)
	# print("Right Points :",right_pts)



	# 	# warped[x,int((pl[0]*(x**2)) + (pl[1]*(x**1)) + (pl[2]))] = [0,0,255]
	# 	# warped[x,int((pr[0]*(x**2)) + (pr[1]*(x**1)) + (pr[2]))] = [0,255,0]
	# # 	# print(warped[0,1])
	# # cv2.imshow("Warped", warped)
	# cv2.waitKey()


def trial():
	img = cv2.imread("/home/default/ENPM673/Project2/Dataset/data_1/data/0000000245.png")
	# cv2.imshow("Img[0]",img[:,:,0])
	# cv2.imshow("Img[1]",img[:,:,1])
	# cv2.imshow("Img[2]",img[:,:,2])

	# imgHLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
	imgG = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# # cv2.imshow("Img[H]",imgHLS[:,:,0])
	# # cv2.imshow("Img[L]",imgHLS[:,:,1])
	# # cv2.imshow("Img[S]",imgHLS[:,:,2])
	# # cv2.waitKey()
	
	# fig = plt.figure(figsize=(10,30))
	# fig.add_subplot(4,1,1)
	# plt.imshow(imgHLS[:,:,0], cmap = "gray")
	# fig.add_subplot(4,1,2)
	# plt.imshow(imgHLS[:,:,1],cmap = "gray")
	# fig.add_subplot(4,1,3)
	# plt.imshow(imgHLS[:,:,2],cmap = "gray")
	# # fig.add_subplot(4,1,4)
	# # plt.imshow(imgG,cmap = "gray")
	# # ax[0,1].imshow(imgHLS[:,:,1])
	# # ax[0,2].imshow(imgHLS[:,:,2])
	# plt.show()
	
	lane = np.zeros((imgG.shape[0],imgG.shape[1]))
	lane[imgG>230]=1
	
	sobel = cv2.Sobel(lane,cv2.CV_64F,1,1)
	
	cv2.imshow("Lanes",lane)
	cv2.imshow("Sobel",sobel)

	cv2.waitKey()
if __name__=="__main__":
	main()
	# trial()