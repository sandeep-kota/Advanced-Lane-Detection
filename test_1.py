import sys
import cv2 
import numpy as np 
import math
import matplotlib.pyplot as plt
import hough_lines
# def hough_line(img, sample_frame):
# 	lines = cv2.HoughLines(img, 1, 2*np.pi/180, 150, lines=None, srn=0, stn=0, min_theta=20*(np.pi/180), max_theta=60*(np.pi/180))
# 	lines2 = cv2.HoughLines(img, 1, 2*np.pi/180, 150, lines=None, srn=0, stn=0, min_theta=110*(np.pi/180), max_theta=160*(np.pi/180))

# 	if lines is None:
# 		lines = lines2
# 	elif lines2 is not None:
# 		lines = np.vstack([lines, lines2])

# 	dist_img_cropped = img[300:611,0:1278]
# 	if lines is not None:
# 		for i in range(0, len(lines)):
# 			rho = lines[i][0][0]
# 			theta = lines[i][0][1]
# 			a = math.cos(theta)
# 			b = math.sin(theta)
# 			x0 = a * rho
# 			y0 = b * rho
# 			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
# 			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
# 			cv2.line(dist_img_cropped, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

# 	cv2.imshow("Hough Lines", sample_frame)
# 	cv2.waitKey(0)

def main():
	# cap = cv2.VideoCapture('/home/revati/ENPM673/Perception-For-Autonomous-Robots/Project2/data_2/challenge_video.mp4')

	# while(cap.isOpened()):
		# Capture frame by frame
		# ret, frame = cap.read()
		# cv2.imwrite("sample_frame.jpg", frame)
	sample_frame = cv2.imread("/home/revati/ENPM673/Perception-For-Autonomous-Robots/Project2/sattu.png")


	# Convert to gray scale
	gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)

	# Display frame
	# cv2.imshow('original video', frame)
	# cv2.waitKey(0)


	# sys.exit(0)
	# Undistort the video
	h, w = sample_frame.shape[:2]
	camera_matrix = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
							[0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
							[0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
	distortion_coef = np.array([[ -2.42565104e-01,  -4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
								2.20573263e-02]])

	newCamMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coef, (w,h), 0, (w,h))
	print("newCamMatrix", newCamMatrix)
	dist_img = cv2.undistort(sample_frame, camera_matrix, distortion_coef, None, newCamMatrix)

	dist_img_gray = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)


	dist_img_gray = cv2.Canny(dist_img_gray, 100, 200)
	hough_lines.hough_line(dist_img_gray, dist_img)


	# Homography
	# pts_src = np.array([[427,604], [1258,495], [950,455], [665,458]])
	# pts_des = np.array([[427,604], [1258,495], [1253,156], [427,156]])

	pts_src = np.array([[364,417],[622,254],[707,254],[846,417]])
	pts_des = np.array(([[64,512],[64,0],[192,0],[192,512]]))

	h, status = cv2.findHomography(pts_src, pts_des)

	# Warp source image to destination based on Homography
	# warp_img = cv2.warpPerspective(gray, h, (gray.shape[1], gray.shape[0]))
	warp_img = cv2.warpPerspective(gray, h, (256, 512))
	# cv2.imshow("Warped Image", warp_img)
	# cv2.waitKey(0)

	# Display the images
	# cv2.imshow("Source Image", gray)
	# cv2.imshow("Warped Image", warp_img)

	# cv2.waitKey(0)


	# Crop Image
	# x, y, w, h = roi
	# dist_img = dist_img[y:y+h, x:x+w]
	# cv2.imshow("Undistorted", dist_img)
	# cv2.waitKey(0)

	# Denoising the image
	# denoise_img = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
	# denoise_img = cv2.GaussianBlur(dist_img, (11,11), 0)
	# denoise_img = cv2.bilateralFilter(dist_img, 9,75,75)
	# denoise_img = cv2.Canny(warp_img, 100, 200)
	denoise_img = warp_img
	# cv2.imshow("Denoised image", denoise_img)
	# cv2.waitKey(0)

	# Extracting the ROI using image cropping
	# crop_img = denoise_img[300:611,0:1278]
	# cv2.imshow("Cropped Image", crop_img)
	# cv2.waitKey(0)


	# Sobel filtering 
	sobelx = cv2.Sobel(denoise_img, cv2.CV_64F, 1, 0, ksize=3)
	# sobely = cv2.Sobel(denoise_img, cv2.CV_64F, 0, 1, ksize=3)

	sobelx = cv2.convertScaleAbs(sobelx)
	# sobely = cv2.convertScaleAbs(sobely)

	# cv2.imshow("Sobel_x", sobelx)
	# cv2.waitKey(0)
	# cv2.imshow("Sobel_y", sobely)

	hist_list = np.sum((sobelx>200), axis=0)
	plt.plot(hist_list)
	plt.show()
	mid_point = len(hist_list)/2
	# sys.exit(0)
	first_half_max = np.argmax(hist_list[:int(mid_point)]) 
	second_half_max = np.argmax(hist_list[int(mid_point):]) + mid_point

	# print(hist_list[first_half_max-20:first_half_max+20])
	# print("len",len(hist_list[first_half_max-20:first_half_max+20]))

	sobelx[:,0:int(first_half_max-20)] = 0
	sobelx[:,int(first_half_max+20):int(second_half_max-20)] = 0
	sobelx[:,int(second_half_max+20):] = 0


	plt.imshow(sobelx)
	plt.show()


	


	# cv2.waitKey(0)

	# Histogram 
	# hist = cv2.calcHist([denoise_img],[0],None,[256],[0,256])
	# plt.plot(hist)
	# plt.show()


	# Release capture


main()

