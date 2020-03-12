import cv2
import numpy as np 
import math

def hough_line(img, sample_frame):


	lines = cv2.HoughLines(img, 1, 1*np.pi/180, 180, lines=None, srn=0, stn=0, min_theta=10*(np.pi/180), max_theta=60*(np.pi/180))
	lines2 = cv2.HoughLines(img, 1, 2*np.pi/180, 180, lines=None, srn=0, stn=0, min_theta=130*(np.pi/180), max_theta=170*(np.pi/180))

	if lines is None:
		lines = lines2
	elif lines2 is not None:
		lines = np.vstack([lines, lines2])

	# dist_img_cropped = sample_frame[300:611,0:1278,:]
	dist_img_cropped = sample_frame
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
			pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
			cv2.line(dist_img_cropped, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

	cv2.imshow("canny", img)
	cv2.imwrite("Canny_Filtered_Image.png", img)
	cv2.imshow("Hough Lines", dist_img_cropped)
	cv2.imwrite("Hough_Line_Transform.png", dist_img_cropped)
	cv2.waitKey(0)