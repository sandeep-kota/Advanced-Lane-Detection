import numpy as np
import cv2
import os
import copy
import sys
sys.write_bytecode = False
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter


# Playing with the brightness and contrast
def improveQuality(frame):
	new_frame = np.zeros_like(frame)
	alpha = 2.5
	beta = 40
	
	new_frame = alpha * frame.astype(np.int32) + beta

	new_frame = cv2.convertScaleAbs(new_frame)

	return new_frame.astype(np.uint8)


def main():
	# video_path = "/home/satyarth934/enpm673/Project2/Data/Night Drive - 2689_Problem1.mp4"
	video_path = sys.argv[1]
	if video_path is None or video_path.split(".")[-1] not in  ["mp4", "avi"]:
		print("Please give path to an mp4 or avi file.")
		sys.exit(0)

	# out = cv2.VideoWriter('Improved_Frames_Prob1.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (1920,1080))

	cap= cv2.VideoCapture(video_path)
	i = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		# frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
		if ret == False:
			break

		improved_frame = improveQuality(frame)

		# out.write(improved_frame)
		
		cv2.imshow('frame.png', frame)
		cv2.imshow('improved_frame.png', improved_frame)
		if cv2.waitKey(1) & 0xFF == ord('w'):
			cv2.imwrite('frame%d.png'%(i), frame)
			cv2.imwrite('./improved_frame%d.png'%(i), improved_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		i += 1
	 
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()