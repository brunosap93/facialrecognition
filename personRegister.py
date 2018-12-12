# import the necessary packages python personDetectRegister.py --dataset dataset --encodings encodings.pickle --videoo output/webcam_face_recognition_output.avi --display 1 --output dataset/TestYo
# python personRegister.py --output dataset\Tania_Reyes_Cabrera
from imutils.video import VideoStream
from imutils import paths
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--dataset", required=True,
#	help="path to input directory of faces + images")
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output",default="dataset\Tania_Reyes_Cabrera",
	help="path to output directory")
#ap.add_argument("-v", "--videoo", type=str,
#	help="path to output video")
#ap.add_argument("-y", "--display", type=int, default=1,
#	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
	
args = vars(ap.parse_args())


#################################################################
# Data Set creation                            ##################
#################################################################

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
writer = None
time.sleep(2.0)
total = 0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, clone it, (just
	# in case we want to write it to disk), and then resize the frame
	# so we can apply face detection faster
	frame = vs.read()
	orig = frame.copy()
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
 
 
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	#encodings = face_recognition.face_encodings(rgb, boxes)
	
	# loop over the recognized faces
	for (top, right, bottom, left) in  boxes:
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
 
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
	cv2.imshow("Frame", frame)		
	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	key = cv2.waitKey(1) & 0xFF
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(
			str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1
 
	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
		
# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))

vs.stop()
###############FIN DATA SET Creation #############

##################################################
#  Encoding process                   ############

print("[INFO] starts encoding process".format(total))



# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["output"]))

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
 
# initialize the list of known encodings and known names
knownEncodings = data["encodings"]#[]
knownNames = data["names"]#[]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	print(imagePath)
	print(name)
 
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
 
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
 
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
		
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()


#######################FIN DE Encoding #########################
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
 
# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()

