# import the necessary packages python videorecorgnition.py --encodings encodings.pickle --output output/webcam_face_recognition_output.avi --display 1
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import requests
from time import gmtime, strftime
from datetime import datetime

def addPersonInStore(step,vid,fullN,timeS,timeE,inS,needH,spend,imN,timeSeen):
	#########################################
	URL = "https://testboti861443trial.hanatrial.ondemand.com/Customer/persondetected.xsjs"
  
	if step == 1:
		imN=fullN+".png"

	# defining a params dict for the parameters to be sent to the API
	PARAMS = {'step':step,'vid': vid, 'fullN':fullN,'timeS':timeS,'timeE':timeE,'inS':inS, 'needH':needH, 'spend':spend,'imN': imN, "timeSeen":timeSeen}
	 
	# sending get request and saving the response as response object
	#r = requests.get(URL, PARAMS)
	r = requests.get(url=URL,params=PARAMS)
	print(r.json())
	# extracting data in json format
	return str(r.json())
	#############################
def maxVisit():
    #########################################
	URL = "https://testboti861443trial.hanatrial.ondemand.com/Customer/maxVisitaId.xsjs"
	r = requests.get(url=URL)
	print(r.json())
	# extracting data in json format
	return str(r.json())
	#############################

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
 
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(1.0)

visitaID = {}
visitaNo = int(maxVisit())
beforeNames = []
timeStart = {}
timeEnd = {}

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
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
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"
 
		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
 
			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
		
		for name in names:
		
			try:
				indexNameBefore = beforeNames.index(name)
			except:
				indexNameBefore = -1
			
			if indexNameBefore <0:
				visitaNo += 1
				visitaID[name]=visitaNo
				timeStart[name] = datetime.now()
				print("-----------------------------")
				print("Nueva Cara")
				print(visitaID[name])
				print(name)
				print(timeStart[name])
				print("-----------------------------")
				addPersonInStore(1,visitaID[name],name,timeStart[name],0,1,0,981,name,0)
			else:
				timeSeen = datetime.now() - timeStart[name]
				timeSeen = int(timeSeen.total_seconds())
				addPersonInStore(2,visitaID[name],0,0,0,0,0,0,0,timeSeen)
				
			
	for beforeName in beforeNames:
			
		try:
			indexName = names.index(beforeName)
		except:
			timeEnd[beforeName] = datetime.now()#.strftime('%Y-%m-%d %H:%M:%S')
			timeSeen = timeEnd[beforeName] -  timeStart[beforeName]
			timeSeen = int(timeSeen.total_seconds())
			print("-----------------------------")
			print("Cara se fue")
			print(visitaID[beforeName])
			print(beforeName)
			print(timeEnd[beforeName])
			print(timeSeen)
			print("-----------------------------")
			addPersonInStore(2,visitaID[beforeName],0,0,timeEnd[beforeName],0,0,0,0,timeSeen)
		
	# save it for the next iteration
	beforeNames = names;
		
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
 
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	"""
	
	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)
 
	# if the writer is not None, write the frame with recognized
	# faces to disk
	if writer is not None:
		writer.write(frame)
	"""
		
	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
 
# check to see if the video writer point needs to be released
if writer is not None:
	writer.release
	


	