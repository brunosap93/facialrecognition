# import the necessary packages python videorecorgnitionT.py --encodings encodings.pickle --output output/webcam_face_recognition_output.avi --display 1
#packages python videorecorgnitionT.py
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import requests
import threading
from time import gmtime, strftime
from datetime import datetime

def addPersonInStore(step,vid,fullN,timeS,timeE,inS,needH,spend,imN,timeSeen):
	#print("Inicia Thread")
	#########################################
	URL = "https://testboti861443trial.hanatrial.ondemand.com/Customer/persondetected.xsjs"
  
	if step == 1:
		imN=fullN+".png"

	# defining a params dict for the parameters to be sent to the API
	PARAMS = {'step':step,'vid': vid, 'fullN':fullN,'timeS':timeS,'timeE':timeE,'inS':inS, 'needH':needH, 'spend':spend,'imN': imN, "timeSeen":timeSeen}
	 
	# sending get request and saving the response as response object
	#r = requests.get(URL, PARAMS)
	r = requests.get(url=URL,params=PARAMS)
	#print(str(r.json()))
	# extracting data in json format
	return #str(r.json())
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
#ap.add_argument("-e", "--encodings", required=True,
#	help="path to serialized db of facial encodings")
#ap.add_argument("-o", "--output", type=str,
#	help="path to output video")
#ap.add_argument("-y", "--display", type=int, default=1,
#	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
 
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(1.0)
fps = FPS().start()

visitaID = {}
visitaNo = int(maxVisit())
beforeNames = []
timeStart = {}
timeEnd = {}
#Start thread list
threads = list()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
 	
	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	
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
				"""print("-----------------------------")
				print("Nueva Cara")
				print(visitaID[name])
				print(name)
				print(timeStart[name])
				print("-----------------------------")"""
				#addPersonInStore(1,visitaID[name],name,timeStart[name],0,1,0,981,name,0)
				#Start Thread for request
				aP = threading.Thread(target=addPersonInStore, args=(1,visitaID[name],name,timeStart[name],0,"En montacarga",0,981,name,0,))
				threads.append(aP)
				aP.start()
			else:
				timeSeen = datetime.now() - timeStart[name]
				timeSeen = int(timeSeen.total_seconds())
				#addPersonInStore(2,visitaID[name],0,0,0,0,0,0,0,timeSeen)
				aP = threading.Thread(target= addPersonInStore, args=(2,visitaID[name],0,0,0,"En montacarga",0,0,0,timeSeen,))
				threads.append(aP)
				aP.start()
			
	for beforeName in beforeNames:
			
		try:
			indexName = names.index(beforeName)
		except:
			timeEnd[beforeName] = datetime.now()#.strftime('%Y-%m-%d %H:%M:%S')
			timeSeen = timeEnd[beforeName] -  timeStart[beforeName]
			timeSeen = int(timeSeen.total_seconds())
			"""print("-----------------------------")
			print("Cara se fue")
			print(visitaID[beforeName])
			print(beforeName)
			print(timeEnd[beforeName])
			print(timeSeen)
			print("-----------------------------")"""
			#addPersonInStore(2,visitaID[beforeName],0,0,timeEnd[beforeName],0,0,0,0,timeSeen)
			aP = threading.Thread(target=addPersonInStore, args=(2,visitaID[beforeName],0,0,timeEnd[beforeName],"Terminó conducción",0,0,0,timeSeen,))
			threads.append(aP)
			aP.start()
		
	# save it for the next iteration
	beforeNames = names;
		
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		'''# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		'''
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(255, 0, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (255, 0, 0), 2)
		'''cv2.putText(frame, "Montacargas CV-18", (left, y-40), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
		cv2.putText(frame, "Uso " + str(visitaNo) +" del mes", (left, y-20), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)'''

		
	# Show in the screen
	
	'''pV = threading.Thread(target=playVideo,args=(frame,))
	threads.append(pV)
	pV.start()'''
	cv2.imshow("Deteccion", frame)
	key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

	