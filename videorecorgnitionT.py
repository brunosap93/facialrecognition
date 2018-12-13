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

def addPersonInStore(step,id,fullN,startD,endD,inS,needH,spend,timeSeen):
	
	#print("Inicia Thread")
	vid = id#maxVisit(fullN) + 1
	#########################################
	URL = "https://xs01i861443trial.hanatrial.ondemand.com/i861443/addperson.xsjs"

	# defining a params dict for the parameters to be sent to the API
	PARAMS = {'step':step,'vid' : vid, 'id': id, 'fullN':fullN, 'startD':startD,'endD':endD, "timeS":timeSeen,'statusS':inS, 'needH':needH, 'spend':spend}
	 
	# sending get request and saving the response as response object
	#r = requests.get(URL, PARAMS)
	#r = requests.get(url=URL,params=PARAMS)
	#print(str(r.json()))
	# extracting data in json format
	return #str(r.json())
	#############################
def maxVisit(nombre):
    #########################################
	if nombre == "todos":
		URL = "https://xs01i861443trial.hanatrial.ondemand.com/i861443/customervisits.xsodata/visits/$count"
	else:
		URL = "https://xs01i861443trial.hanatrial.ondemand.com/i861443/customervisits.xsodata/visits/$count?$filter=FULLNAME eq '"+nombre+"'"
	#r = requests.get(url=URL)
	#result = int(r.json())
	#print(r.json())
	# extracting data in json format
	return result
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

with open("encodings.pickle",'r+b') as f:
	data = pickle.dump({"a dict":True},f,protocol=2)
 
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = cv2.VideoCapture(1)
vs = VideoStream(src=0).start()
#vs.set(3, 1920)
#vs.set(4, 1080)

sapLogo = cv2.imread("sap.png",1)
#sapLogo = cv2.resize(sapLogo, (88,45))


time.sleep(1.0)
fps = FPS().start()

visitaID = {}
visitaNo = 23#maxVisit("todos")
#visitaBrunoNo = maxVisit("Bruno Raul Guerrero")
beforeNames = []
timeStart = {}
timeEnd = {}
#Start thread list
threads = list()

needH = "No"

#print("Visitas Totales: "+repr(visitaNo)+"; Visitas Bruno Totales: "+repr(visitaBrunoNo))

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(rgb, width=750)
	r = frame.shape[1] / float(rgb.shape[1])
 	
	# detect es of the bounding boxes
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
		name = "Empleado no registrado"
 
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
				aP = threading.Thread(target=addPersonInStore, args=(1,visitaID[name],name,timeStart[name],0,"En tienda","No",981,0,))
				threads.append(aP)
				aP.start()
			else:
				timeSeen = datetime.now() - timeStart[name]
				timeSeen = int(timeSeen.total_seconds())
				#addPersonInStore(2,visitaID[name],0,0,0,0,0,0,0,timeSeen)
				if timeSeen > 30:
					needH = "Yes"
				else:
					needH = "No"
				aP = threading.Thread(target= addPersonInStore, args=(2,visitaID[name],name,0,0,"En tienda",needH,0,timeSeen,))
				threads.append(aP)
				aP.start()
		if len(names) == 0:
			aP = threading.Thread(target=addPersonInStore, args=(3,0,0,0,datetime.now(),"Termino Visita",0,0,0,))
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
			aP = threading.Thread(target=addPersonInStore, args=(4,visitaID[beforeName],beforeName,0,timeEnd[beforeName],"Termino Visita",needH,0,timeSeen,))
			threads.append(aP)
			aP.start()
		
	# save it for the next iteration
	beforeNames = names;
		
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		
		idEmpledo = ""
		if name == "Bruno_Guerrero_Padilla":
			idEmpledo = "I861443"
		elif name == "Adrian_Ramirez":
			idEmpledo = "K549301"
		elif name == "Sergio_Sahagun":
			idEmpledo = "I861892"
			
		name = name.replace("_", " ")
 
		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = bottom + 15 if bottom + 15 < 15 else bottom - 15
		cv2.putText(frame, "Control and automation engineer", (left, y+50), 16,
			0.6, (0, 255, 0), 2)
		cv2.putText(frame, name, (left, y+70), 16,
			0.6, (0, 255, 0), 2)
		ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		#cv2.putText(frame, "Bienvenido a KUO "+ahora, (50, 30), 16,
		#	0.8, (229, 25, 55), 2
		cv2.putText(frame, "Hello Stefan", (50, 30), 16,
			0.8, (229, 25, 55), 2)
		
	# Show in the screen
	
	'''pV = threading.Thread(target=playVideo,args=(frame,))
	threads.append(pV)
	pV.start()'''
	
	#cv2.namedWindow("Bienvenido",cv2.WINDOW_NORMAL)
	#cv2.resizeWindow("Bienvenido", 1280,1024)
	'''
	sapLogo.copyTo(frame(cv2.Rect(10,10,sapLogo.cols, sapLogo.rows)))
	
	x_offset=y_offset=10
	frame[y_offset:y_offset+sapLogo.shape[0],x_offset:x_offset+sapLogo.shape[1]] = sapLogo
	'''
	#cv2.imshow("Logo", sapLogo)
	cv2.imshow("Bienvenido", frame)
	
	key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		addPersonInStore(3,0,0,0,datetime.now(),"Termino Visita",0,0,0,)
		break
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()


	