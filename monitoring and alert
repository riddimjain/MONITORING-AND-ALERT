# import modules
import pandas
from datetime import datetime
import tkinter.messagebox
from tkinter import *
from PIL import Image,ImageTk
import cv2
import dlib
import playsound
from threading import Thread
import face_recognition
import numpy as np
import winsound
import time
import smtplib
from email.message import EmailMessage
msz = EmailMessage()

#creating gui window
root = Tk()

#title of GUI
root.title('Monitoring & Alert window')

#geometry of GUI
root.geometry('515x450')
frame=Frame(root,relief=RIDGE,borderwidth=7)
frame.pack(fill=BOTH,expand=1)
frame.config(background='lightgreen')
label=Label(frame,text="MONITORING & ALERT ",bg='lightgreen',font="Times 20 bold")
label.place(x=10,y=15)
load=Image.open("img3.jpg")
render=ImageTk.PhotoImage(load)
img=Label(root,image=render)
img.place(x=4,y=40)
label.pack(side=TOP)
def hel():
    tkinter.messagebox.showinfo( "Monitoring & Alert Docs", "This software is for monitoring purpose\nand gives an alert " )
    help(cv2)
def contri():
    tkinter.messagebox.showinfo("Contibutors","\n1. PRIYA RAJ\n2. RAHUL TIWARI \n3. RIDDIM JAIN \n")
def anotherWin():
    tkinter.messagebox.showinfo( "About",'Monitoring & Alert system \n made using \n-Opencv \n-Numpy\n-Tkinter\n-Panads'
                                         '\n-stmplib\n-EmailMessage\n-playsound\n-face_recognition\n-winsound\n-PIL\n-"In Python 3' )
menu =Menu(root)
root.config(menu=menu)
subm1 =Menu(menu)
menu.add_cascade(label='Info',menu=subm1)
subm1.add_command(label="Monitoring & Alert Docs",command=hel)
subm2=Menu(menu)
menu.add_cascade(label='About',menu=subm2)
subm2.add_command(label="Monitoring & Alert",command = anotherWin)
subm2.add_command(label="Contributors",command = contri)

def exitt():
    exit()

def monitor():

    ##Initializing the CascadeClassifier , face detector,landmark detector,alarm.wave
    cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml' )
    shape_preditor_path = "shape_predictor_68_face_landmarks.dat"
    alarm_sound_path = "alarm.wav"

    # status marking for current state
    first_frame = None
    status_list = [None, None]
    times = []
    cl = 0


    # DataFrame to store the time values during which object detection and movement appears
    df = pandas.DataFrame( columns=["Start", "End"] )

    # create a videoCapture object to record video using web cam
    cap = cv2.VideoCapture( 0 )

    # Initializing the face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor( shape_preditor_path )

    def sound_alarm(path):
        playsound.playsound( path )

    def detect_faces(img):
        faces = detector( img )
        return faces

    def is_face_detected(face):
        detected = True
        if not face:
            detected = False
        return detected

    # variables
    FACE_CONSEQ_FRAME1 = 60

    # Counter
    FACE_COUNTER = 0
    FACE_NO_COUNTER=10
    ALARM1_ON = False
    # c=0
    image1 = face_recognition.load_image_file( "priya.jpg" )
    face_encoding1 = face_recognition.face_encodings( image1 )[0]
    image2 = face_recognition.load_image_file( "Riddim.jpg" )
    face_encoding2 = face_recognition.face_encodings( image2 )[0]
    image3 = face_recognition.load_image_file( "rahul.jpg" )
    face_encoding3 = face_recognition.face_encodings( image3 )[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [face_encoding1, face_encoding2, face_encoding3]
    known_face_names = ["PRIYA RAJ", "RIDDIM JAIN", "RAHUL TIWARI"]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while cap.isOpened():

        # Represents NumPy array and booldata type (returns true if python is able to read the videocapture)
        ret, frame = cap.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize( frame, (0, 0), fx=0.25, fy=0.25 )

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations( rgb_small_frame )
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations )
            face_names = []
            for face_encoding in face_encodings:

                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces( known_face_encodings, face_encoding )
                name = "Unknown"
                face_distances = face_recognition.face_distance( known_face_encodings, face_encoding )
                best_match_index = np.argmin( face_distances )
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append( name )
        process_this_frame = not process_this_frame

        # status at the beginning of the recording is zero as the object is not visible
        status = 0

        # Convert the frame color to gray  scale
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

        # Convert the gray scale frame to gaussianBlur
        gray = cv2.GaussianBlur( gray, (21, 21), 0 )
        detection = cascade.detectMultiScale( gray, 1.1, 4 )

        clahe = cv2.createCLAHE()
        enc_img = clahe.apply( gray )
        faces = detect_faces( enc_img )

        # This is used to store the first image/frame of the video
        if first_frame is None:
            first_frame = gray
            continue

        # Calculate the diff. B/W 1st frame and the other frame
        delta_frame = cv2.absdiff( first_frame, gray )

        # Provides the threshold value
        thresh_frame = cv2.threshold( delta_frame, 30, 255, cv2.THRESH_BINARY )[1]
        thresh_frame = cv2.dilate( thresh_frame, None, iterations=2 )

        # Define the contour area basically ,add the borders
        (cnts, _) = cv2.findContours( thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        # Remove noise and shadows.
        for contour in cnts:
            if cv2.contourArea( contour ) < 10000:
                continue

            # Change in status when the object is being detected
            status = 1

            # Create a rectangular box around the object in the frame
            (x, y, w, h) = cv2.boundingRect( contour )
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

            # Display the results
            for (top, right, bottom, left), name in zip( face_locations, face_names ):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw a box around the face
                cv2.rectangle( frame, (left, top), (right, bottom), (0, 0, 255), 2 )
                # Draw a label with a name below the face
                cv2.rectangle( frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED )
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText( frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1 )

        # List of status for every frame
        status_list.append( status )
        status_list = status_list[-2:]

        # Record datetime in a list when change occurs
        if status_list[-1] == 1 and status_list[-2] == 0:
            times.append( datetime.now() )
        if status_list[-1] == 0 and status_list[-2] == 1:
            times.append( datetime.now() )
        face_detected = is_face_detected( faces )

        # To draw rectangle around the face
        for (x, y, w, h) in detection:
            cv2.rectangle( frame, (x, y), (x + w, y + h), (255, 0, 0), 2 )
            cv2.putText( frame, "Person detected ", (320, 30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2 )
            print("person detected")
            cv2.putText( frame, "Press esc key to exit ", (10, 420), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2,
                         (255, 255, 0), 2 )

            #checking face is detected or  not
        if not face_detected:
            c = 0
            FACE_COUNTER += 1
            if not face_detected:
                print( "No face detected" )
                cv2.putText( frame, "Person not Detected", (10, 320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2,
                             (255, 255, 0), 2 )

                #checking face counter
            if FACE_COUNTER >= FACE_CONSEQ_FRAME1:
                if not ALARM1_ON:
                    ALARM1_ON = True
                    cl = cl + 1
                    t = Thread( target=sound_alarm, args=(alarm_sound_path,) )
                    t.start()
                    FACE_COUNTER = 0
                    print( "by FACE not detected." )
                    cv2.putText( frame, "Face not Detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3 )
                    ALARM1_ON = False
        else:
            FACE_COUNTER = 0
        cv2.imshow( 'Monetring window', frame )

        # This will generate a new frame after every 1 miliseconds
        key = cv2.waitKey( 1 ) & 0xFF

        # This will break the loop once the user presses 'esc'
        if cl == 5:
            if status == 1:
                times.append( datetime.now() )
                print( "Maximum time reached" )
            break
        if key == 27:
            if status == 1:
                times.append( datetime.now() )
            break
    print( status_list )
    print( times )

    # Release the camera in some millisecond
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

    # Stores time values in a DataFrame
    for i in range( 0, len( times ), 2 ):
        df = df.append( {"Start": times[i], "End": times[i + 1]}, ignore_index=True )

    # write the DataFrame to a CSV file
    df.to_csv( "Times.csv" )
    msz = EmailMessage()
    msz['subject'] = 'Alert Notification'
    msz['from'] = 'Monitoring Team'
    msz['to'] = 'aarhanaroy@gmail.com'
    with open( "EmailTemplate.txt" ) as myfile:
        data = myfile.read()
        msz.set_content( data )
    with open( "Times.csv", "rb" ) as f:
        file_data = f.read()
        print( "file data in binary", file_data )
        file_name = f.name
        print( "file name is", file_name )
        msz.add_attachment( file_data, maintype="application", subtype="csv", filename=file_name )
    with smtplib.SMTP_SSL( 'smtp.gmail.com', 465 ) as server:
        server.login( "priyaraj00987@gmail.com", "80844444709" )
        server.send_message( msz )
    print( "email sent" )

    # Release the camera in some millisecond
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cap.release()

def monitor2():
    COUNTER = 0
    FACE_COUNTER = 30
    COUNTER_limit = 0
    fcount=0 #unknown face count
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture( 0 )

    # Load a sample picture and learn how to recognize it.
    image1 = face_recognition.load_image_file( "priya.jpg" )
    face_encoding1 = face_recognition.face_encodings( image1 )[0]
    image2 = face_recognition.load_image_file( "Riddim.jpg" )
    face_encoding2 = face_recognition.face_encodings( image2 )[0]
    image3 = face_recognition.load_image_file( "rahul.jpg" )
    face_encoding3 = face_recognition.face_encodings( image3 )[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [face_encoding1, face_encoding2, face_encoding3]
    known_face_names = ["PRIYA RAJ", "RIDDIM JAIN", "RAHUL TIWARI"]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize( frame, (0, 0), fx=0.25, fy=0.25 )

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations( rgb_small_frame )
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations )
            face_names = []

            #checking face encoding
            for face_encoding in face_encodings:


                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces( known_face_encodings, face_encoding )
                name = "Unknown"
                face_distances = face_recognition.face_distance( known_face_encodings, face_encoding )
                best_match_index = np.argmin( face_distances )
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    print( "matches" )
                face_names.append( name )
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip( face_locations, face_names ):


            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle( frame, (left, top), (right, bottom), (255, 0, 0), 2 )

            # Draw a label with a name below the face
            cv2.rectangle( frame, (left, bottom - 35), (right, bottom), (0, 0, 0), cv2.FILLED )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText( frame, name, (left + 6, bottom - 6), font, 1.0, (255, 0, 0), 1 )
            cv2.putText( frame, "Press esc key to exit ", (10, 420), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2,
                         (255, 255, 0), 2 )

            #matching faces
            if not matches[best_match_index]:
                COUNTER = COUNTER + 1
                print( "No  matched face detected" )
                cv2.putText( frame, "unknown face Detected", (10, 320), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2,
                             (255, 255, 0), 2 )

                #counter checking
            if COUNTER >= FACE_COUNTER:
                fr = 500
                d = 2000
                COUNTER_limit = COUNTER_limit + 1
                winsound.Beep( fr, d )
                t = time.strftime( "%y-%m=%d_%H-%M-%S" )
                print( "image" + t + "saved" )
                file = 'E:/image/' + t + '.jpg'
                cv2.imwrite( file, frame )
                fcount += 1

                # Email sending
                COUNTER = 0
                print( "by FACE not detected." )
                import smtplib
                from email.message import EmailMessage
                msz = EmailMessage()
                msz['subject'] = 'Alert_Notification !!!!!!!!!!!!! Unknown face detected'
                msz['from'] = 'MONITORING & Alert TEAM '
                msz['to'] = 'aarhanaroy@gmail.com'
                with open( "EmailTemplate2.txt" ) as myfile:
                    data = myfile.read()
                    msz.set_content( data )
                    with open( 'bad.png', "rb" ) as f:
                        file_data = f.read()
                        print( "file data in binary", file_data )
                        file_name = f.name
                        print( "file name is", file_name )
                        msz.add_attachment( file_data, maintype="application", subtype="csv", filename=file_name )
                with smtplib.SMTP_SSL( 'smtp.gmail.com', 465 ) as server:
                    server.login( "priyaraj00987@gmail.com", "80844444709" )
                    server.send_message( msz )
                print( "email sent" )

        # Display the resulting image
        cv2.imshow( 'Video', frame )

        # Hit 'esc' on the keyboard to quit!
        key = cv2.waitKey( 1 ) & 0xFF
        if COUNTER_limit == 5:
            break
        if key == 27:
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

#creating buttons command

but1= Button(root,padx=5,pady=5,width=30,bg='lightgreen',fg='black',relief=GROOVE,command=monitor,text='START MONITORING',font='helvetica 15 bold')
but1.place(x=70,y=70)

but2= Button(root,padx=5,pady=5,width=30,bg='lightgreen',fg='black',relief=GROOVE,command=monitor2,text='DETECT KNOWNFACE',font='helvetica 15 bold')
but2.place(x=70,y=180)

but3= Button(root,padx=5,pady=5,width=10,bg='lightgreen',fg='black',relief=GROOVE,command=exitt,text='EXIT',font='helvetica 15 bold')
but3.place(x=180,y=280)

root.mainloop()
