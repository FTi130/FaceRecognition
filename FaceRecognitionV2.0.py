import face_recognition
import platform
import pickle
import numpy as np
from datetime import datetime, timedelta
import cv2

knownfacesencoded = []
knownfacesmeta = []

def save_known_faces():
    with open("faces.dat","wb") as face_data_file:
        face_data= [knownfacesencoded,knownfacesmeta]
        pickle.dump(face_data,face_data_file)
        print("Лицо записано на диск")
        

def load_known_faces():
    global knownfacesencoded,knownfacesmeta

    try:
        with open("faces.dat","rb") as face_data_file:
            knownfacesencoded, knownfacesmeta=pickle.load(face_data_file)
            print("Лица загружены с диска")
    except FileNotFoundError as e:
        print("Данные не найдены в базе, создается новая запись")
        pass

def register_new_face(face_encoding, face_image):
    knownfacesencoded.append(face_encoding)
    knownfacesmeta.append({"first_seen": datetime.now(),
                           "first_seen_interaction": datetime.now(),
                           "last_seen":datetime.now(),
                           "seen_count":1,
                           "seen_frames":1,
                           "face_image":face_image})

def lookup_known_face(face_encoding):
    metadata =None
    if len(knownfacesencoded)==0:
        return metadata
    face_distances = face_recognition.face_distance(knownfacesencoded,face_encoding)
    best_match_index = np.argmin(face_distances)
    if  face_distances[best_match_index] < 0.65:
        metadata = knownfacesmeta[best_match_index]
        metadata["last_seen"]=datetime.now()
        metadata["seen_count"] +=1
    return metadata

def main_loop():
    cap=cv2.VideoCapture(0)
    number_since_save = 0
    while True:
        ret,frame=cap.read()
        small_frame = cv2.resize(frame,(0,0), fx=0.25,fy=0.25)
        
        face_locations=face_recognition.face_locations(small_frame)
        face_encodings=face_recognition.face_encodings(small_frame,face_locations)


        face_labels=[]
        for face_location, face_encoding in zip(face_locations,face_encodings):
            metadata=lookup_known_face(face_encoding)

            if metadata is not None:
                time_at_door=datetime.now()-metadata["first_seen_interaction"]
                face_label= f"V kadre {int(time_at_door.total_seconds())}s"
            else:
                face_label="Kto to noviy!"
                top,right,bottom,left = face_location
                face_image=small_frame[top:bottom,left:right]
                face_image =cv2.resize(face_image,(150,150))

                register_new_face(face_encoding,face_image)

            face_labels.append(face_label)

        for (top,right,bottom,left), face_label in zip(face_locations,face_labels):
            top *=4
            right *=4
            bottom *=4
            left *=4

            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

            cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
            cv2.putText(frame,face_label,(left+6,bottom-6),cv2.FONT_HERSHEY_DUPLEX, 0.7,(255,255,255),1)
            
        recv =0
        for metadata in knownfacesmeta:
            if datetime.now()-metadata["last_seen"] < timedelta(seconds=10) and metadata["seen_frames"] > 5:
                x_position = recv *150
                frame[30:180,x_position:x_position+150] = metadata["face_image"]
                recv +=1

                visits = metadata["seen_count"]
                visit_label=f"{visits} visits"
                if visits ==1:
                    visit_label="Первое посещение"
                cv2.putText(frame,visit_label,(x_position+10,170),cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1)
        if recv > 0:
            cv2.putText(frame,"Персоны перед камерой",(5,18),cv2.FONT_HERSHEY_DUPLEX, 0.9,(255,255,255),1)

        cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            save_known_faces()
            break

        if len(face_locations)>0 and number_since_save > 100:
            save_known_faces()
            number_since_save=0
        else:
            number_since_save+=1
                        
    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    load_known_faces()
    main_loop()
    
