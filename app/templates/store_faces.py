import cv2
import numpy as np
import os
vid  = cv2.VideoCapture(0)
dataset = cv2.CascadeClassifier("data.xml")
i = 0
face_list = []
while True:
    ret, frame = vid.read()
    if ret:
        print(i)
        i+=1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(frame, 1.2)
        for x,y,w,h in faces:
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255,0), 2)
        cv2.imwrite("face.png", face)
        face = cv2.resize(face, (50, 50))
        face_list.append(face)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27 or i == 70:
            break
    else:
        print("Camera Not Found")

file_read = open("user.txt",'r')
i=int(file_read.read())
np.save(f"faces/user_{i}.npy", np.array(face_list))
vid.release()
cv2.destroyAllWindows()
i=i+1
file_write = open("user.txt",'w')
file_write.write(str(i)) 
file_write.close()
file_read.close()

    # name_file=open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/name.txt",'a')
    # name_file.writelines(str(name))
    # name_file.writelines("\n")
    # name_file.close()

    # password1=open("email.txt",'r')
    # p=password1.readlines()[5]
    # password1.close()
    # p=p.strip()
    # print(p)


