import cv2
import numpy as np
import os

current_dir = os.getcwd()
facesPath = r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/faces"
# print(facesPath)
facesList = os.listdir(facesPath)
print(facesList)
facesArray = []
for i in range(len(facesList)):
    if "ds" not in facesList[i].lower():
        face = np.load(facesPath+'/'+facesList[i])
        face = face.reshape(face.shape[0],-1)
        facesArray.append(face)

facesArray = np.asarray(facesArray)
facesArray = np.vstack(facesArray)

userName = {}
for i in range(len(facesList)):
    name = facesList[i].split(".")[0]
    userName[i] = name

labels = np.zeros((facesArray.shape[0], 1))
n = len(facesArray) // len(facesList)
for i in range(len(facesList)):
    labels[i*n:,:] = float(i)
# print(labels)

def distance(x2,x1):
    return np.sqrt(sum((x1 - x2) ** 2))

def knn(x,train,k=5):
    n = train.shape[0]
    d = []
    for i in range(n):
        d.append(distance(x,train[i]))
    d = np.asarray(d)
    indexes = np.argsort(d)
    sortedLabels = labels[indexes][:k]
    count = np.unique(sortedLabels, return_counts=True)
    return count[0][np.argmax(count[1])]

font = cv2.FONT_HERSHEY_COMPLEX
dataset = cv2.CascadeClassifier('/Users/umesh/Desktop/Machine Learning 2/face_recognation_1/data.xml')
capture = cv2.VideoCapture(0)
result_name=[]
iter=0
while True:
    ret,img = capture.read()
    if ret and iter<50:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        newFace = dataset.detectMultiScale(gray)
        # print(faces)
       
        for x, y, w, h in newFace:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (50,50))
            label = knn(face.flatten(), facesArray)
            name = userName[int(label)]
            cv2.putText(img, name, (x,y), font, 1, (255, 0, 0), 2)
            result_name.append(name)
        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
        iter=iter+1
        if iter==50:
            cv2.destroyAllWindows()
            capture.release()
            break
    else:
        print("Camera not working")


print(len(result_name),result_name.count("user_1"))
if len(result_name)==0:
    result_name.append('none')

confidence=(result_name.count("user_1")/len(result_name))*100
print(confidence)
if confidence>80:
    print("unlock")
else:
    print("lock")