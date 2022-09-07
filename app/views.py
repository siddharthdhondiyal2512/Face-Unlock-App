from django.shortcuts import render
import cv2
import numpy as np
import os

def index(request):
    return render(request,'homepage.html')


# def predict(request):
    # name=request.GET['FullName']
    # email=request.GET['Email']
    # return render(request,"hello.html")


def predict(request):

    name=request.GET['FullName']
    email=request.GET['Email']

    
    email_file=open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/email.txt",'a')
    email_file.writelines(str(email))
    email_file.writelines("\n")
    email_file.close()
    name_file=open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/name.txt",'a')
    name_file.writelines(str(name))
    name_file.writelines("\n")
    name_file.close()

    # password1=open("email.txt",'r')
    # p=password1.readlines()[5]
    # password1.close()
    # p=p.strip()
    # print(p)

    vid  = cv2.VideoCapture(0)
    dataset = cv2.CascadeClassifier("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/data.xml")
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
            if cv2.waitKey(1) == 27 or i == 50:
                break
        else:
            print("Camera Not Found")

    file_read = open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/user.txt",'r')
    i=int(file_read.read())
    np.save(f"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/faces/user_{i}.npy", np.array(face_list))
    vid.release()
    cv2.destroyAllWindows()
    i=i+1
    file_write = open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/user.txt",'w')
    file_write.write(str(i))   

    name=name.title()
    return render(request,'success.html',{'user_name':name})


def recognize(request):

    # name=request.GET['FullName']
    email=request.GET['Email']

    
    email_file= open(r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/email.txt", 'r')
    name_file= open(r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/name.txt", 'r')
    user_count=1
    while email_file:
        line  = email_file.readline()
        if line.startswith(email):
            break
        elif line=="":
            break
        else:
            user_count=user_count+1

    with open(r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/email.txt", 'r') as fp:
        no_of_users = sum(1 for line in fp)

    if(user_count==no_of_users+1):
        return render(request,"failure.html")  
         
    lines=name_file.readlines()
    name_1=lines[user_count-1]
        
     
    email_file.close()
    name_file.close()

    # name_file=open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/name.txt",'a')
    # name_file.writelines(str(name))
    # name_file.writelines("\n")
    # name_file.close()

    # password1=open("email.txt",'r')
    # p=password1.readlines()[5]
    # password1.close()
    # p=p.strip()
    # print(p)

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


    print(len(result_name),result_name.count(f"user_{user_count}"))
    if len(result_name)==0:
        result_name.append('none')

    confidence=(result_name.count(f"user_{user_count}")/len(result_name))*100
    print(confidence)
    if confidence>=50:
        print("unlock")
        name_1=name_1.title()
        return render(request,'welcome.html',{'user_name':name_1})
    else:
        print("lock")
        return render(request,"failure.html")



def login(request):
    return render(request,'login_page.html')

def sign(request):
    return render(request,'sign_up.html')

def home(request):
    return render(request,'homepage.html')
