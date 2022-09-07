# open file in read mode

email="shubham@gmail..com"

# email_file=open("/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/email.txt",'r')
# email_file.readlines(str(email))
# email_file.writelines("\n")
# email_file.close()

myfile= open(r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/email.txt", 'r')
myfile_1= open(r"/Users/umesh/Desktop/Machine Learning 2/face_recognition_app/FaceApp/app/static/name.txt", 'r')
    # for count, line in enumerate(fp):
    #     if(line==email):
    #         print(count)
# print('email is at line no. :', count + 1)
i=1
while myfile:
    line  = myfile.readline()
    if line.startswith(email):
        break
    elif line=="":
        break
    else:
        i=i+1

lines=myfile_1.readlines()
name=lines[i-1]

myfile.close()
myfile_1.close()


print(i)  
print(name)  

