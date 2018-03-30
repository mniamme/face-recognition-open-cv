import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# get the username of the new user
ID = raw_input('Enter the user ID: ')
name = raw_input('Enter the user name: ')

sample_no = 0

while True:
    
    # read a frame from the camera
    ret, frame = cap.read()

    # take a copy of the image
    color_img = frame.copy()
    
    # convert image to grayscale image
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    # draw rectangle around the face
    for (x, y, w, h) in faces:
        sample_no = sample_no +  1
        
        # save the face to the dataset folder
        cv2.imwrite("dataset/user_" + name + "_" + str(ID) + '_' + str(sample_no) + "_.jpg", gray_img[y:y+h, x:x+w])        
        cv2.rectangle(color_img, (x,y), (x+w, y+h), (255,0,0), 2)

        cv2.waitKey(100)
    # show the image
    cv2.imshow('cam_0', color_img)

    cv2.waitKey(1)
    if sample_no > 20:
        break
        
    
print 'End of creating dataset'
cap.release()
cv2.destroyAllWindows()
