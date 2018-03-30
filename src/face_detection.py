import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
        cv2.rectangle(color_img, (x,y), (x+w, y+h), (255,0,0), 2)

    # show the image
    cv2.imshow('cam_0', color_img)

    # work until q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
