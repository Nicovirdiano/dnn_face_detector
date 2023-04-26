import cv2
import numpy as np
# -----read
prototxt = "C:\\Users\\cecil\\Desktop\\dnn_face_detector\\model\\deploy.prototxt.txt"
model = "C:\\Users\\cecil\\Desktop\\dnn_face_detector\\model\\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Read and preprocessin image
 #cap = cv2.VideoCapture("C:\\Users\\cecil\\Desktop\\dnn_face_detector\\images_videos\\Cecilia.mp4") 
       
cap = cv2.VideoCapture(0)
        
cv2.namedWindow("Frame", cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("Frame", 300, 300)
        
         
while True:
      ret, frame = cap.read()
      if ret == False:
           break
           
            
      height, width, _ = frame.shape
      frame_resized = cv2.resize(frame, (300, 300))

# create a blob

      blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 117, 123))
 


# detectins and prediction
      net.setInput(blob)
      detections = net.forward()
      #print("detections.shape:", detections.shape)

      for detection in detections[0][0]:
    #print("detection:",detection)
        if detection[2] > 0.5:
           box = detection[3:7] * [width, height, width, height]
           x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
           cv2.rectangle(frame,(x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
           cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 10), 3, 1.2, (0, 255, 255),2)

     # rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
      cv2.imshow ("Frame", frame) 
      if cv2.waitKey(1) & 0xFF == 27:
            break
      
cv2.waitKey(1)
cv2.destroyAllWindows()
