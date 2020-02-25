import cv2
import numpy as np
import dlib
import math



cap = cv2.VideoCapture("videos/video_1.hevc")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def euclidean(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    d = math.sqrt( (x2-x1)**2 +(y2-y1)**2  )
    return d

def aspectRatio(eye):

    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])

    ratio = ( A + B ) / ( 2 * C )

    return ratio
    
def detectDrowsiness(eyes):

    leftEye = eyes[0:6]
    rightEye = eyes[6:]

    leftRatio = aspectRatio(leftEye)
    rightRatio = aspectRatio(rightEye)

    return (leftRatio, rightRatio)

def detectHeadPose(image_points):

    size = (800,800)
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
        
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return (p1,p2)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame,(800,800))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        landmarks = predictor(gray,face)
        eyes = []
        for i in range(36,48):

            x = landmarks.part(i).x
            y = landmarks.part(i).y
            eyes.append((x,y))
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        leftRatio, rightRatio = detectDrowsiness(eyes)
        #print(leftRatio,rightRatio)

        image_points = np.array([
                            (landmarks.part(33).x, landmarks.part(33).y),     # Nose tip
                            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
                            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
                            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corne
                            (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
                            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
                        ], dtype="double")

        
        p1,p2 = detectHeadPose(image_points)
        cv2.line(frame, p1, p2, (255,0,0), 2)
        theta = math.atan( (p2[1] - p1[1]) / ( p2[0] - p1[0] ) )
        theta = theta*180/3.14
        #print(theta)

        font = cv2.FONT_HERSHEY_SIMPLEX 
        if theta >0 and theta < 20 :
            cv2.putText(frame, 'Attentive', (50,50), font,  
                   1, (255,255,255), 3, cv2.LINE_AA)

        else :
            cv2.putText(frame, 'InAttentive', (50,50), font,  
                   1, (255,255,255), 3, cv2.LINE_AA)


        if leftRatio <0.25 and rightRatio < 0.25 :
            cv2.putText(frame, 'Drowsy', (50,100), font,  
                   1, (255,255,255), 3, cv2.LINE_AA)

        else :
            cv2.putText(frame, 'Non Drowsy', (50,100), font,  
                   1, (255,255,255), 3, cv2.LINE_AA)  
        
    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)
    if key==27:
       cv2.destroyAllWindows()
       break


cap.release()
