import cv2
import mediapipe as mp
import math
import numpy as np
#import matplotlib.pyplot as plt

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points using the law of cosines
    """
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

angleslist=np.empty((96,15))
# Load video
video_path = [r'C:\Users\psaar\Documents\biceps\b2\b201.mp4', r'C:\Users\psaar\Documents\biceps\b2\b202.mp4', r'C:\Users\psaar\Documents\biceps\b2\b203.mp4', r'C:\Users\psaar\Documents\biceps\b2\b204.mp4', r'C:\Users\psaar\Documents\biceps\b2\b205.mp4', r'C:\Users\psaar\Documents\biceps\b2\b206.mp4', r'C:\Users\psaar\Documents\biceps\b2\b207.mp4', r'C:\Users\psaar\Documents\biceps\b2\b208.mp4', r'C:\Users\psaar\Documents\biceps\b2\b209.mp4', r'C:\Users\psaar\Documents\biceps\b2\b210.mp4', r'C:\Users\psaar\Documents\biceps\b2\b211.mp4', r'C:\Users\psaar\Documents\biceps\b2\b212.mp4', r'C:\Users\psaar\Documents\biceps\b2\b213.mp4', r'C:\Users\psaar\Documents\biceps\b2\b214.mp4', r'C:\Users\psaar\Documents\biceps\b2\b215.mp4', r'C:\Users\psaar\Documents\biceps\b2\b216.mp4', r'C:\Users\psaar\Documents\biceps\b2\b217.mp4', r'C:\Users\psaar\Documents\biceps\b2\b218.mp4', r'C:\Users\psaar\Documents\biceps\b2\b219.mp4', r'C:\Users\psaar\Documents\biceps\b2\b220.mp4', r'C:\Users\psaar\Documents\biceps\b2\b221.mp4', r'C:\Users\psaar\Documents\biceps\b2\b222.mp4', r'C:\Users\psaar\Documents\biceps\b2\b223.mp4', r'C:\Users\psaar\Documents\biceps\b2\b224.mp4', r'C:\Users\psaar\Documents\biceps\b2\b225.mp4', r'C:\Users\psaar\Documents\biceps\b2\b226.mp4', r'C:\Users\psaar\Documents\biceps\b2\b227.mp4', r'C:\Users\psaar\Documents\biceps\b2\b228.mp4', r'C:\Users\psaar\Documents\biceps\b2\b229.mp4', r'C:\Users\psaar\Documents\biceps\b2\b230.mp4', r'C:\Users\psaar\Documents\biceps\b2\b231.mp4', r'C:\Users\psaar\Documents\biceps\b2\b232.mp4', r'C:\Users\psaar\Documents\biceps\b2\b233.mp4', r'C:\Users\psaar\Documents\biceps\b3\b301.mp4', r'C:\Users\psaar\Documents\biceps\b3\b302.mp4', r'C:\Users\psaar\Documents\biceps\b3\b303.mp4', r'C:\Users\psaar\Documents\biceps\b3\b304.mp4', r'C:\Users\psaar\Documents\biceps\b3\b305.mp4', r'C:\Users\psaar\Documents\biceps\b3\b306.mp4', r'C:\Users\psaar\Documents\biceps\b3\b307.mp4', r'C:\Users\psaar\Documents\biceps\b3\b308.mp4', r'C:\Users\psaar\Documents\biceps\b3\b309.mp4', r'C:\Users\psaar\Documents\biceps\b3\b310.mp4', r'C:\Users\psaar\Documents\biceps\b3\b311.mp4', r'C:\Users\psaar\Documents\biceps\b3\b312.mp4', r'C:\Users\psaar\Documents\biceps\b3\b313.mp4', r'C:\Users\psaar\Documents\biceps\b3\b314.mp4', r'C:\Users\psaar\Documents\biceps\b3\b315.mp4', r'C:\Users\psaar\Documents\biceps\b3\b316.mp4', r'C:\Users\psaar\Documents\biceps\b3\b317.mp4', r'C:\Users\psaar\Documents\biceps\b3\b318.mp4', r'C:\Users\psaar\Documents\biceps\b3\b319.mp4', r'C:\Users\psaar\Documents\biceps\b3\b320.mp4', r'C:\Users\psaar\Documents\biceps\b3\b321.mp4', r'C:\Users\psaar\Documents\biceps\b3\b322.mp4', r'C:\Users\psaar\Documents\biceps\b3\b323.mp4', r'C:\Users\psaar\Documents\biceps\b3\b324.mp4', r'C:\Users\psaar\Documents\biceps\b3\b325.mp4', r'C:\Users\psaar\Documents\biceps\b3\b326.mp4', r'C:\Users\psaar\Documents\biceps\b3\b327.mp4', r'C:\Users\psaar\Documents\biceps\b3\b328.mp4', r'C:\Users\psaar\Documents\biceps\b3\b329.mp4', r'C:\Users\psaar\Documents\biceps\g\g10.mp4', r'C:\Users\psaar\Documents\biceps\g\g11.mp4', r'C:\Users\psaar\Documents\biceps\g\g12.mp4', r'C:\Users\psaar\Documents\biceps\g\g13.mp4', r'C:\Users\psaar\Documents\biceps\g\g14.mp4', r'C:\Users\psaar\Documents\biceps\g\g15.mp4', r'C:\Users\psaar\Documents\biceps\g\g16.mp4', r'C:\Users\psaar\Documents\biceps\g\g17.mp4', r'C:\Users\psaar\Documents\biceps\g\g18.mp4', r'C:\Users\psaar\Documents\biceps\g\g19.mp4', r'C:\Users\psaar\Documents\biceps\g\g20.mp4', r'C:\Users\psaar\Documents\biceps\g\g21.mp4', r'C:\Users\psaar\Documents\biceps\g\g22.mp4', r'C:\Users\psaar\Documents\biceps\g\g23.mp4', r'C:\Users\psaar\Documents\biceps\g\g24.mp4', r'C:\Users\psaar\Documents\biceps\g\g25.mp4', r'C:\Users\psaar\Documents\biceps\g\g26.mp4', r'C:\Users\psaar\Documents\biceps\g\g27.mp4', r'C:\Users\psaar\Documents\biceps\g\g28.mp4', r'C:\Users\psaar\Documents\biceps\g\g29.mp4', r'C:\Users\psaar\Documents\biceps\g\g30.mp4', r'C:\Users\psaar\Documents\biceps\g\g31.mp4', r'C:\Users\psaar\Documents\biceps\g\g32.mp4', r'C:\Users\psaar\Documents\biceps\g\g33.mp4', r'C:\Users\psaar\Documents\biceps\g\g34.mp4', r'C:\Users\psaar\Documents\biceps\g\g35.mp4', r'C:\Users\psaar\Documents\biceps\g\g36.mp4', r'C:\Users\psaar\Documents\biceps\g\g37.mp4', r'C:\Users\psaar\Documents\biceps\g\g38.mp4', r'C:\Users\psaar\Documents\biceps\g\g39.mp4', r'C:\Users\psaar\Documents\biceps\g\g40.mp4', r'C:\Users\psaar\Documents\biceps\g\g41.mp4', r'C:\Users\psaar\Documents\biceps\g\g42.mp4', r'C:\Users\psaar\Documents\biceps\g\g43.mp4']

for numv in range(96):
    cap = cv2.VideoCapture(video_path[numv])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = 15
    frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (num_frames * 1.0))
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*frame_interval)
        ret, frame = cap.read()
        frames.append(frame)

    # Initialize MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        #angles = []  # List to store angles over time

        for i in range(num_frames):

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frames[i], 1)

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe Pose
            results = pose.process(image)

            # Draw pose landmarks on the image
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmark coordinates for left elbow, shoulder, and hip
            if results.pose_landmarks is not None:
                left_elbow = (results.pose_landmarks.landmark[14].x * frame.shape[1],
                              results.pose_landmarks.landmark[14].y * frame.shape[0])
                left_shoulder = (results.pose_landmarks.landmark[12].x * frame.shape[1],
                                 results.pose_landmarks.landmark[12].y * frame.shape[0])
                left_hip = (results.pose_landmarks.landmark[24].x * frame.shape[1],
                            results.pose_landmarks.landmark[24].y * frame.shape[0])

                # Calculate the angle between left elbow, shoulder, and hip
                angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                #angles.append(angle)
                angleslist[numv][i]=angle

                # Display the angle on the frame
                #cv2.putText(image, f'Angle: {angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            #cv2.imshow('Video', image)

            # Exit when 'q' is pressed
            #if cv2.waitKey(0) & 0xFF == ord('q'):
            #    break
            

        # Release the video capture and close the window
        cap.release()
        cv2.destroyAllWindows()
        
np.save(r'C:\Users\psaar\Documents\biceps\angles.npy', angleslist) 
    #plt.plot(angles)
    #plt.xlabel('Frame')
    #plt.ylabel('Angle (degrees)')
    #plt.title('Angle over Time')
    #plt.show()