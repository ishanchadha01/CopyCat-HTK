import cv2
import mediapipe as mp
import time


def draw_mediapipe_landmarks(video_filepath="/home/ishan/Documents/research/ccg/copycat/DATA/input/Jinghong/alligator_above_bed/1612482965/Jinghong.alligator_above_bed.1612482965.mp4", \
  new_video_filepath="/home/ishan/Documents/research/ccg/elan/landmarks.mp4"):

  mp_drawing = mp.solutions.drawing_utils
  mp_holistic = mp.solutions.holistic

  # For video input:
  holistic = mp_holistic.Holistic(
      min_detection_confidence=0.5, min_tracking_confidence=0.1)
  cap = cv2.VideoCapture(video_filepath)
  result = cv2.VideoWriter(new_video_filepath, 
                         cv2.VideoWriter.fourcc(*"mp4v"),
                         10, (int(cap.get(3)), int(cap.get(4))))
  start = time.time()
  num_frames = 0
  pose_null = 0
  while cap.isOpened():
    success, image = cap.read()
    num_frames += 1
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # result.write(image)
    # continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.pose_landmarks is None:
      pose_null += 1

    #define the screen resulation
    # screen_res = 1280, 720
    # scale_width = screen_res[0] / image.shape[1]
    # scale_height = screen_res[1] / image.shape[0]
    # scale = min(scale_width, scale_height)
    #resized window width and height
    # window_width = int(image.shape[1] * scale)
    # window_height = int(image.shape[0] * scale)
    #cv2.WINDOW_NORMAL makes the output window resizealbe
    window_width = int(cap.get(3))
    window_height = int(cap.get(4))
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    #resize the window according to the screen resolution
    cv2.resizeWindow('Resized Window', window_width, window_height)
    cv2.imshow('Resized Window', image)
    result.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
  
  
  end = time.time() - start
  print("Number of times pose is none = " + str(pose_null))
  print("Time taken = " + str(end))
  print("Total frames = " + str(num_frames))
  print("Frames processed per second = " + str(num_frames/end))
  holistic.close()
  cap.release()
  result.release()

draw_mediapipe_landmarks()