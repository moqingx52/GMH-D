"""
Copyright (c) 2023. Code developed by Gianluca Amprimo, PhD. student at Politecnico di Torino, Italy.
If you use this code for research, please cite:

 <G. Amprimo, C. Ferraris, G. Masi, G. Pettiti and L. Priano, "GMH-D: Combining Google MediaPipe and RGB-Depth
 Cameras for Hand Motor Skills Remote Assessment," 2022 IEEE International Conference on Digital Health (ICDH),
 Barcelona, Spain, 2022, pp. 132-141, doi: 10.1109/ICDH55609.2022.00029.>

 <Amprimo, Gianluca, et al. "Hand tracking for clinical applications: validation of the google mediapipe hand (gmh) and
 the depth-enhanced gmh-d frameworks." arXiv preprint arXiv:2308.01088 (2023).>

Script for running GMH-D handtracking algorithm with Azure Kinect or RGB-D cameras supporting Azure Kinect SDK (ORBBEC FEMTO cameras)
"""
import os
import time
import wget
import jsonpickle
import click as click
import cv2
import mediapipe as mp
import pyk4a
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from collections import OrderedDict
import json
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from pyk4a import *

#Setup all mediapipe components needed for running the code
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode



def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    """
    Method to convert the image format of the color stream from Kinect to BGR (as required by body tracking Kinect
    and MediaPipe
    :param color_format: instance of ImageFormat enumerator from Azure Kinect SDK
    :param color_image: the image read from Kinect to convert using opencv
    :return: the converted image
    """
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image

#for data saving
TRACKING_DATA=[]

@click.command()
@click.option("--mode", default="online", help="Processing mode (online or offline)")
@click.option("--mkvfilepath", default="-1", help="Path to folder containing mkv file")
@click.option("--mkvfilename", default="-1", help="MKV filename")
@click.option("--save", default="yes", help="Save tracked joints (yes or no)")
@click.option("--outputpath", default='.', help="Absolute path to output folder (only if --save is yes)")
@click.option("--outputname", default='tracking_data', help="Name for tracking output json file (only if --save is yes")
@click.option("--n_hands", default=2, help="Number of hands to track (>=1)")
@click.option("--handconf", default=0.5, help="Confidence threshold for hand tracking [0,1]")
@click.option("--rerun_pd", default=0.2, help="Confidence of detection before rerunning Palm Detector [0,1]")
@click.option("--jointconf", default=0.5, help="Confidence threshold for joint tracking [0,1]")
@click.option("--interval", default=10, help="Set >0 for automatically recording t seconds [1, +inf]")
@click.option("--visualize", default='yes', help="Visualize tracking while processing video (yes/no)")



def main(**cfg):
    """
    Main method manages the type of tracking (online or offline) and setups the environment.
    :param cfg: dictionary containing the input arguments from command line
    :return: none
    """
    # Define the URL of the file
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    # Define the directory where you want to save the file
    directory = os.path.join(".", "MP_model")

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the path where you want to save the file
    filename = os.path.join(directory, "hand_landmarker.task")
    # Check if the file already exists
    if not os.path.exists(filename):
        # If the file doesn't exist, download it
        print("Downloading file...")
        wget.download(url, out=directory)
        print("\nDownload complete.")
    else:
        print("File already exists.")
    cfg['model_asset_path'] = filename


    if cfg['mode']=='offline':
       if cfg['mkvfilepath']=="-1" or cfg['mkvfilename']=='-1':
           print("Offline mode, but file path or file name not specified. Use --mkvfilepath option to specify the path and --mkfilename the name of mkv file to process")
           return -1
       if not os.path.exists(os.path.join(cfg['mkvfilepath'], cfg['mkvfilename'])):
           print("The specified file does not exist.") 
           return -1
       offline_tracking(cfg)
    else:
       online_tracking(cfg)


class GMHDLandmark:
    def __init__(self, point3D, visibility, presence):
        self.x=point3D[0]
        self.y=point3D[1]
        self.z=point3D[2]
        self.visibility=visibility
        self.presence=presence

class GMHDHand:
    def __init__(self, gmhd_joints, handedness):
        self.joints=gmhd_joints
        self.handedness=handedness


class Frame:
    def __init__(self, timestamp, hands_gmhd_list):
        self.timestamp=timestamp
        self.hands=hands_gmhd_list

def GMHD_estimation(hand_landmarks, depth_image):
    """
    Method to estimate the GMHD coordinates of handlandmarks detected by MediaPipe
    :param hand_landmarks: NormalizedHandLandmarks for a single detected hand
    :param depth_image: dpeth image to retrieve depth of wrist estimated by ToF sensor
    :return:
    """
    frameWidth=depth_image.shape[1]
    frameHeigth=depth_image.shape[0]
    frameHeigth
    wrist_depth_tof=0
    joint_list=[]
    for joint_name in solutions.hands.HandLandmark:
        point=hand_landmarks[joint_name]
        pixelCoordinatesLandmark = solutions.drawing_utils._normalized_to_pixel_coordinates(
            point.x,
            point.y,
            frameWidth,
            frameHeigth)
        # if point is wrist, we get the tof estimation of the joint to compute GMHD
        if joint_name==solutions.hands.HandLandmark.WRIST:
            wrist_depth_tof= depth_image[pixelCoordinatesLandmark[1], pixelCoordinatesLandmark[0]] #axis are reversed for retrieving from depth map
            depth_estimated= wrist_depth_tof
            # Convert the depth image to RGB for visualization
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_INFERNO)
        else:
            depth_estimated= wrist_depth_tof + wrist_depth_tof * point.z #GMH-D depth estimation for all other joints
        try:
            point_3D= k4a.calibration.convert_2d_to_3d([int(pixelCoordinatesLandmark[0]), int(pixelCoordinatesLandmark[1])], int(depth_estimated),
                                               CalibrationType.COLOR, CalibrationType.COLOR)
        except Exception as e:
            print(e)
            print(f'3D conversion failed: are {wrist_depth_tof} and {depth_estimated} from depthmap invalid? Appending None')
            point_3D = (None, None, None)
        gmhd_point=GMHDLandmark(point_3D, point.visibility, point.presence)
        joint_list.append(gmhd_point)
    return joint_list

def process_sync_tracking(detection_result: HandLandmarkerResult, bgr_image, depth_image, timestamp_ms: int):
  """

  :param detection_result: HandLandmarkerResult object from MediaPipe inference
  :param bgr_image: BGR image for visualization of landmarks
  :param depth_image: depth image to estimate Wrist depth as reference
  :param timestamp_ms: timestamp of data acquisition
  :return:
  """
  try:
      if detection_result is None or detection_result.handedness==[]:
          return
      MARGIN = 10  # pixels
      FONT_SIZE = 1
      FONT_THICKNESS = 1
      HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

      hand_landmarks_list = detection_result.hand_landmarks

      handedness_list = detection_result.handedness
      annotated_image=np.copy(cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR))

      # Loop through the detected hands to visualize and get GMH-D coordinates
      hands_GMHD=[]
      for idx in range(len(hand_landmarks_list)):

        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # apply GMHD and save tracking
        joints_GMHD =GMHD_estimation(hand_landmarks, depth_image)
        hands_GMHD.append(GMHDHand(joints_GMHD, handedness))
        # Draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        cv2.flip(annotated_image, 1)

  except Exception as e:
    print("GMHD computation failed", e)
  cv2.imshow('Mediapipe tracking',annotated_image)
  cv2.waitKey(5)  # altrimenti non visualizza l'immagine
  TRACKING_DATA.append(Frame(timestamp_ms, hands_GMHD))

def online_tracking(cfg):
    """
    Method to run GMH-D online, by processing input from a connected Azure Kinect device (only one device at a time)
    :param cfg: dictionary containing all the input configuration from command line execution
    """
    #setup mediapipe tracking
    print("----GMH-D tracking ACTIVED----")
    print("Instantiating tracking utilities..")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg['model_asset_path']),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=cfg['n_hands'], min_hand_detection_confidence=cfg['handconf'], min_hand_presence_confidence=cfg['rerun_pd'],
        min_tracking_confidence=cfg['jointconf'])
    detector= HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    #open Azure Kinect
    global k4a
    k4a= PyK4A(Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                color_format=pyk4a.ImageFormat.COLOR_BGRA32,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            ))
    k4a.start()
    # calculate FPS
    previousTime_FPS = 0
    currentTime_FPS = 0
    startTime = time.time()
    currentTime = 0

    while True:
        # Get the next capture (blocking function)
        capture = k4a.get_capture()
        img_color = capture.color
        depth_image = capture.transformed_depth  # depth trasformata in color
        if img_color is not None and depth_image is not None:
            color_timestamp = capture.color_timestamp_usec
            rgb_image = cv2.cvtColor(img_color, cv2.COLOR_BGRA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            # STEP 4: Detect hand landmarks from the input image.
            detection_result=detector.detect_for_video(mp_image, color_timestamp)
            currentTime = time.time()
            process_sync_tracking(detection_result, img_color, depth_image, color_timestamp)
            if (previousTime_FPS > 0):
              # Calculating the fps
              fps = (1 / (color_timestamp - previousTime_FPS))*1e6
              print("Frame rate: ", fps)
            previousTime_FPS = color_timestamp
        else:
            print("Impossible to retrieve color or depth frame from camera")
        if 0xFF == 27:
            print("Tracking interrupted!")
            cv2.destroyAllWindows()
            break
        # stop execution after --interval set seconds
        if  ((currentTime - startTime) > cfg['interval']):
            print("Tracking recording completed")
            cv2.destroyAllWindows()
            break
    if cfg['save']=='yes':
        save_tracking_data(os.path.join(cfg['outputpath'], cfg['outputname']))

def save_tracking_data(filepath):
    """
    Method to save GMH-D tracking data as a json dump file.
    :param filepath: output path where to dump the tracking data
    """
    data = {
        "code_version": "GMHD_1.0",
        "recorded_tracking": jsonpickle.encode(TRACKING_DATA)
    }

    # Serialize data to JSON
    with open(filepath, 'w') as fp:
        data=json.dumps(data, indent=4)
        fp.write(data)

def offline_tracking(cfg):
    """
       Method to run GMH-D offline, by processing an input mkv file obtained by using Azure Kinect recording utilities
       :param cfg: dictionary containing all the input configuration from command line execution
       """
    frameWidth = 1920
    frameHeight = 1080

    print("----GMH-D post-processing of MKV ----")
    print("Instantiating tracking utilities..")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg['model_asset_path']),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=cfg['n_hands'], min_hand_detection_confidence=cfg['handconf'],
        min_hand_presence_confidence=cfg['rerun_pd'],
        min_tracking_confidence=cfg['jointconf'])
    detector = HandLandmarker.create_from_options(options)

    os.add_dll_directory("C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin")

    # PERFORM BODY TRACKING WITH KINECT AZURE
    # Initialize the library, if the library is not found, add the library path as argument
    playback = PyK4APlayback(os.path.join(cfg['mkvfilepath'], cfg['mkvfilename']))
    playback.open()
    global k4a
    k4a = playback
    # print(playback_config)
    # calculate FPS
    previousTime_FPS = 0
    currentTime_FPS = 0
    startTime = time.time()
    currentTime = 0

    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None and capture.depth is not None:
                img_color = convert_to_bgra_if_required(playback.configuration['color_format'],capture.color)
                depth_image = capture.transformed_depth  # depth trasformata in color
                color_timestamp = capture.color_timestamp_usec
                rgb_image = cv2.cvtColor(img_color, cv2.COLOR_BGRA2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                # STEP 4: Detect hand landmarks from the input image.
                detection_result = detector.detect_for_video(mp_image, color_timestamp)
                currentTime = time.time()
                process_sync_tracking(detection_result, img_color, depth_image, color_timestamp)
                if (previousTime_FPS > 0):
                    # Calculating the fps
                    fps = (1 / (color_timestamp - previousTime_FPS)) * 1e6
                    print("Frame rate: ", fps)
                previousTime_FPS = color_timestamp
            else:
                print("Impossible to retrieve color or depth frame from camera")
            # stop execution after --interval set seconds
            if ((currentTime - startTime) > cfg['interval']):
                print("Tracking recording completed")
                cv2.destroyAllWindows()
                break
        except EOFError:
            break
    if cfg['save'] == 'yes':
        save_tracking_data(os.path.join(cfg['outputpath'], cfg['outputname']))
    print("Extraction of tracking data completed")

if __name__ == "__main__":
    main()


