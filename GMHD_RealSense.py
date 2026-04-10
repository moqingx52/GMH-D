"""
Copyright (c) 2023. Code developed by Gianluca Amprimo, PhD. student at Politecnico di Torino, Italy.
If you use this code for research, please cite:

 <G. Amprimo, C. Ferraris, G. Masi, G. Pettiti and L. Priano, "GMH-D: Combining Google MediaPipe and RGB-Depth
 Cameras for Hand Motor Skills Remote Assessment," 2022 IEEE International Conference on Digital Health (ICDH),
 Barcelona, Spain, 2022, pp. 132-141, doi: 10.1109/ICDH55609.2022.00029.>

 <Amprimo, Gianluca, et al. "Hand tracking for clinical applications: validation of the google mediapipe hand (gmh) and
 the depth-enhanced gmh-d frameworks." arXiv preprint arXiv:2308.01088 (2023).>

Script for running GMH-D handtracking algorithm with RGB-D camera from Intel Realsense D4xx family
"""
import os
import time
import wget
import click as click
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import pyrealsense2 as rs

from tracking_export import save_xr_teleop_tracking_json, build_tracking_debug_stats, print_tracking_debug_stats

#Setup all mediapipe components needed for running the code
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
TRACKING_DATA=[]


def _as_bool_yes(value):
    return str(value).strip().lower() in {"yes", "y", "true", "1", "on"}


def _colorize_depth_for_display(depth_image):
    if depth_image is None:
        return None
    depth = np.asarray(depth_image)
    if depth.size == 0:
        return None
    valid = depth[np.isfinite(depth)]
    valid = valid[valid > 0]
    if valid.size == 0:
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    near = float(np.percentile(valid, 2))
    far = float(np.percentile(valid, 98))
    if far <= near:
        far = near + 1.0
    depth_clipped = np.clip(depth, near, far)
    depth_norm = ((depth_clipped - near) / (far - near) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)


def _show_capture_preview(color_bgra, depth_image, window_name="RealSense capture preview"):
    color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
    depth_viz = _colorize_depth_for_display(depth_image)
    if depth_viz is None:
        preview = color_bgr
    else:
        if depth_viz.shape[:2] != color_bgr.shape[:2]:
            depth_viz = cv2.resize(depth_viz, (color_bgr.shape[1], color_bgr.shape[0]))
        preview = np.hstack([color_bgr, depth_viz])
    cv2.imshow(window_name, preview)
    return cv2.waitKey(1) & 0xFF


def _record_then_process_bag(cfg):
    output_bag = (cfg.get("record_bag_path") or "").strip()
    if not output_bag:
        output_bag = os.path.join(cfg["outputpath"], f"{cfg['outputname']}.bag")
    output_bag = os.path.abspath(output_bag)
    os.makedirs(os.path.dirname(output_bag), exist_ok=True)

    print("----GMH-D capture stage ACTIVED----")
    print(f"Recording RealSense bag to: {output_bag}")
    print("Preview: left RGB, right depth. Press ESC to stop recording early.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgra8, 30)
    config.enable_record_to_file(output_bag)

    recorded_frames = 0
    start_time = time.time()
    interrupted = False
    try:
        pipeline.start(config)
        while True:
            try:
                capture = pipeline.wait_for_frames(5000)
            except RuntimeError as e:
                print(f"RealSense frame timeout during recording: {e}")
                break

            color_frame = capture.get_color_frame()
            depth_frame = capture.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Incomplete RealSense frames received during recording")
                continue

            img_color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            if img_color is None or depth_image is None:
                continue

            recorded_frames += 1
            if _as_bool_yes(cfg.get("visualize", "yes")):
                key = _show_capture_preview(img_color, depth_image)
                if key == 27:
                    interrupted = True
                    print("Recording interrupted by user")
                    break

            if cfg['interval'] > 0 and (time.time() - start_time) > cfg['interval']:
                print("Tracking recording completed")
                break
    finally:
        pipeline.stop()
        cv2.destroyWindow("RealSense capture preview")

    elapsed = max(time.time() - start_time, 1e-6)
    print(f"Capture summary: {recorded_frames} frames, elapsed {elapsed:.2f}s, avg FPS {recorded_frames / elapsed:.2f}")

    offline_cfg = dict(cfg)
    offline_cfg['bagfilepath'] = os.path.dirname(output_bag)
    offline_cfg['bagfilename'] = os.path.basename(output_bag)
    offline_cfg['mode'] = 'offline'
    offline_cfg['record_interrupted'] = interrupted
    print("----GMH-D processing recorded bag----")
    offline_tracking(offline_cfg)

def convert_to_bgra_if_required(input_format, input_data):
    """
    Method to convert the image format of the color stream from Kinect to BGR (as required by body tracking Kinect
    and MediaPipe
    :param color_format: instance of ImageFormat enumerator from Azure Kinect SDK
    :param color_image: the image read from Kinect to convert using opencv
    :return: the converted image
    """
    if input_format == 'YUYV':
        # Convert YUYV to BGRA
        bgra = cv2.cvtColor(input_data, cv2.COLOR_YUV2BGRA)
    elif input_format == 'RGB8':
        # Convert RGB8 to BGRA
        bgra = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGRA)
    elif input_format == 'BGR8':
        # Convert BGR8 to BGRA
        bgra = cv2.cvtColor(input_data, cv2.COLOR_BGR2BGRA)
    elif input_format == 'RGBA8':
        # Convert RGBA8 to BGRA
        bgra = cv2.cvtColor(input_data, cv2.COLOR_RGBA2BGRA)
    elif input_format == 'BGRA8':
        # No need to convert, already in BGRA format
        bgra = input_data
    elif input_format == 'Y8':
        # Convert Y8 to BGRA
        bgra = cv2.cvtColor(cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    elif input_format == 'Y16':
        # Convert Y16 to BGRA
        bgra = cv2.cvtColor(cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    elif input_format == 'RAW16':
        # Convert RAW16 to BGRA
        bgra = cv2.cvtColor(cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2BGRA)
    else:
        raise ValueError("Unsupported input format")

    return bgra

@click.command()
@click.option("--mode", default="offline", help="Processing mode (online or offline)")
@click.option("--bagfilepath", default="-1", help="Path to folder containing mkv file")
@click.option("--bagfilename", default="-1", help="MKV filename")
@click.option("--save", default="yes", help="Save tracked joints (yes or no)")
@click.option("--outputpath", default='.', help="Absolute path to output folder (only if --save is yes)")
@click.option("--outputname", default='tracking_data', help="Name for tracking output json file (only if --save is yes")
@click.option("--n_hands", default=2, help="Number of hands to track (>=1)")
@click.option("--handconf", default=0.5, help="Confidence threshold for hand tracking [0,1]")
@click.option("--rerun_pd", default=0.2, help="Confidence of detection before rerunning Palm Detector [0,1]")
@click.option("--jointconf", default=0.5, help="Confidence threshold for joint tracking [0,1]")
@click.option("--interval", default=10, help="Set >0 for automatically recording t seconds [1, +inf]")
@click.option("--visualize", default='yes', help="Visualize tracking while processing video (yes/no)")
@click.option(
    "--online-workflow",
    default="stream",
    help="在线模式流程：stream=边采集边跟踪，record_then_process=先录制并预览，再离线处理",
)
@click.option(
    "--record-bag-path",
    default="",
    help="online + record_then_process 时可选：录制 bag 输出路径；默认写到 outputpath/outputname.bag",
)
@click.option(
    "--cam2base-json",
    default="",
    help="可选：含 T_cam2base(4x4) 的 JSON，导出时同时写入 p_wrist_base/R_wrist_base（与 HaMeR/xr_teleoperate 约定一致）",
)
@click.option("--debug", is_flag=True, default=False, help="输出 GMH-D 调试统计信息")

def main(**cfg):
    """
    Main method manages the type of tracking (online or offline) and setups the environment.
    :param cfg: dictionary containing the input arguments from command line
    :return: none
    """
    TRACKING_DATA.clear()
    os.makedirs(cfg['outputpath'], exist_ok=True)
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
       if cfg['bagfilepath']=="-1" or cfg['bagfilename']=='-1':
           print("Offline mode, but file path or file name not specified. Use --bagfilepath option to specify the path and --bagfilename the name of bag file to process")
           return -1
       if not os.path.exists(os.path.join(cfg['bagfilepath'], cfg['bagfilename'])):
           print("The specified file does not exist.") 
           return -1
       offline_tracking(cfg)
    else:
       workflow = str(cfg.get('online_workflow', 'stream')).strip().lower()
       if workflow == 'record_then_process':
           _record_then_process_bag(cfg)
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


def _is_valid_depth_value(depth_value):
    try:
        d = float(depth_value)
    except Exception:
        return False
    return np.isfinite(d) and d > 0.0


def _sample_valid_depth_meters(depth_frame, px, py, max_radius=3, z_min=0.05, z_max=2.5):
    h = int(depth_frame.get_height())
    w = int(depth_frame.get_width())
    if px is None or py is None:
        return None
    px = int(px)
    py = int(py)
    if px < 0 or py < 0 or px >= w or py >= h:
        return None

    center = float(depth_frame.get_distance(px, py))
    if _is_valid_depth_value(center):
        if z_min < center < z_max:
            return float(center)

    for radius in range(1, int(max_radius) + 1):
        x0 = max(0, px - radius)
        x1 = min(w - 1, px + radius)
        y0 = max(0, py - radius)
        y1 = min(h - 1, py + radius)
        vals = []
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                d = float(depth_frame.get_distance(xx, yy))
                if z_min < d < z_max and np.isfinite(d):
                    vals.append(d)
        vals = np.asarray(vals, dtype=np.float32)
        if vals.size > 0:
            return float(np.median(vals))
    return None


def _is_valid_point3d(point_3d):
    try:
        arr = np.asarray(point_3d, dtype=np.float64).reshape(3)
    except Exception:
        return False
    return np.all(np.isfinite(arr)) and float(np.linalg.norm(arr)) > 1e-9

def convert_depth_to_phys_coord_using_realsense(x, y, depth, intrinsics):
  result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)  #result[0]: right, result[1]: down, result[2]: forward
  return result[0], result[1], result[2]

def _palm_anchor_depth(hand_landmarks, frame_width, frame_height, depth_frame):
    anchor_ids = [
        solutions.hands.HandLandmark.WRIST,
        solutions.hands.HandLandmark.THUMB_CMC,
        solutions.hands.HandLandmark.INDEX_FINGER_MCP,
        solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
        solutions.hands.HandLandmark.RING_FINGER_MCP,
        solutions.hands.HandLandmark.PINKY_MCP,
    ]
    samples = []
    for joint_id in anchor_ids:
        p = hand_landmarks[joint_id]
        pix = solutions.drawing_utils._normalized_to_pixel_coordinates(
            p.x, p.y, frame_width, frame_height
        )
        if pix is None:
            continue
        d = _sample_valid_depth_meters(depth_frame, int(pix[0]), int(pix[1]), max_radius=3)
        if d is not None:
            samples.append(d)
    if len(samples) < 3:
        return None
    return float(np.median(np.asarray(samples, dtype=np.float32)))


def GMHD_estimation(hand_landmarks, depth_frame, cameraInfo):
    """
    Method to estimate the GMHD coordinates of handlandmarks detected by MediaPipe
    :param hand_landmarks: NormalizedHandLandmarks for a single detected hand
    :param depth_image: dpeth image to retrieve depth of wrist estimated by ToF sensor
    :return:
    """
    frameWidth = int(depth_frame.get_width())
    frameHeigth = int(depth_frame.get_height())
    frameHeigth
    palm_depth_tof = _palm_anchor_depth(hand_landmarks, frameWidth, frameHeigth, depth_frame)
    joint_list=[]
    for joint_name in solutions.hands.HandLandmark:
        point=hand_landmarks[joint_name]
        pixelCoordinatesLandmark = solutions.drawing_utils._normalized_to_pixel_coordinates(
            point.x,
            point.y,
            frameWidth,
            frameHeigth)
        point_3D = (None, None, None)
        if pixelCoordinatesLandmark is None:
            gmhd_point=GMHDLandmark(point_3D, point.visibility, point.presence)
            joint_list.append(gmhd_point)
            continue
        px, py = int(pixelCoordinatesLandmark[0]), int(pixelCoordinatesLandmark[1])
        depth_estimated = _sample_valid_depth_meters(depth_frame, px, py, max_radius=3)
        if depth_estimated is None and palm_depth_tof is not None:
            # Fallback to palm anchor depth + normalized relative z
            depth_estimated = palm_depth_tof * (1.0 + float(point.z))
            if not _is_valid_depth_value(depth_estimated):
                depth_estimated = None
        if depth_estimated is not None:
            try:
                point_3D= convert_depth_to_phys_coord_using_realsense(px, py, float(depth_estimated), cameraInfo)
                if not _is_valid_point3d(point_3D):
                    point_3D = (None, None, None)
            except Exception as e:
                print(e)
                print(f'3D conversion failed: are {palm_depth_tof} and {depth_estimated} from depthmap invalid? Appending None')
                point_3D = (None, None, None)
        gmhd_point=GMHDLandmark(point_3D, point.visibility, point.presence)
        joint_list.append(gmhd_point)
    return joint_list

def process_sync_tracking(detection_result: HandLandmarkerResult, bgr_image, depth_frame, timestamp_ms: int, cameraInfo, show_window=True):
  """

  :param detection_result: HandLandmarkerResult object from MediaPipe inference
  :param bgr_image: BGR image for visualization of landmarks
  :param depth_image: depth image to estimate Wrist depth as reference
  :param timestamp_ms: timestamp of data acquisition
  :return:
  """
  try:
      hands_GMHD=[]
      annotated_image=np.copy(cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR))
      if detection_result is None or detection_result.handedness==[]:
          if show_window:
              cv2.imshow('Mediapipe tracking', annotated_image)
              cv2.waitKey(1)
          TRACKING_DATA.append(Frame(timestamp_ms, hands_GMHD))
          return
      MARGIN = 10  # pixels
      FONT_SIZE = 1
      FONT_THICKNESS = 1
      HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

      hand_landmarks_list = detection_result.hand_landmarks

      handedness_list = detection_result.handedness

      # Loop through the detected hands to visualize and get GMH-D coordinates
      hands_GMHD=[]
      for idx in range(len(hand_landmarks_list)):

        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # apply GMHD and save tracking
        joints_GMHD =GMHD_estimation(hand_landmarks, depth_frame, cameraInfo)
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
  except Exception as e:
    print("GMHD computation failed", e)
  if show_window:
      cv2.imshow('Mediapipe tracking',annotated_image)
      cv2.waitKey(1)  # altrimenti non visualizza l'immagine
  TRACKING_DATA.append(Frame(timestamp_ms, hands_GMHD))

def online_tracking(cfg):
    """
    Method to run GMH-D online, by processing input from a connected Realsense D4XX device (only one device at a time)
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
    #open Realsense camera
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)

    # Configure streams
    config = rs.config()

    #NB: color and depth resolution can change but they should be equal (or one should be TRANSFORMED to match the other)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgra8, 30)

    pipeline.start(config)

    # calculate FPS
    previousTime_FPS = -1
    startTime = time.time()
    currentTime = 0
    consecutive_timeouts = 0

    try:
        processed_frames = 0
        fps_samples = []
        while True:
            loop_start_time = time.time()
            try:
                # Get the next capture (blocking function)
                capture = pipeline.wait_for_frames(5000)
                aligned_capture = align.process(capture)
                consecutive_timeouts = 0
            except RuntimeError as e:
                consecutive_timeouts += 1
                print(f"RealSense frame timeout ({consecutive_timeouts}): {e}")
                if consecutive_timeouts >= 3:
                    print("Tracking interrupted after repeated frame timeouts")
                    break
                continue

            color_frame = aligned_capture.get_color_frame()
            depth_frame = aligned_capture.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Incomplete RealSense frames received")
                continue
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            img_color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # depth trasformed in color
            if img_color is not None and depth_image is not None:
                color_timestamp = capture.get_timestamp()
                #captures may be asyncronously managed in recording, so we must ensure temporal consistency of consecutive frames
                if previousTime_FPS>=color_timestamp:
                    continue
                rgb_image = cv2.cvtColor(img_color, cv2.COLOR_BGRA2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                # STEP 4: Detect hand landmarks from the input image.
                detection_result=detector.detect_for_video(mp_image, int(color_timestamp))
                currentTime = time.time()
                process_sync_tracking(
                    detection_result,
                    img_color,
                    depth_frame,
                    color_timestamp,
                    depth_intrinsics,
                    show_window=_as_bool_yes(cfg.get('visualize', 'yes')),
                )
                processed_frames += 1
                if (previousTime_FPS > 0):
                  # Calculating the fps
                  fps = (1 / (color_timestamp - previousTime_FPS))*1e3
                  fps_samples.append(fps)
                  print("Frame rate: ", fps)
                if cfg.get('debug'):
                    processing_ms = (time.time() - loop_start_time) * 1000.0
                    print(f"Loop processing time: {processing_ms:.1f} ms")
                previousTime_FPS = color_timestamp
            else:
                print("Impossible to retrieve color or depth frame from camera")
            # stop execution after --interval set seconds
            if  ((currentTime - startTime) > cfg['interval']):
                print("Tracking recording completed")
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
    elapsed = max(time.time() - startTime, 1e-6)
    avg_fps = (sum(fps_samples) / len(fps_samples)) if fps_samples else 0.0
    print(f"Online processing summary: {processed_frames} frames, elapsed {elapsed:.2f}s, avg FPS {avg_fps:.2f}")
    if cfg['save']=='yes':
        if cfg.get('debug'):
            print_tracking_debug_stats(build_tracking_debug_stats(TRACKING_DATA))
        cb = (cfg.get("cam2base_json") or "").strip() or None
        out = save_xr_teleop_tracking_json(
            TRACKING_DATA,
            os.path.join(cfg["outputpath"], cfg["outputname"]),
            cam2base_json=cb,
        )
        print(f"Saved xr_teleoperate-compatible tracking JSON: {out}")

def offline_tracking(cfg):
    """
       Method to run GMH-D offline, by processing an input bag file obtained by using Intel Realsense recording utilities
       :param cfg: dictionary containing all the input configuration from command line execution
       """

    print("----GMH-D post-processing of MKV ----")
    print("Instantiating tracking utilities..")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg['model_asset_path']),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=cfg['n_hands'], min_hand_detection_confidence=cfg['handconf'],
        min_hand_presence_confidence=cfg['rerun_pd'],
        min_tracking_confidence=cfg['jointconf'])
    detector = HandLandmarker.create_from_options(options)

    # open Realsense camera
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)

    # Configure streams
    config = rs.config()
    rs.config.enable_device_from_file(config, os.path.join(cfg['bagfilepath'], cfg['bagfilename']), repeat_playback=False)

    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    config_pipeline = pipeline.start(config)

    # Disable real-time playback to process frames sequentially.
    playback = config_pipeline.get_device().as_playback()
    playback.set_real_time(False)

    # print(playback_config)
    # calculate FPS
    previousTime_FPS = -1
    success = True
    frame_count = 0
    fps_samples = []
    processing_start_time = time.time()
    try:
        while success:
            loop_start_time = time.time()
            # Get the next capture (blocking function)
            success, capture = pipeline.try_wait_for_frames(60)

            if not success:
                print('\033[96m' + "End of frames"+'\033[0m')
                break

            # both outputs as numpy array    
            aligned_capture = align.process(capture)
            color_frame = aligned_capture.get_color_frame()
            depth_frame = aligned_capture.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            img_color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())


            if img_color is not None and depth_image is not None:

                # save the timestamp at which the each image is captured
                color_timestamp = capture.get_timestamp()

                #captures may be asyncronously managed in recording, so we must ensure temporal consistency of consecutive frames
                if previousTime_FPS>=color_timestamp:
                    continue

                # get the color format of the image /BRG/ other...
                color_format = str(color_frame.profile).split(" ")[-1].strip('>')
                img_color=convert_to_bgra_if_required(color_format, img_color)
                rgb_image = cv2.cvtColor(img_color, cv2.COLOR_BGRA2RGB)

                # setup mediapipe image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


                # Detect hand landmarks from the input image.
                detection_result = detector.detect_for_video(mp_image, int(color_timestamp))


                process_sync_tracking(
                    detection_result,
                    img_color,
                    depth_frame,
                    color_timestamp,
                    intrinsics,
                    show_window=_as_bool_yes(cfg.get('visualize', 'yes')),
                )
                frame_count += 1

                if (previousTime_FPS > 0):
                    # Calculating the fps
                    fps = (1 / (color_timestamp - previousTime_FPS)) * 1e3
                    fps_samples.append(fps)
                    print("Frame rate: ", fps)
                if cfg.get('debug'):
                    processing_ms = (time.time() - loop_start_time) * 1000.0
                    print(f"Loop processing time: {processing_ms:.1f} ms")
                previousTime_FPS = color_timestamp
            else:
                print("Impossible to retrieve color or depth frame from camera")
    except RuntimeError as end:
        print(f"Playback runtime error: {end}")
        print("Extraction of tracking data completed")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    elapsed = max(time.time() - processing_start_time, 1e-6)
    avg_fps = (sum(fps_samples) / len(fps_samples)) if fps_samples else 0.0
    print(f"Offline processing summary: {frame_count} frames, elapsed {elapsed:.2f}s, avg FPS {avg_fps:.2f}")
    if cfg.get('debug'):
        print_tracking_debug_stats(build_tracking_debug_stats(TRACKING_DATA))
    if cfg['save'] == 'yes':
        cb = (cfg.get("cam2base_json") or "").strip() or None
        out = save_xr_teleop_tracking_json(
            TRACKING_DATA,
            os.path.join(cfg["outputpath"], cfg["outputname"]),
            cam2base_json=cb,
        )
        print(f"Saved xr_teleoperate-compatible tracking JSON: {out}")


if __name__ == "__main__":
    main()


