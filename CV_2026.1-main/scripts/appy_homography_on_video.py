from copy import Error
import json
import cv2
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import argparse


def draw_circle_on_position(frame: np.ndarray, positions: list[tuple[float, float]], show: bool = False, radius: int = 10, color: tuple[int, int, int] = (0, 0, 255), thickness: int = -1, transparent_background: bool = False) -> np.ndarray :
    """
    Draw circles on specified positions in a video frame.
    This function draws circles at given (x, y) positions on a frame. It supports
    displaying the result and optionally creating a transparent background for the circles.
    Args:
      frame (np.ndarray): Input frame/image where circles will be drawn. Must be a valid image array.
      positions (list[tuple[float, float]]): List of (x, y) coordinates where circles should be drawn.
      show (bool, optional): If True, displays the frame with circles using matplotlib. Defaults to False.
      radius (int, optional): Radius of the circles in pixels. Defaults to 10.
      color (tuple[int, int, int], optional): BGR color of the circles as (B, G, R). Defaults to (0, 0, 255) (red).
      thickness (int, optional): Thickness of the circle outline. Use -1 for filled circles. Defaults to -1 (filled).
      transparent_background (bool, optional): If True, creates a transparent background (black) for circles. Defaults to False.
    Returns:
      np.ndarray | None: The modified frame with drawn circles if show=False, otherwise None.
    Raises:
      ValueError: If the input frame is None or invalid.
    Examples:
      >>> frame = cv2.imread('image.jpg')
      >>> positions = [(100, 150), (200, 250)]
      >>> result = draw_circle_on_position(frame, positions, show=False, radius=15)
      >>> result = draw_circle_on_position(frame, positions, show=True, transparent_background=True)
    """
    if frame is None:
        raise ValueError("Frame inválido.")

    if transparent_background:
        height, width, _ = frame.shape
        modified_frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        modified_frame = frame.copy()

    for x, y in positions:
        cv2.circle(modified_frame, (int(x), int(y)), radius, color, thickness)

    if show:
        image_rgb = cv2.cvtColor(modified_frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.title("Circles on Frame")
        plt.axis('off')
        plt.show()
    else:
      return modified_frame


def get_video_info(video_path: str) -> tuple[float, int, int, int]:
  """
  Extract video information from a video file.
  
  Retrieves the frames per second (fps), width, height, and total frame count
  from a video file.
  
  Args:
    video_path (str): Path to the video file.
  
  Returns:
    tuple[float, int, int, int]: A tuple containing:
      - fps (float): Frames per second of the video.
      - width (int): Frame width in pixels.
      - height (int): Frame height in pixels.
      - total_frames (int): Total number of frames in the video.
  
  Raises:
    ValueError: If the video file cannot be opened.
  
  Examples:
    >>> fps, width, height, frames = get_video_info('video.mp4')
    >>> print(f"Video: {fps} fps, {width}x{height}, {frames} frames")
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise ValueError("Não foi possível abrir o vídeo.")

  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  cap.release()
  return fps, width, height, total_frames



def apply_homography_to_frame(
  frame: np.ndarray,
  dst_homo: list[dict],
  court_model: InferenceHTTPClient,
  player_model: InferenceHTTPClient,
  background_path: str = None,
  confidence_threshold: float = 0.5,
  width: int = 1290,
  height: int = 720
) -> np.ndarray:
  """
  Apply homography transformation to a video frame with player and court detection.
  
  This function detects players and court keypoints in a frame, computes a homography
  transformation, and blends the warped frame with a background image.
  
  Args:
    frame (np.ndarray): Input video frame to process.
    dst_homo (list[dict]): List of destination homography coordinates with 'id', 'x', 'y'.
    court_model (InferenceHTTPClient): Roboflow model client for court detection.
    player_model (InferenceHTTPClient): Roboflow model client for player detection.
    background_path (str | None, optional): Path to background image. If None, uses black background. Defaults to None.
    confidence_threshold (float, optional): Minimum confidence threshold for detections. Defaults to 0.5.
    width (int, optional): Frame width in pixels. Defaults to 1290.
    height (int, optional): Frame height in pixels. Defaults to 720.
  
  Returns:
    np.ndarray: Transformed frame with homography applied and blended with background.
  
  Raises:
    ValueError: If no valid keypoints are found for homography computation.
  """
  frame = cv2.resize(frame, (width, height))
  
  # Model para coletar coordenadas dos players
  player_results = player_model.infer(frame)
  
  players_coords = []

  # Busca matches de coordenadas de forma que sejam players e tenham um confidence adequado
  for entity in player_results['predictions']:
    if entity['class'] == "Player" and entity['confidence'] > confidence_threshold:
      players_coords.append([entity['x'], entity['y']])

  # Model para coletar pontos da quadra
  court_result = court_model.infer(frame)
  key_points = court_result['predictions'][0]['keypoints']
  
  kp_passed = []
  dst_passed = []

  # Busca matches de coordenadas de forma que colete suas ID_classes e tenham um confidence adequado
  for kp, coord in zip(key_points, dst_homo):
    if kp['confidence'] > confidence_threshold:
      if int(kp['class']) == coord['id']:
        kp_passed.append([kp['x'], kp['y']])
        dst_passed.append([coord['x'], coord['y']])
  
  kp_passed = np.array(kp_passed, dtype=np.float32)
  dst_passed = np.array(dst_passed, dtype=np.float32)

  # Fundo
  if background_path is None:
    background = np.zeros((height, width, 3), dtype=np.uint8)
  else:
    background = cv2.imread(background_path)
    background = cv2.resize(background, (width, height))

  # Escrevo no fundo as coordenadas dos players
  frame = draw_circle_on_position(background, players_coords, show=False, transparent_background=True)

  if len(kp_passed) > 0:
    homography = cv2.findHomography(kp_passed, dst_passed)
    warped_frame = cv2.warpPerspective(frame, homography[0], (width, height))
  else:
    raise ValueError("KP len 0")

  # Tudo homografado
  blended = cv2.addWeighted(warped_frame, 0.5, background, 0.5, 0)
  return blended


def apply_homography(
  video_path: str,
  dst_homo: list[dict],
  court_model: InferenceHTTPClient,
  player_model: InferenceHTTPClient,
  width: int = 1290,
  height: int = 720,
  background_path: str = None,
  confidence_threshold: float = 0.5,
  output: str = "./homography/video_homography.mp4"
) -> None:
  """
  Apply homography transformation to all frames in a video and save the result.
  
  Processes each frame of a video, applies homography transformation using player
  and court detection, and writes the transformed frames to an output video file.
  
  Args:
    video_path (str): Path to the input video file.
    dst_homo (list[dict]): List of destination homography coordinates with 'id', 'x', 'y'.
    court_model (InferenceHTTPClient): Roboflow model client for court detection.
    player_model (InferenceHTTPClient): Roboflow model client for player detection.
    width (int, optional): Frame width in pixels. Defaults to 1290.
    height (int, optional): Frame height in pixels. Defaults to 720.
    background_path (str | None, optional): Path to background image. If None, uses black background. Defaults to None.
    confidence_threshold (float, optional): Minimum confidence threshold for detections. Defaults to 0.5.
    output (str, optional): Path to output video file. Defaults to "./homography/video_homography.mp4".
  
  Returns:
    None
  
  Raises:
    ValueError: If the video file cannot be opened or if homography computation fails.
  """
  cap = cv2.VideoCapture(video_path)
  
  fps, width, height, total_frames = get_video_info(video_path)

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(output, fourcc, fps, (width, height))

  for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
      break
    
    homo_frame = apply_homography_to_frame(frame, dst_homo, court_model, player_model, background_path, confidence_threshold, width, height)
    out.write(homo_frame)

  cap.release()
  out.release()
  print(f"Homografia concluída. Video salvo em {output}")




if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Apply homography transformation to video. Example: python script.py dst_homo.json video.mp4 YOUR_API_KEY court-model-id player-model-id --output-path output.mp4",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  
  parser.add_argument("dst_path", type=str,  help="Path to dst_homo.json file")
  parser.add_argument("--background_path", type=str, default=None, help="Path to background image")
  parser.add_argument("video_path", type=str,  help="Path to input video")
  parser.add_argument("--output_path", default="./homography/video_homography.mp4", type=str, help="Path to output video")
  parser.add_argument("api_key", type=str,  help="Roboflow API key")
  parser.add_argument("court_model_id", type=str,  help="Court detection model ID")
  parser.add_argument("player_model_id", type=str,  help="Player detection model ID")
  parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for detections")
  parser.add_argument("--width", type=int, default=1290, help="Frame width")
  parser.add_argument("--height", type=int, default=720, help="Frame height")
  
  args = parser.parse_args()
  
  with open(args.dst_path, "r", encoding="utf-8") as f:
    dst_data = json.load(f)
  
  dst_data = dst_data["dst_homo"]
  
  player_model = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=args.api_key
  )
  player_model.select_model(args.player_model_id)
  
  court_model = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=args.api_key
  )
  court_model.select_model(args.court_model_id)
  
  apply_homography(
    video_path=args.video_path,
    dst_homo=dst_data,
    court_model=court_model,
    player_model=player_model,
    background_path=args.background_path,
    output=args.output_path,
    confidence_threshold=args.confidence_threshold,
    width=args.width,
    height=args.height
  )
