import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np

mp_pose = mp.solutions.pose

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'pose_landmarker_heavy.task'

latest_result = None

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
	global latest_result
	latest_result = result

def draw_landmarks_on_frame(frame, result: PoseLandmarkerResult):
    if not result or not result.pose_landmarks:
        return frame

    pose_landmarks = result.pose_landmarks[0]
    image_height, image_width, _ = frame.shape

    # 線を描画（骨の接続）
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]
        x0, y0 = int(start.x * image_width), int(start.y * image_height)
        x1, y1 = int(end.x * image_width), int(end.y * image_height)
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # 点を描画（関節）
    for landmark in pose_landmarks:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    return frame

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='green', s=20)
lines = []

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel("-y")
ax.view_init(elev=10, azim=110)

mp_pose = mp.solutions.pose
connections = list(mp_pose.POSE_CONNECTIONS)

cap = cv2.VideoCapture(0)
with PoseLandmarker.create_from_options(options) as landmarker:
	while True:
		rep, frame = cap.read()
		frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
		frame_timestamp_ms = int(time.time() * 1000)
		landmarker.detect_async(mp_image, frame_timestamp_ms)
		frame = draw_landmarks_on_frame(frame, latest_result)
		cv2.imshow('MediaPipe Pose Landmarker', frame)
		if latest_result and latest_result.pose_world_landmarks:
			landmarks = latest_result.pose_world_landmarks[0]
			x = np.array([lm.x for lm in landmarks])
			y = np.array([lm.y for lm in landmarks])
			z = np.array([lm.z for lm in landmarks])
			sc._offsets3d = (x, z, -y)
			for line in lines:
				line.remove()
			lines.clear()
			for start, end in connections:
				x_pair = [landmarks[start].x, landmarks[end].x]
				y_pair = [landmarks[start].y, landmarks[end].y]
				z_pair = [landmarks[start].z, landmarks[end].z]
				line = ax.plot(x_pair, z_pair, [-v for v in y_pair], c='orange')[0]
				lines.append(line)
			plt.draw()
			plt.pause(0.001)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
