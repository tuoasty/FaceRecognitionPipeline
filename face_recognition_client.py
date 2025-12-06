import signal
import sys
import os
import cv2
import json
import numpy as np
from typing import Dict, List
import argparse
import time
import torch
import requests
import base64
from datetime import datetime

from face_recognition import FaceProcessor
from performance_monitor_client import PerformanceMonitorClient

class FaceRecognitionClient:
  def __init__(self,
              server_url='http://localhost:5000',
              use_gpu=True,
              auto_snapshot=True,
              snapshot_interval=5.0,
              max_tracking_distance=100,
              session_name=None,
              enable_performance_monitoring=True):
    
    print("\n" + "="*70)
    print("INITIALIZING FACE RECOGNITION CLIENT")
    print("="*70)
    
    self.server_url = server_url.rstrip('/')
    self.enable_performance_monitoring = enable_performance_monitoring
    
    try:
      response = requests.get(f'{self.server_url}/health', timeout=5)
      if response.status_code == 200:
        print(f"\nConnected to server at {self.server_url}")
        data = response.json()
        print(f"Server session: {data.get('session')}")
      else:
        raise Exception(f"Server returned status {response.status_code}")
    except Exception as e:
      raise Exception(f"Failed to connect to server at {self.server_url}: {e}")
    
  
    if session_name is None:
      session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      
    try:
      response = requests.post(
        f'{self.server_url}/init_session',
        json={'session_name': session_name},
        timeout=5
      )
      if response.status_code == 200:
        data = response.json()
        print(f"Session initialized: {data.get('session_name')}")
      else:
        print(f"Warning: Could not initialize session on server: {response.status_code}")
    except Exception as e:
      print(f"Warning: Could not initialize session on server: {e}")

    self.session_name = session_name
    
    if self.enable_performance_monitoring:
      temp_output_dir = os.path.join('.', 'temp_client_perf')
      self.perf_monitor = PerformanceMonitorClient(
        session_name=self.session_name,
        output_dir=temp_output_dir,
        latency_window_size=100
      )
      self.perf_monitor.log_detailed_frames = False
    else:
      self.perf_monitor = None
    
    print("\nLoading face detection model...")
    if use_gpu and torch.cuda.is_available():
      providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
      print("  Using GPU for face detection")
    else:
      providers = ['CPUExecutionProvider']
      print("  Using CPU for face detection")
    
    self.face_processor = FaceProcessor(
      output_size=112,
      det_size=(640, 640),
      det_thresh=0.5,
      quality_filter_config={
        'min_det_score': 0.5,
        'min_face_size': 40,
        'max_yaw': 60,
        'max_pitch': 45,
        'max_roll': 45,
        'check_blur': True,
        'blur_threshold': 50
      },
      providers=providers
    )
    
    self.auto_snapshot = auto_snapshot
    self.snapshot_interval = snapshot_interval
    self.last_snapshot_time = None
    
    self.frame_count = 0
    self.active_tracks = {}
    self.next_track_id = 0
    self.max_tracking_distance = max_tracking_distance
    
    self.recognized_tracks = {}
    self.recognition_attempts = {}
    self.failed_recognition_attempts = {}
    self.running = True
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)
    
    print("\n" + "="*70)
    print("Client ready!")
    print(f"Server: {self.server_url}")
    print(f"Session: {self.session_name}")
    print("="*70 + "\n")

  def _signal_handler(self, sig, frame):
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    self.running = False
  
  def _simple_track_assignment(self, faces: List[Dict]) -> Dict:
    new_assignments = {}
    
    for face in faces:
      bbox = face['bbox']
      center_x = (bbox[0] + bbox[2]) / 2
      center_y = (bbox[1] + bbox[3]) / 2

      best_track_id = None
      best_distance = self.max_tracking_distance
      
      for track_id, last_pos in self.active_tracks.items():
        dx = center_x - last_pos['x']
        dy = center_y - last_pos['y']
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < best_distance:
          best_distance = distance
          best_track_id = track_id
  
      if best_track_id is not None and best_track_id not in new_assignments:
        track_id = best_track_id
      else:
        track_id = self.next_track_id
        self.next_track_id += 1
      
      new_assignments[track_id] = {
        'x': center_x,
        'y': center_y,
        'face': face
      }
    
    self.active_tracks = {
      tid: {'x': data['x'], 'y': data['y'], 'face': data['face']}
      for tid, data in new_assignments.items()
    }
    self.failed_recognition_tracks = {
      tid: cooldown_end 
      for tid, cooldown_end in self.failed_recognition_tracks.items()
      if tid in self.active_tracks
    }

    return new_assignments
  
  def _encode_image_base64(self, image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')
  
  def _prepare_face_data(self, face: Dict, track_id: int) -> Dict:
    quality_metrics = face.get('quality_metrics', {})
    quality_metrics_clean = {
      k: float(v) if isinstance(v, (np.integer, np.floating)) else v
      for k, v in quality_metrics.items()
    }
    
    face_data = {
      'track_id': int(track_id),
      'bbox': [float(x) for x in face['bbox']],
      'det_score': float(face['det_score']),
      'quality_metrics': quality_metrics_clean,
      'aligned_face_base64': self._encode_image_base64(face['aligned_face']),
      'timestamp': datetime.now().isoformat()
    }

    if 'original_crop' in face and face['original_crop'].size > 0:
      face_data['original_crop_base64'] = self._encode_image_base64(face['original_crop'])
    
    return face_data
  
  def process_frame(self, frame: np.ndarray) -> Dict:
    self.frame_count += 1

    if self.perf_monitor:
      timings = self.perf_monitor.start_frame()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if self.perf_monitor:
      self.perf_monitor.mark_capture_end(timings)

    faces = self.face_processor.process_numpy(frame_rgb, return_all=True)
    
    if self.perf_monitor:
      self.perf_monitor.mark_detection_end(timings)
    
    for face in faces:
      x1, y1, x2, y2 = map(int, face['bbox'])
      face['original_crop'] = frame_rgb[y1:y2, x1:x2].copy()
    
    tracked_faces = self._simple_track_assignment(faces)

    faces_to_send = []
    current_time = time.time()

    for track_id, track_data in tracked_faces.items():
      if track_id in self.failed_recognition_tracks:
        cooldown_end = self.failed_recognition_tracks[track_id]
        if current_time < cooldown_end:
          continue
        else:
          del self.failed_recognition_tracks[track_id]
      
      face = track_data['face']
      face_data = self._prepare_face_data(face, track_id)
      faces_to_send.append(face_data)

    result = {'faces_detected': len(faces), 'active_tracks': len(tracked_faces)}
    network_request_sent = False
    
    if faces_to_send:
      if self.perf_monitor:
        self.perf_monitor.mark_network_start(timings)
      
      try:
        response = requests.post(
          f'{self.server_url}/process_faces',
          json={
            'faces': faces_to_send,
            'frame_count': self.frame_count
          },
          timeout=5
        )
        
        if self.perf_monitor:
          self.perf_monitor.mark_network_end(timings)
        
        network_request_sent = True
        
        if response.status_code == 200:
          server_result = response.json()
          result.update(server_result)

          self.recognized_tracks = server_result.get('recognized_tracks', {})
          self.recognition_attempts = server_result.get('recognition_attempts', {})
          for track_id_str, attempts in self.recognition_attempts.items():
            track_id = int(track_id_str)
            if attempts >= 3 and track_id_str not in self.recognized_tracks:
              self.failed_recognition_tracks[track_id] = time.time() + 10.0
        else:
          print(f"Server error: {response.status_code}")
      except Exception as e:
        print(f"Error sending to server: {e}")
        if self.perf_monitor:
          self.perf_monitor.mark_network_end(timings)
    
    if self.perf_monitor:
      perf_metrics = self.perf_monitor.end_frame(
        timings,
        num_faces_detected=len(faces),
        network_request_sent=network_request_sent
      )
      result['client_performance'] = perf_metrics
    
    return result
  
  def send_snapshot(self, frame: np.ndarray):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _, buffer = cv2.imencode('.jpg', frame)
    snapshot_base64 = base64.b64encode(buffer).decode('utf-8')
    
    try:
      response = requests.post(
        f'{self.server_url}/save_snapshot',
        json={
          'snapshot': snapshot_base64,
          'frame_count': self.frame_count,
          'timestamp': timestamp
        },
        timeout=5
      )
      
      if response.status_code == 200:
        data = response.json()
        print(f"[Snapshot] Saved on server: frame {self.frame_count}")
        return True
      else:
        print(f"Snapshot error: {response.status_code}")
        return False
    except Exception as e:
      print(f"Error sending snapshot: {e}")
      return False

  def finalize_session(self):
    client_report = None
    if self.perf_monitor:
      client_report = self.perf_monitor.finalize_session()
    
    try:
      payload = {}
      if client_report:
        payload['client_performance_report'] = client_report
      
      response = requests.post(
        f'{self.server_url}/finalize',
        json=payload,
        timeout=10
      )
      
      if response.status_code == 200:
        print("Session finalized on server")
      else:
        print(f"Finalize error: {response.status_code}")
    except Exception as e:
      print(f"Error finalizing: {e}")

  def run(self, camera_index=0, display=True, max_frames=None):
    print(f"Starting camera feed (camera {camera_index})...")
    print("Press 'q' to quit, 's' to save snapshot\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
      raise RuntimeError(f"Failed to open camera {camera_index}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    if self.auto_snapshot:
      self.last_snapshot_time = time.time()
    
    try:
      while self.running:
        ret, frame = cap.read()
        if not ret:
          print("Failed to read frame")
          break
        
        result = self.process_frame(frame)
        
        fps_frame_count += 1
        if fps_frame_count >= 30:
          fps_end_time = time.time()
          current_fps = fps_frame_count / (fps_end_time - fps_start_time)
          fps_start_time = fps_end_time
          fps_frame_count = 0

        if self.auto_snapshot:
          current_time = time.time()
          if current_time - self.last_snapshot_time >= self.snapshot_interval:
            self.send_snapshot(frame)
            self.last_snapshot_time = current_time
        
        if display:
          display_frame = self._draw_display(frame, result, current_fps)
          cv2.imshow('Face Recognition Client', display_frame)
          key = cv2.waitKey(1) & 0xFF
          
          if key == ord('q'):
            print("\nStopping...")
            self.running = False
          elif key == ord('s'):
            self.send_snapshot(frame)
        else:
          time.sleep(0.001)

        if max_frames and self.frame_count >= max_frames:
          print(f"\nReached max frames ({max_frames})")
          break
  
    finally:
      cap.release()
      if display:
        cv2.destroyAllWindows()
      
      self.finalize_session()

  def _draw_display(self, frame: np.ndarray, result: Dict, fps: float) -> np.ndarray:
    display = frame.copy()

    for track_id, track_data in self.active_tracks.items():
      face = track_data.get('face')
      if face is None:
        continue
      
      bbox = face['bbox']
      x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

      if str(track_id) in self.recognized_tracks:
        student_info = self.recognized_tracks[str(track_id)]
        color = (0, 255, 0)
        label = f"{student_info['name']} ({student_info['confidence']:.2f})"
      elif str(track_id) in self.recognition_attempts:
        attempts = self.recognition_attempts[str(track_id)]
        color = (0, 255, 255)
        label = f"Detecting... ({attempts}/3)"
      else:
        color = (255, 255, 0)
        label = f"track_{track_id:04d}"
  
      cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

      (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(display, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
      cv2.putText(display, label, (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    stats_y = 30
    stats = [
      f"Frame: {self.frame_count}",
      f"FPS: {fps:.1f}",
      f"Faces: {result.get('faces_detected', 0)}",
      f"Active Tracks: {result.get('active_tracks', 0)}",
      f"Recognized: {len(self.recognized_tracks)}"
    ]
    
    if 'client_performance' in result:
      perf = result['client_performance']
      stats.append(f"Capture: {perf.get('latency_capture_ms', 0):.1f}ms")
      stats.append(f"Detection: {perf.get('latency_detection_ms', 0):.1f}ms")
      stats.append(f"Network: {perf.get('latency_network_send_ms', 0):.1f}ms")
    
    for stat in stats:
      cv2.putText(display, stat, (10, stats_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      stats_y += 25
    
    return display
  
def main():
  parser = argparse.ArgumentParser(
    description='Face Recognition Client - Detects and tracks faces, sends to server'
  )
  parser.add_argument(
    '--camera',
    type=int,
    default=0,
    help='Camera index (default: 0)'
  )
  parser.add_argument(
    '--server',
    type=str,
    default='http://localhost:5000',
    help='Server URL (default: http://localhost:5000)'
  )
  parser.add_argument(
    '--no_display',
    action='store_true',
    help='Run without video display (headless)'
  )
  parser.add_argument(
    '--max_frames',
    type=int,
    default=None,
    help='Maximum frames to process (for testing)'
  )
  parser.add_argument(
    '--use_gpu',
    action='store_true',
    default=True,
    help='Use GPU acceleration (default: True)'
  )
  parser.add_argument(
    '--use_cpu',
    action='store_true',
    help='Force CPU usage (overrides --use_gpu)'
  )
  parser.add_argument(
    '--auto_snapshot',
    action='store_true',
    default=True,
    help='Enable automatic snapshot capture (default: True)'
  )
  parser.add_argument(
    '--no_auto_snapshot',
    action='store_true',
    help='Disable automatic snapshot capture'
  )
  parser.add_argument(
    '--snapshot_interval',
    type=float,
    default=30.0,
    help='Interval between auto-snapshots in seconds (default: 30.0)'
  )
  parser.add_argument(
    '--session_name',
    type=str,
    default=None,
    help='Custom session name (default: auto-generated)'
  )
  parser.add_argument(
    '--enable_perf_monitor',
    action='store_true',
    default=True,
    help='Enable performance monitoring (default: True)'
  )
      
  args = parser.parse_args()
  
  use_gpu = args.use_gpu and not args.use_cpu
  if use_gpu:
    if torch.cuda.is_available():
      print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
      print("GPU requested but not available, falling back to CPU")
      use_gpu = False
  else:
    print("Using CPU (GPU disabled)")

  auto_snapshot = args.auto_snapshot and not args.no_auto_snapshot

  client = FaceRecognitionClient(
    server_url=args.server,
    use_gpu=use_gpu,
    auto_snapshot=auto_snapshot,
    snapshot_interval=args.snapshot_interval,
    session_name=args.session_name,
    enable_performance_monitoring=args.enable_perf_monitor
  )
  
  client.run(
    camera_index=args.camera,
    display=not args.no_display,
    max_frames=args.max_frames
  )


if __name__ == '__main__':
  main()