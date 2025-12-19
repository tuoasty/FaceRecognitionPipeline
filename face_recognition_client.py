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

from performance_monitor_client import PerformanceMonitorClient

class FaceRecognitionClient:
  def __init__(self,
              server_url='http://localhost:5000',
              auto_snapshot=True,
              snapshot_interval=5.0,
              session_name=None,
              enable_performance_monitoring=True,
              frame_skip=3):
    
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
  
    self.auto_snapshot = auto_snapshot
    self.snapshot_interval = snapshot_interval
    self.last_snapshot_time = None
    
    self.frame_count = 0
    self.active_tracks = {}
    self.recognized_tracks = {}
    self.recognition_attempts = {}
    self.failed_tracks = {}
    self.running = True
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)

    self.frame_skip = frame_skip
    self.frames_since_last_send = 0
    print("\n" + "="*70)
    print("Client ready!")
    print(f"Server: {self.server_url}")
    print(f"Session: {self.session_name}")
    print("="*70 + "\n")

  def _signal_handler(self, sig, frame):
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    self.running = False
  
  def _encode_image_base64(self, image: np.ndarray, format='png') -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if format == 'png':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        _, buffer = cv2.imencode('.png', image_bgr, encode_params)
    else:
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 100,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        ]
        _, buffer = cv2.imencode('.jpg', image_bgr, encode_params)
    
    return base64.b64encode(buffer).decode('utf-8')
  
  def _get_camera_name(self, cap: cv2.VideoCapture, camera_index: int) -> str:
    try:
        backend = cap.getBackendName()
        try:
            desc = cap.get(cv2.CAP_PROP_DESCRIPTION)
            if desc:
                return f"{desc}"
        except:
            pass
            
        return f"Camera {camera_index} ({backend})"
    except Exception as e:
        return f"Camera {camera_index}"
    
  def _get_max_camera_resolution(self, cap: cv2.VideoCapture) -> tuple:
    test_resolutions = [
        (3840, 2160),
        (2560, 1440),
        (1920, 1080), 
        (1280, 720),
        (640, 480), 
    ]

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_width, max_height = original_width, original_height
    
    for width, height in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(cap.get(cv2.CAP_PROP_FOURCC))
        print(cap.get(cv2.CAP_PROP_FPS))
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            max_width, max_height = width, height
            print(f"  Supported: {width}x{height} ✓")
            break
        else:
            print(f"  Not supported: {width}x{height} (got {actual_width}x{actual_height})")
    
    return max_width, max_height
  
  def _log_recognition_updates(self, result: Dict):
    if 'newly_recognized' in result:
        for track_id, student_info in result['newly_recognized'].items():
            print(f"[RECOGNIZED] Track {track_id}: {student_info['name']} "
                  f"(confidence: {student_info['confidence']:.3f})")

    if 'closest_matches' in result:
        for track_id, match_info in result['closest_matches'].items():
            if match_info:
                print(f"[CLOSEST MATCH] Track {track_id}: {match_info['name']} "
                      f"(distance: {match_info['distance']:.3f})")

    if 'newly_failed' in result:
        for track_id in result['newly_failed']:
            print(f"[FAILED] Track {track_id}: No match found after 3 attempts")
  
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
      'aligned_face_base64': self._encode_image_base64(face['aligned_face'], format='png'),
      'timestamp': datetime.now().isoformat()
    }

    if 'original_crop' in face and face['original_crop'].size > 0:
      face_data['original_crop_base64'] = self._encode_image_base64(face['original_crop'], quality=98)
    
    return face_data
  
  def process_frame(self, frame: np.ndarray) -> Dict:
    self.frame_count += 1

    if self.perf_monitor:
      timings = self.perf_monitor.start_frame()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if self.perf_monitor:
      self.perf_monitor.mark_capture_end(timings)

    self.frames_since_last_send += 1
    
    result = {
        'faces_detected': 0,
        'active_tracks': 0,
        'recognized_tracks': self.recognized_tracks,
        'recognition_attempts': self.recognition_attempts,
        'failed_tracks': self.failed_tracks
    }
    network_request_sent = False

    if self.frames_since_last_send >= self.frame_skip:
        if self.perf_monitor:
            self.perf_monitor.mark_network_start(timings)
        
        try:
            frame_base64 = self._encode_image_base64(frame_rgb, format='png')
            
            response = requests.post(
                f'{self.server_url}/process_frame',
                json={
                    'frame': frame_base64,
                    'frame_count': self.frame_count,
                    'timestamp': datetime.now().isoformat()
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
                self.failed_tracks = server_result.get('failed_tracks', {})

                if 'tracks' in server_result:
                    self.active_tracks = {}
                    for track in server_result['tracks']:
                        track_id = track['track_id']
                        self.active_tracks[track_id] = {
                            'x': (track['bbox'][0] + track['bbox'][2]) / 2,
                            'y': (track['bbox'][1] + track['bbox'][3]) / 2,
                            'face': track
                        }
                
                self._log_recognition_updates(server_result)
            else:
                print(f"Server error: {response.status_code}")
        except Exception as e:
            print(f"Error sending to server: {e}")
            if self.perf_monitor:
                self.perf_monitor.mark_network_end(timings)
        finally:
            self.frames_since_last_send = 0
    else:
        if self.perf_monitor:
            self.perf_monitor.mark_network_start(timings)
            self.perf_monitor.mark_network_end(timings)
    
    if self.perf_monitor:
      perf_metrics = self.perf_monitor.end_frame(
          timings,
          num_faces_detected=result.get('faces_detected', 0),
          network_request_sent=network_request_sent
      )
      result['client_performance'] = perf_metrics
    
    return result

  def send_snapshot(self, frame: np.ndarray):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    snapshot_base64 = self._encode_image_base64(frame_rgb, format='png')
    
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
    
    camera_name = self._get_camera_name(cap, camera_index)
    print(f"Camera device: {camera_name}")
    print("Detecting maximum camera resolution...")
    max_width, max_height = self._get_max_camera_resolution(cap)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Detection mode: Fast tracking (640x640) + High-quality crops")
    if display:
      window_name = 'Face Recognition Client'
      cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
      cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
      print("Display: Fullscreen mode enabled")
      print("Press 'f' to toggle fullscreen, 'q' to quit\n")
    else:
      window_name = None
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    if self.auto_snapshot:
      self.last_snapshot_time = time.time()
    
    test = 0

    try:
      while self.running:
        ret, frame = cap.read()
        if test == 0:
           cv2.imwrite("raw_test.png", frame)
           test = 1
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
      elif str(track_id) in self.failed_tracks:
        color = (0, 0, 255)
        label = "Unrecognized"
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
  parser.add_argument(
    '--frame_skip',
    type=int,
    default=5,
    help='Number of faces to accumulate before sending to server (default: 5)'
  )

      
  args = parser.parse_args()
  auto_snapshot = args.auto_snapshot and not args.no_auto_snapshot

  client = FaceRecognitionClient(
    server_url=args.server,
    auto_snapshot=auto_snapshot,
    snapshot_interval=args.snapshot_interval,
    session_name=args.session_name,
    enable_performance_monitoring=args.enable_perf_monitor,
    frame_skip=args.frame_skip
  )
  
  client.run(
    camera_index=args.camera,
    display=not args.no_display,
    max_frames=args.max_frames
  )


if __name__ == '__main__':
  main()