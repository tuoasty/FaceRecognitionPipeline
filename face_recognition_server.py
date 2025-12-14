import os
import cv2
import json
import numpy as np
from typing import Dict, List, Optional
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
import torch
import time
from flask import Flask, request, jsonify
import base64
import traceback

from face_embedder import FaceEmbedder
from gallery_manager import GalleryManager
from performance_monitor_server import PerformanceMonitorServer
from face_recognition import FaceProcessor

app = Flask(__name__)

class LiveRecognitionTracker:
  def __init__(self, recognition_interval=30, max_attempts=3, buffer_size=10, retry_cooldown=10.0):
    self.recognized_tracks = {}  #track_id -> student_info
    self.recognition_attempts = {}  #track_id -> attempt_count
    self.track_frame_buffers = {}  #track_id -> deque of recent frames
    self.track_first_seen = {}  #track_id -> timestamp
    self.track_last_seen = {}  #track_id -> timestamp
    self.track_last_attempt = {}
    self.client_tracks = {}  # track_id -> {'bbox': [x1,y1,x2,y2], 'last_seen': timestamp}
    self.track_cooldowns = {}
        
    self.recognition_interval = recognition_interval
    self.max_attempts = max_attempts
    self.buffer_size = buffer_size
    self.retry_cooldown = retry_cooldown
        
  def should_recognize(self, track_id: int, frame_count: int) -> bool:
    if track_id in self.recognized_tracks:
        return False
    
    # Check cooldown first
    if self.is_track_in_cooldown(track_id):
        return False

    attempts = self.recognition_attempts.get(track_id, 0)

    if attempts >= self.max_attempts:
        self.set_track_cooldown(track_id, self.retry_cooldown)
        return False
    
    if track_id in self.track_frame_buffers:
        buffer = self.track_frame_buffers[track_id]
        if len(buffer) >= 1:
        
            best = max(buffer, key=lambda f: f.get('det_score', 0) * min(f.get('quality_metrics', {}).get('blur_score', 0) / 100.0, 1.0))
            if best.get('det_score', 0) > 0.6:  # Good detection
                return True
    return False
  
  def add_frame(self, track_id: int, face_data: Dict, timestamp: str):
    if track_id not in self.track_frame_buffers:
      self.track_frame_buffers[track_id] = deque(maxlen=self.buffer_size)
      self.track_first_seen[track_id] = timestamp
    
    self.track_last_seen[track_id] = timestamp
    self.track_frame_buffers[track_id].append(face_data)

  def get_best_frame(self, track_id: int) -> Optional[Dict]:
    if track_id not in self.track_frame_buffers:
      return None
    
    buffer = list(self.track_frame_buffers[track_id])
    if not buffer:
      return None

    def quality_score(face):
      det_score = face.get('det_score', 0)
      blur_score = face.get('quality_metrics', {}).get('blur_score', 0)
      blur_normalized = min(blur_score / 100.0, 1.0)
      return det_score * blur_normalized
    
    best_frame = max(buffer, key=quality_score)
    return best_frame
  
  def mark_recognized(self, track_id: int, student_info: Dict):
    self.recognized_tracks[track_id] = student_info
  
  def increment_attempts(self, track_id: int):
    self.recognition_attempts[track_id] = self.recognition_attempts.get(track_id, 0) + 1
    self.track_last_attempt[track_id] = datetime.now().isoformat()
  
  def get_track_duration(self, track_id: int) -> float:
    if track_id not in self.track_first_seen or track_id not in self.track_last_seen:
      return 0.0
    
    first = datetime.fromisoformat(self.track_first_seen[track_id])
    last = datetime.fromisoformat(self.track_last_seen[track_id])
    return (last - first).total_seconds()
  
  # def update_client_track(self, track_id: int, bbox: List[float], timestamp: str):
  #   """Update track position from client"""
  #   self.client_tracks[track_id] = {
  #       'bbox': bbox,
  #       'last_seen': timestamp
  #   }
    
  def is_track_in_cooldown(self, track_id: int) -> bool:
      if track_id in self.track_cooldowns:
          if time.time() < self.track_cooldowns[track_id]:
              return True
          else:
              # Cooldown expired - reset
              del self.track_cooldowns[track_id]
              self.recognition_attempts[track_id] = 0
              if track_id in self.track_frame_buffers:
                  self.track_frame_buffers[track_id].clear()
              return False
      return False

  def set_track_cooldown(self, track_id: int, cooldown_seconds: float = 3.0):
      """Set cooldown for failed track"""
      self.track_cooldowns[track_id] = time.time() + cooldown_seconds

class FaceRecognitionServer:
  def __init__(self,
                gallery_path='gallery/students.pkl',
                similarity_threshold=0.5,
                output_dir='sessions',
                session_name=None,
                require_session_name=False,
                recognition_interval=30,
                max_recognition_attempts=3,
                frame_buffer_size=10,
                enable_performance_monitoring=True,
                enable_gpu_monitoring=True,
                use_gpu=True,
                model_type='adaface',
                architecture='ir_101'):
      
    print("\n" + "="*70)
    print("INITIALIZING FACE RECOGNITION SERVER")
    print("="*70)
 
    self.similarity_threshold = similarity_threshold
    self.recognition_interval = recognition_interval
    self.max_recognition_attempts = max_recognition_attempts
    self.output_dir = output_dir
    self.require_session_name = require_session_name
    self.enable_performance_monitoring = enable_performance_monitoring
    self.enable_gpu_monitoring = enable_gpu_monitoring
    self.frame_buffer_size = frame_buffer_size

    print(f"\nLoading {model_type.upper()} recognition model ({architecture})...")
    if use_gpu and torch.cuda.is_available():
      device = torch.device('cuda')
      print("  Using GPU for face recognition")
    else:
      device = torch.device('cpu')
      print("  Using CPU for face recognition")

    self.embedder = FaceEmbedder(
      architecture=architecture,
      model_type=model_type,
      device=device
    )
    self.model_type = model_type
    self.architecture = architecture

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
    
    self.high_quality_crop_size = 600
    self.max_tracking_distance = 100
    self.next_track_id = 0
    
    print("\nLoading student gallery...")
    self.gallery = GalleryManager(gallery_path=gallery_path)
    num_students = len(self.gallery.get_all_students())
    print(f"Loaded {num_students} enrolled students")
    
    if num_students == 0:
      print("\nWARNING: Gallery is empty! Please enroll students first.")

    self.session_name = None
    self.session_dir = None
    self.session_start = None
    self.recognized_faces_dir = None
    self.unrecognized_faces_dir = None
    self.snapshots_dir = None
    self.perf_monitor = None
    self.tracker = None
    
    self.frame_count = 0
    self.total_faces_detected = 0
    self.total_recognition_attempts = 0

    if session_name:
      self._create_session(session_name)
    else:
      print("\n" + "="*70)
      print(f"Server ready!")
      print("Waiting for client to initialize session...")
      print(f"Recognition interval: every {recognition_interval} frames")
      print(f"Similarity threshold: {similarity_threshold}")
      print("="*70 + "\n")

  def _create_session(self, session_name: str):
    self.session_name = session_name
    self.session_dir = os.path.join(self.output_dir, self.session_name)
    os.makedirs(self.session_dir, exist_ok=True)

    if self.enable_performance_monitoring:
      model_id = f"{self.model_type.upper()}_{self.architecture.upper()}"
      device_str = "GPU" if str(self.embedder.device) == "cuda" else "CPU"
      model_id += f"_{device_str}"
      self.perf_monitor = PerformanceMonitorServer(
        model_identifier=model_id,
        session_name=self.session_name,
        output_dir=self.session_dir,
        enable_gpu_monitoring=self.enable_gpu_monitoring,
        latency_window_size=100
      )
      self.perf_monitor.log_detailed_requests = False

    self.tracker = LiveRecognitionTracker(
      recognition_interval=self.recognition_interval,
      max_attempts=self.max_recognition_attempts,
      buffer_size=self.frame_buffer_size
    )

    self.recognized_faces_dir = os.path.join(self.session_dir, 'recognized_faces')
    self.unrecognized_faces_dir = os.path.join(self.session_dir, 'unrecognized_faces')
    self.snapshots_dir = os.path.join(self.session_dir, 'snapshots')
    
    os.makedirs(self.recognized_faces_dir, exist_ok=True)
    os.makedirs(self.unrecognized_faces_dir, exist_ok=True)
    os.makedirs(self.snapshots_dir, exist_ok=True)

    self.session_start = datetime.now()
    self.frame_count = 0
    self.total_faces_detected = 0
    self.total_recognition_attempts = 0

    self._init_session_files()

    print("\n" + "="*70)
    print(f"Session created: {self.session_name}")
    print(f"Output: {self.session_dir}")
    print("="*70 + "\n")

  def _init_session_files(self):
    session_data = {
      "session_id": self.session_name,
      "start_time": self.session_start.isoformat(),
      "end_time": None,
      "status": "active",
      "settings": {
        "similarity_threshold": self.similarity_threshold,
        "recognition_interval": self.recognition_interval,
        "max_recognition_attempts": self.max_recognition_attempts
      },
      "statistics": {
        "total_frames_processed": 0,
        "total_faces_detected": 0,
        "total_recognition_attempts": 0,
        "unique_students_recognized": 0,
        "unrecognized_tracks": 0
      }
    }
    
    attendance_data = {
      "session_id": self.session_name,
      "last_updated": datetime.now().isoformat(),
      "recognized": [],
      "unrecognized": []
    }
    
    self._write_session(session_data)
    self._write_attendance(attendance_data)

  def _write_session(self, data: Dict):
    session_path = os.path.join(self.session_dir, 'session.json')
    with open(session_path, 'w') as f:
      json.dump(data, f, indent=2)
  
  def _write_attendance(self, data: Dict):
    attendance_path = os.path.join(self.session_dir, 'attendance.json')
    with open(attendance_path, 'w') as f:
      json.dump(data, f, indent=2)
  
  def _recognize_face(self, face_data: Dict, track_id: int) -> Optional[Dict]:
    self.total_recognition_attempts += 1

    aligned_face_bytes = base64.b64decode(face_data['aligned_face_base64'])
    aligned_face = cv2.imdecode(np.frombuffer(aligned_face_bytes, np.uint8), cv2.IMREAD_COLOR)
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    
    embedding = self.embedder.extract_embedding(aligned_face, normalize=True)
    
    results = self.gallery.search(embedding, top_k=3)
    
    if len(results) == 0:
      return None
    
    student_id, name, score = results[0]
    
    recognition_result = {
      'student_id': student_id,
      'name': name,
      'confidence': float(score),
      'track_id': track_id,
      'recognized': score >= self.similarity_threshold,
      'top_matches': [
        {'student_id': sid, 'name': n, 'score': float(s)}
        for sid, n, s in results
      ],
      'timestamp': datetime.now().isoformat(),
      'detection_quality': {
        'det_score': float(face_data['det_score']),
        'blur_score': face_data['quality_metrics'].get('blur_score', 0)
      }
    }
    
    return recognition_result
  
  def process_faces(self, faces_data: List[Dict], frame_count: int) -> Dict:
    # Start request timing
    if self.perf_monitor:
      timings = self.perf_monitor.start_request()

    self.frame_count = frame_count
    self.tracker.cleanup_stale_tracks(timeout_seconds=2.0)
    timestamp = datetime.now().isoformat()
    
    self.total_faces_detected += len(faces_data)
    
    recognition_events = []
    num_recognized_this_request = 0
    num_unknown_this_request = 0
    
    # Mark recognition processing start
    if self.perf_monitor:
      self.perf_monitor.mark_recognition_start(timings)
    
    for face_data in faces_data:
      track_id = face_data['track_id']
      
      self.tracker.add_frame(track_id, face_data, timestamp)

      if self.tracker.should_recognize(track_id, self.frame_count):
        best_frame = self.tracker.get_best_frame(track_id)
        
        if best_frame is not None:
          result = self._recognize_face(best_frame, track_id)
          self.tracker.increment_attempts(track_id)
          
          if result is not None:
            if result['recognized']:
              num_recognized_this_request += 1
              self.tracker.mark_recognized(track_id, result)
              face_filename = self._save_face_image(
                best_frame['aligned_face_base64'],
                track_id,
                result['student_id'],
                result['name'],
                result['confidence'],
                recognized=True,
                original_crop_base64=best_frame.get("original_crop_base64")
              )
              result['saved_face_path'] = face_filename
              recognition_events.append(('recognized', result))
              print(f"[Frame {self.frame_count}] Recognized: {result['name']} "
                f"(track_{track_id:04d}, confidence: {result['confidence']:.3f})")
            else:
              print(f"[Frame {self.frame_count}] Below threshold: {result['name']} "
                f"(track_{track_id:04d}, confidence: {result['confidence']:.3f}, "
                f"threshold: {self.similarity_threshold})")

              if self.tracker.recognition_attempts.get(track_id, 0) >= self.max_recognition_attempts:
                num_unknown_this_request += 1
                face_filename = self._save_face_image(
                  best_frame['aligned_face_base64'],
                  track_id,
                  result['student_id'],
                  result['name'],
                  result['confidence'],
                  recognized=False
                )
                result['saved_face_path'] = face_filename
                recognition_events.append(('unrecognized', result))
    
    # Mark recognition processing end
    if self.perf_monitor:
      self.perf_monitor.mark_recognition_end(timings)

    if recognition_events:
      self._update_attendance(recognition_events)

    # End request timing
    if self.perf_monitor:
      perf_metrics = self.perf_monitor.end_request(
        timings,
        num_faces_processed=len(faces_data),
        num_faces_recognized=num_recognized_this_request,
        num_faces_unknown=num_unknown_this_request
      )
    else:
      perf_metrics = {}
    
    return {
        'frame_count': self.frame_count,
        'faces_processed': len(faces_data),
        'recognition_events': len(recognition_events),
        'recognized_tracks': {str(k): v for k, v in self.tracker.recognized_tracks.items()},
        'recognition_attempts': {str(k): v for k, v in self.tracker.recognition_attempts.items()},
        'failed_tracks': {str(k): True for k in self.tracker.recognition_attempts.keys() 
                        if self.tracker.recognition_attempts[k] >= self.max_recognition_attempts 
                        and k not in self.tracker.recognized_tracks},
        'tracks_in_cooldown': {str(k): True for k in self.tracker.track_cooldowns.keys()},
        'performance': perf_metrics
    }
  
  def _save_face_image(self, 
                      aligned_face_base64: str,
                      track_id: int,
                      student_id: str,
                      name: str,
                      confidence: float,
                      recognized: bool,
                      original_crop_base64: str = None) -> str:
    output_dir = self.recognized_faces_dir if recognized else self.unrecognized_faces_dir

    if recognized:
      student_dir = os.path.join(output_dir, f"{student_id}_{name.replace(' ', '_')}")
      os.makedirs(student_dir, exist_ok=True)
      output_dir = student_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    filename_aligned = f"track_{track_id:04d}_{timestamp}_conf{confidence:.3f}_aligned.png"
    filepath_aligned = os.path.join(output_dir, filename_aligned)
    aligned_face_bytes = base64.b64decode(aligned_face_base64)
    with open(filepath_aligned, 'wb') as f:
      f.write(aligned_face_bytes)

    if original_crop_base64:
      filename_original = f"track_{track_id:04d}_{timestamp}_conf{confidence:.3f}_original.png"
      filepath_original = os.path.join(output_dir, filename_original)
      original_crop_bytes = base64.b64decode(original_crop_base64)
      with open(filepath_original, 'wb') as f:
        f.write(original_crop_bytes)

    return filepath_aligned

  def _update_attendance(self, events: List[tuple]):
    attendance_path = os.path.join(self.session_dir, 'attendance.json')
    with open(attendance_path, 'r') as f:
      attendance = json.load(f)

    for event_type, result in events:
      track_id = result['track_id']
      first_seen = self.tracker.track_first_seen.get(track_id, result['timestamp'])
      duration = self.tracker.get_track_duration(track_id)
      
      if event_type == 'recognized':
        existing = next((s for s in attendance['recognized'] 
                      if s['student_id'] == result['student_id']), None)
        
        if existing is None:
          attendance['recognized'].append({
            'student_id': result['student_id'],
            'name': result['name'],
            'first_seen': first_seen,
            'confidence': result['confidence'],
            'track_id': f"track_{track_id:04d}",
            'duration_seconds': duration,
            'detection_quality': result['detection_quality'],
            'saved_face_path': result.get('saved_face_path', '')
          })
        else:
          if result['confidence'] > existing['confidence']:
            existing['confidence'] = result['confidence']
            existing['detection_quality'] = result['detection_quality']
    
      elif event_type == 'unrecognized':
        attendance['unrecognized'].append({
          'track_id': f"track_{track_id:04d}",
          'first_seen': first_seen,
          'duration_seconds': duration,
          'best_match': {
            'name': result['name'],
            'student_id': result['student_id'],
            'confidence': result['confidence']
          },
          'reason': 'below_threshold',
          'threshold': self.similarity_threshold,
          'attempts': self.tracker.recognition_attempts.get(track_id, 0),
          'top_matches': result['top_matches'],
          'saved_face_path': result.get('saved_face_path', '')
        })

    attendance['last_updated'] = datetime.now().isoformat()
    self._write_attendance(attendance)
  
  def save_snapshot(self, snapshot_base64: str, frame_count: int, timestamp: str) -> str:
    snapshot_bytes = base64.b64decode(snapshot_base64)
    filename = f'snapshot_frame_{frame_count:06d}_{timestamp}.png'
    filepath = os.path.join(self.snapshots_dir, filename)
    
    with open(filepath, 'wb') as f:
      f.write(snapshot_bytes)
    
    return filepath
  
  def finalize_session(self, client_report: Optional[Dict] = None):
    print("\n" + "="*70)
    print("FINALIZING SESSION")
    print("="*70)
    
    session_end = datetime.now()
    duration = (session_end - self.session_start).total_seconds()

    if self.perf_monitor:
      perf_report = self.perf_monitor.finalize_session(client_report=client_report)

    session_path = os.path.join(self.session_dir, 'session.json')
    attendance_path = os.path.join(self.session_dir, 'attendance.json')
    
    with open(session_path, 'r') as f:
      session_data = json.load(f)
    
    with open(attendance_path, 'r') as f:
      attendance_data = json.load(f)
    
    session_data['end_time'] = session_end.isoformat()
    session_data['status'] = 'completed'
    session_data['duration_seconds'] = duration
    session_data['statistics'] = {
      'total_frames_processed': self.frame_count,
      'total_faces_detected': self.total_faces_detected,
      'total_recognition_attempts': self.total_recognition_attempts,
      'unique_students_recognized': len(attendance_data['recognized']),
      'unrecognized_tracks': len(attendance_data['unrecognized'])
    }
    
    self._write_session(session_data)
    print(f"\nSession: {self.session_name}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Frames processed: {self.frame_count}")

    print(f"\nRecognized students: {len(attendance_data['recognized'])}")
    for student in attendance_data['recognized']:
      print(f"  {student['name']} ({student['student_id']}) - confidence: {student['confidence']:.3f}")
    
    print(f"\nUnrecognized tracks: {len(attendance_data['unrecognized'])}")
    for track in attendance_data['unrecognized']:
      best = track['best_match']
      print(f"  {track['track_id']} - best match: {best['name']} ({best['confidence']:.3f})")
    
    print(f"\nOutput directory: {self.session_dir}")
    print("="*70 + "\n")

  def process_full_frame(self, frame_rgb: np.ndarray, frame_count: int, timestamp: str) -> Dict:
    """Process full frame: detect -> track -> recognize"""
    
    if self.perf_monitor:
      timings = self.perf_monitor.start_request()

    self.frame_count = frame_count
    
    # Detect faces
    faces = self.face_processor.process_numpy(frame_rgb, return_all=True)
    
    # Extract high-quality crops
    for face in faces:
      x1, y1, x2, y2 = map(int, face['bbox'])
      
      face_width = x2 - x1
      face_height = y2 - y1
      margin = int(max(face_width, face_height) * 0.3)
      
      crop_x1 = max(0, x1 - margin)
      crop_y1 = max(0, y1 - margin)
      crop_x2 = min(frame_rgb.shape[1], x2 + margin)
      crop_y2 = min(frame_rgb.shape[0], y2 + margin)
      
      high_res_crop = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
      
      if high_res_crop.shape[0] > self.high_quality_crop_size or high_res_crop.shape[1] > self.high_quality_crop_size:
        scale = self.high_quality_crop_size / max(high_res_crop.shape[0], high_res_crop.shape[1])
        new_width = int(high_res_crop.shape[1] * scale)
        new_height = int(high_res_crop.shape[0] * scale)
        high_res_crop = cv2.resize(high_res_crop, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
  
      face['original_crop'] = high_res_crop
    
    # Track faces
    tracked_faces = self._simple_track_assignment(faces, timestamp)
    
    # Prepare faces for recognition
    faces_data = []
    for track_id, track_data in tracked_faces.items():
      face = track_data['face']
      face_data = self._prepare_face_data(face, track_id, timestamp)
      faces_data.append(face_data)
    
    # Process recognition
    if self.perf_monitor:
      self.perf_monitor.mark_recognition_start(timings)
    
    recognition_events = []
    num_recognized = 0
    num_unknown = 0
    
    for face_data in faces_data:
      track_id = face_data['track_id']
      self.tracker.add_frame(track_id, face_data, timestamp)
      
      if self.tracker.should_recognize(track_id, self.frame_count):
        best_frame = self.tracker.get_best_frame(track_id)
        
        if best_frame is not None:
          result = self._recognize_face(best_frame, track_id)
          self.tracker.increment_attempts(track_id)
          
          if result is not None:
            if result['recognized']:
              num_recognized += 1
              self.tracker.mark_recognized(track_id, result)
              face_filename = self._save_face_image(
                best_frame['aligned_face_base64'],
                track_id,
                result['student_id'],
                result['name'],
                result['confidence'],
                recognized=True,
                original_crop_base64=best_frame.get("original_crop_base64")
              )
              result['saved_face_path'] = face_filename
              recognition_events.append(('recognized', result))
              print(f"[Frame {self.frame_count}] Recognized: {result['name']} "
                f"(track_{track_id:04d}, confidence: {result['confidence']:.3f})")
            else:
              if self.tracker.recognition_attempts.get(track_id, 0) >= self.max_recognition_attempts:
                num_unknown += 1
                face_filename = self._save_face_image(
                  best_frame['aligned_face_base64'],
                  track_id,
                  result['student_id'],
                  result['name'],
                  result['confidence'],
                  recognized=False
                )
                result['saved_face_path'] = face_filename
                recognition_events.append(('unrecognized', result))
    
    if self.perf_monitor:
      self.perf_monitor.mark_recognition_end(timings)
    
    if recognition_events:
      self._update_attendance(recognition_events)
    
    # Prepare tracks for client display
    # Prepare tracks for client display
    tracks_for_client = []
    for track_id, track_data in tracked_faces.items():
      face = track_data['face']
      tracks_for_client.append({
        'track_id': track_id,
        'bbox': [float(x) for x in face['bbox']],  # <-- Convert to list of floats
        'det_score': float(face['det_score'])
      })
    if self.perf_monitor:
      perf_metrics = self.perf_monitor.end_request(
        timings,
        num_faces_processed=len(faces_data),
        num_faces_recognized=num_recognized,
        num_faces_unknown=num_unknown
      )
    else:
      perf_metrics = {}
    
    # Replace the return statement at the end of process_full_frame (around line 745-760)

    # Build newly_recognized dict properly
    newly_recognized = {}
    for event_type, result in recognition_events:
      if event_type == 'recognized':
        track_id = result['track_id']
        newly_recognized[str(track_id)] = {
          'student_id': result['student_id'],
          'name': result['name'],
          'confidence': result['confidence'],
          'timestamp': result['timestamp']
        }
    
    # Build newly_failed list properly
    newly_failed = []
    for event_type, result in recognition_events:
      if event_type == 'unrecognized':
        newly_failed.append(str(result['track_id']))
    
    return {
      'frame_count': self.frame_count,
      'faces_detected': len(faces),
      'active_tracks': len(tracked_faces),
      'tracks': tracks_for_client,
      'recognized_tracks': {str(k): v for k, v in self.tracker.recognized_tracks.items()},
      'recognition_attempts': {str(k): v for k, v in self.tracker.recognition_attempts.items()},
      'failed_tracks': {str(k): True for k in self.tracker.recognition_attempts.keys() 
                      if self.tracker.recognition_attempts[k] >= self.max_recognition_attempts 
                      and k not in self.tracker.recognized_tracks},
      'newly_recognized': newly_recognized,
      'newly_failed': newly_failed,
      'performance': perf_metrics
    }
  
  def _simple_track_assignment(self, faces: List[Dict], timestamp: str) -> Dict:
    """Simple tracking based on spatial proximity"""
    new_assignments = {}
    
    # Get existing active tracks from tracker
    active_tracks = {}
    current_time = datetime.now()
    timeout_seconds = 2.0  # Consider tracks stale after 2 seconds

    for track_id, track_info in self.tracker.client_tracks.items():
      # Check if track is still recent
      last_seen = datetime.fromisoformat(track_info['last_seen'])
      if (current_time - last_seen).total_seconds() > timeout_seconds:
        continue
        
      bbox = track_info['bbox']
      active_tracks[track_id] = {
        'x': (bbox[0] + bbox[2]) / 2,
        'y': (bbox[1] + bbox[3]) / 2
      }
    
    for face in faces:
      bbox = face['bbox']
      center_x = (bbox[0] + bbox[2]) / 2
      center_y = (bbox[1] + bbox[3]) / 2

      best_track_id = None
      best_distance = self.max_tracking_distance
      
      for track_id, last_pos in active_tracks.items():
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
      
      # Update tracker
      # self.tracker.update_client_track(track_id, bbox, timestamp)
    
    return new_assignments
  
  def _prepare_face_data(self, face: Dict, track_id: int, timestamp: str) -> Dict:
    """Prepare face data for recognition"""
    quality_metrics = face.get('quality_metrics', {})
    quality_metrics_clean = {
      k: float(v) if isinstance(v, (np.integer, np.floating)) else v
      for k, v in quality_metrics.items()
    }
    
    # Encode aligned face
    aligned_face_bgr = cv2.cvtColor(face['aligned_face'], cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', aligned_face_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    aligned_base64 = base64.b64encode(buffer).decode('utf-8')
    
    face_data = {
      'track_id': int(track_id),
      'bbox': [float(x) for x in face['bbox']],
      'det_score': float(face['det_score']),
      'quality_metrics': quality_metrics_clean,
      'aligned_face_base64': aligned_base64,
      'timestamp': timestamp
    }

    if 'original_crop' in face and face['original_crop'].size > 0:
      crop_bgr = cv2.cvtColor(face['original_crop'], cv2.COLOR_RGB2BGR)
      _, buffer = cv2.imencode('.png', crop_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
      face_data['original_crop_base64'] = base64.b64encode(buffer).decode('utf-8')
    
    return face_data
  
  def cleanup_stale_tracks(self, timeout_seconds=5.0):
    """Remove tracks that haven't been seen recently"""
    current_time = datetime.now()
    stale_tracks = []
    
    for track_id, track_info in self.client_tracks.items():
        last_seen = datetime.fromisoformat(track_info['last_seen'])
        if (current_time - last_seen).total_seconds() > timeout_seconds:
            stale_tracks.append(track_id)
    
    for track_id in stale_tracks:
        del self.client_tracks[track_id]
        # Optionally clean up other track data
        if track_id in self.track_frame_buffers:
            del self.track_frame_buffers[track_id]


@app.route('/health', methods=['GET'])
def health():
  return jsonify({'status': 'ok', 'session': server.session_name if server else None})

# @app.route('/process_faces', methods=['POST'])
# def process_faces():
#   if server.session_name is None:
#     return jsonify({'error': 'No active session. Call /init_session first'}), 400
  
#   try:
#     data = request.json
#     faces_data = data.get('faces', [])
#     frame_count = data.get('frame_count', 0)
    
#     result = server.process_faces(faces_data, frame_count)
#     return jsonify(result)
#   except Exception as e:
#     import traceback
#     error_details = {
#       'error': str(e),
#       'error_type': type(e).__name__,
#       'traceback': traceback.format_exc()
#     }
#     print(f"\n[ERROR] process_faces failed: {error_details['traceback']}")
#     return jsonify(error_details), 500

@app.route('/save_snapshot', methods=['POST'])
def save_snapshot():
  if server.session_name is None:
    return jsonify({'error': 'No active session. Call /init_session first'}), 400
  
  try:
    data = request.json
    snapshot_base64 = data.get('snapshot')
    frame_count = data.get('frame_count', 0)
    timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    filepath = server.save_snapshot(snapshot_base64, frame_count, timestamp)
    return jsonify({'saved': True, 'path': filepath})
  except Exception as e:
    import traceback
    error_details = {
      'error': str(e),
      'error_type': type(e).__name__,
      'traceback': traceback.format_exc()
    }
    print(f"\n[ERROR] save_snapshot failed: {error_details['traceback']}")
    return jsonify(error_details), 500
  
@app.route('/finalize', methods=['POST'])
def finalize():
  if server.session_name is None:
    return jsonify({'error': 'No active session'}), 400
  
  try:
    # Get client performance report if provided
    data = request.json or {}
    client_report = data.get('client_performance_report')
    
    server.finalize_session(client_report=client_report)
    return jsonify({'status': 'finalized'})
  except Exception as e:
    import traceback
    error_details = {
      'error': str(e),
      'error_type': type(e).__name__,
      'traceback': traceback.format_exc()
    }
    print(f"\n[ERROR] finalize failed: {error_details['traceback']}")
    return jsonify(error_details), 500

@app.route('/init_session', methods=['POST'])
def init_session():
  try:
    data = request.json
    session_name = data.get('session_name')
    
    if not session_name:
      return jsonify({'error': 'session_name is required'}), 400

    server._create_session(session_name)
    
    return jsonify({
      'status': 'session_initialized', 
      'session_name': session_name,
      'session_dir': server.session_dir
    })
  except Exception as e:
    import traceback
    error_details = {
      'error': str(e),
      'error_type': type(e).__name__,
      'traceback': traceback.format_exc()
    }
    print(f"\n[ERROR] init_session failed: {error_details['traceback']}")
    return jsonify(error_details), 500
  
@app.route('/process_frame', methods=['POST'])
def process_frame():
  if server.session_name is None:
    return jsonify({'error': 'No active session. Call /init_session first'}), 400
  
  try:
    data = request.json
    frame_base64 = data.get('frame')
    frame_count = data.get('frame_count', 0)
    timestamp = data.get('timestamp', datetime.now().isoformat())
  

    # Decode frame
    frame_bytes = base64.b64decode(frame_base64)
    print(len(frame_base64) / 1024 / 1024, "MB")
    frame_array = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame on server
    result = server.process_full_frame(frame_rgb, frame_count, timestamp)
    return jsonify(result)
  except Exception as e:
    import traceback
    error_details = {
      'error': str(e),
      'error_type': type(e).__name__,
      'traceback': traceback.format_exc()
    }
    print(f"\n[ERROR] process_frame failed: {error_details['traceback']}")
    return jsonify(error_details), 500


def main():
  parser = argparse.ArgumentParser(
    description='Face Recognition Server for classroom attendance'
  )
  parser.add_argument(
    '--gallery',
    type=str,
    default='gallery/adaface_ir101.pkl',
    help='Path to student gallery database'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.4,
    help='Similarity threshold for recognition (0-1)'
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default='sessions',
    help='Directory to save session data'
  )
  parser.add_argument(
    '--session_name',
    type=str,
    default=None,
    help='Custom session name (default: auto-generated)'
  )
  parser.add_argument(
    '--recognition_interval',
    type=int,
    default=30,
    help='Frames between recognition attempts (default: 30)'
  )
  parser.add_argument(
    '--max_attempts',
    type=int,
    default=3,
    help='Max recognition attempts per track (default: 3)'
  )
  parser.add_argument(
    '--enable_perf_monitor',
    action='store_true',
    default=True,
    help='Enable performance monitoring (default: True)'
  )
  parser.add_argument(
    '--enable_gpu_monitor',
    action='store_true',
    default=True,
    help='Enable GPU monitoring if available (default: True)'
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
    '--model_type',
    type=str,
    default='adaface',
    choices=['adaface', 'arcface'],
    help='Type of face recognition model to use (default: adaface)'
  )
  parser.add_argument(
    '--architecture',
    type=str,
    default='ir_101',
    choices=['ir_50', 'ir_101'],
    help='Model architecture - ir_50 or ir_101 (default: ir_101)'
  )
  parser.add_argument(
    '--host',
    type=str,
    default='0.0.0.0',
    help='Server host (default: 0.0.0.0)'
  )
  parser.add_argument(
    '--port',
    type=int,
    default=5000,
    help='Server port (default: 5000)'
  )
  parser.add_argument(
    '--require_session_name',
    action='store_true',
    help='Require session name from client (default: False)'
  )
  parser.add_argument(
    '--frame_skip',
    type=int,
    default=3,
    help='Send every Nth frame to reduce bandwidth (default: 3)'
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

  global server
  server = FaceRecognitionServer(
    gallery_path=args.gallery,
    similarity_threshold=args.threshold,
    output_dir=args.output_dir,
    session_name=args.session_name,
    recognition_interval=args.recognition_interval,
    max_recognition_attempts=args.max_attempts,
    enable_performance_monitoring=args.enable_perf_monitor,
    enable_gpu_monitoring=args.enable_gpu_monitor,
    use_gpu=use_gpu,
    model_type=args.model_type,
    architecture=args.architecture,
    require_session_name=args.require_session_name
  )
  
  print(f"\nStarting server on {args.host}:{args.port}...")
  app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
  main()