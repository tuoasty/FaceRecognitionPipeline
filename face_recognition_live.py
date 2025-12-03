import os
import cv2
import json
import numpy as np
from typing import Dict, List, Optional
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque
import time
import torch

from face_recognition import FaceProcessor
from face_embedder import FaceEmbedder
from gallery_manager import GalleryManager
from performance_monitor_server import PerformanceMonitor

class LiveRecognitionTracker:
  def __init__(self, recognition_interval=30, max_attempts=3, buffer_size=10):
      self.recognized_tracks = {}  #track_id -> student_info
      self.recognition_attempts = {}  #track_id -> attempt_count
      self.track_frame_buffers = {}  #track_id -> deque of recent frames
      self.track_first_seen = {}  #track_id -> timestamp
      self.track_last_seen = {}  #track_id -> timestamp
      
      self.recognition_interval = recognition_interval
      self.max_attempts = max_attempts
      self.buffer_size = buffer_size
        
  def should_recognize(self, track_id: int, frame_count: int) -> bool:
    if track_id in self.recognized_tracks:
      return False

    attempts = self.recognition_attempts.get(track_id, 0)
    if attempts >= self.max_attempts:
      return False

    if frame_count % self.recognition_interval == 0:
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
  
  def get_track_duration(self, track_id: int) -> float:
    if track_id not in self.track_first_seen or track_id not in self.track_last_seen:
      return 0.0
    
    first = datetime.fromisoformat(self.track_first_seen[track_id])
    last = datetime.fromisoformat(self.track_last_seen[track_id])
    return (last - first).total_seconds()

class LiveFaceRecognition:
  def __init__(self,
                gallery_path='gallery/students.pkl',
                similarity_threshold=0.5,
                output_dir='sessions',
                session_name=None,
                recognition_interval=30,
                max_recognition_attempts=3,
                frame_buffer_size=10,
                enable_performance_monitoring=True,
                enable_gpu_monitoring=True,
                use_gpu=True,
                model_type='adaface',
                architecture='ir_101',
                auto_snapshot=True,
                snapshot_interval=5.0):
    
    print("\n" + "="*70)
    print("INITIALIZING LIVE FACE RECOGNITION SYSTEM")
    print("="*70)
    
    self.similarity_threshold = similarity_threshold
    self.recognition_interval = recognition_interval
    self.max_recognition_attempts = max_recognition_attempts
    
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
    
    print("\nLoading student gallery...")
    self.gallery = GalleryManager(gallery_path=gallery_path)
    num_students = len(self.gallery.get_all_students())
    print(f"Loaded {num_students} enrolled students")
    
    if num_students == 0:
      print("\nWARNING: Gallery is empty! Please enroll students first.")

    self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    self.session_dir = os.path.join(output_dir, self.session_name)
    os.makedirs(self.session_dir, exist_ok=True)

    if enable_performance_monitoring:
      model_id = f"{model_type.upper()}_{architecture.upper()}_InsightFace_Buffalo"
      device_str = "GPU" if (use_gpu and torch.cuda.is_available()) else "CPU"
      model_id += f"_{device_str}"
      self.perf_monitor = PerformanceMonitor(
        model_identifier=model_id,
        session_name=self.session_name,
        output_dir=self.session_dir,
        enable_gpu_monitoring=enable_gpu_monitoring,
        latency_window_size=100
      )
      self.perf_monitor.log_detailed_frames = False
    else:
      self.perf_monitor = None

    self.tracker = LiveRecognitionTracker(
      recognition_interval=recognition_interval,
      max_attempts=max_recognition_attempts,
      buffer_size=frame_buffer_size
    )
    
    self.recognized_faces_dir = os.path.join(self.session_dir, 'recognized_faces')
    self.unrecognized_faces_dir = os.path.join(self.session_dir, 'unrecognized_faces')
    os.makedirs(self.recognized_faces_dir, exist_ok=True)
    os.makedirs(self.unrecognized_faces_dir, exist_ok=True)

    self.snapshots_dir = os.path.join(self.session_dir, 'snapshots')
    if auto_snapshot:
      os.makedirs(self.snapshots_dir, exist_ok=True)

    self.auto_snapshot = auto_snapshot
    self.snapshot_interval = snapshot_interval
    self.last_snapshot_time = None
    self.snapshot_count = 0
        
    self.session_start = datetime.now()
    self.frame_count = 0
    self.total_faces_detected = 0
    self.total_recognition_attempts = 0
    self.active_tracks = {}
    self.next_track_id = 0
    
    print("\n" + "="*70)
    print(f"System ready!")
    print(f"Session: {self.session_name}")
    print(f"Output: {self.session_dir}")
    print(f"Recognition interval: every {recognition_interval} frames")
    print(f"Similarity threshold: {similarity_threshold}")
    print("="*70 + "\n")
    self._init_session_files()
  
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
  
  def _simple_track_assignment(self, faces: List[Dict], max_distance=100):
    new_assignments = {}
    
    for face in faces:
      bbox = face['bbox']
      center_x = (bbox[0] + bbox[2]) / 2
      center_y = (bbox[1] + bbox[3]) / 2

      best_track_id = None
      best_distance = max_distance
      
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
    return new_assignments
  
  def _recognize_face(self, face_data: Dict, track_id: int) -> Optional[Dict]:
    self.total_recognition_attempts += 1
    
    aligned_face = face_data['aligned_face']
    
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
  
  def process_frame(self, frame: np.ndarray) -> Dict:
    if self.perf_monitor:
      timings = self.perf_monitor.start_frame()

    self.frame_count += 1
    timestamp = datetime.now().isoformat()
 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if self.perf_monitor:
      self.perf_monitor.mark_capture_end(timings)
    faces = self.face_processor.process_numpy(frame_rgb, return_all=True)
    for face in faces:
      x1, y1, x2, y2 = map(int, face['bbox'])
      face['original_crop'] = frame_rgb[y1:y2, x1:x2].copy()
    self.total_faces_detected += len(faces)

    if self.perf_monitor:
      self.perf_monitor.mark_detection_end(timings)

    tracked_faces = self._simple_track_assignment(faces)
    
    recognition_events = []
    num_recognized_this_frame = 0
    num_unknown_this_frame = 0
    
    for track_id, track_data in tracked_faces.items():
      face = track_data['face']

      self.tracker.add_frame(track_id, face, timestamp)

      if self.tracker.should_recognize(track_id, self.frame_count):
        best_frame = self.tracker.get_best_frame(track_id)
        
        if best_frame is not None:
          result = self._recognize_face(best_frame, track_id)
          self.tracker.increment_attempts(track_id)
          
          if result is not None:
            if result['recognized']:
              num_recognized_this_frame += 1
              self.tracker.mark_recognized(track_id, result)
              face_filename = self._save_face_image(
                best_frame['aligned_face'],
                track_id,
                result['student_id'],
                result['name'],
                result['confidence'],
                recognized=True,
                original_crop=best_frame.get("original_crop")
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
                num_unknown_this_frame += 1
                face_filename = self._save_face_image(
                  best_frame['aligned_face'],
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

    if self.perf_monitor:
      perf_metrics = self.perf_monitor.end_frame(
        timings,
        num_faces_detected=len(faces),
        num_faces_recognized=num_recognized_this_frame,
        num_faces_unknown=num_unknown_this_frame
      )
    else:
      perf_metrics = {}
    
    return {
      'frame_count': self.frame_count,
      'faces_detected': len(faces),
      'active_tracks': len(tracked_faces),
      'recognition_events': len(recognition_events),
      'performance': perf_metrics
    }
  
  def _save_snapshot(self, frame: np.ndarray, frame_count: int, timestamp: str = None) -> str:
    if timestamp is None:
      timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    self.snapshot_count += 1
    
    filename = f'snapshot_{self.snapshot_count:04d}_frame_{frame_count:06d}_{timestamp}.jpg'
    filepath = os.path.join(self.snapshots_dir, filename)
    
    cv2.imwrite(filepath, frame)
    return filepath
  
  def _save_face_image(self, 
                        aligned_face: np.ndarray,
                        track_id: int,
                        student_id: str,
                        name: str,
                        confidence: float,
                        recognized: bool,
                        original_crop: np.ndarray = None) -> str:
    output_dir = self.recognized_faces_dir if recognized else self.unrecognized_faces_dir

    if recognized:
      student_dir = os.path.join(output_dir, f"{student_id}_{name.replace(' ', '_')}")
      os.makedirs(student_dir, exist_ok=True)
      output_dir = student_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    filename_aligned = f"track_{track_id:04d}_{timestamp}_conf{confidence:.3f}_aligned.jpg"
    filepath_aligned = os.path.join(output_dir, filename_aligned)
    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath_aligned, aligned_face_bgr)

    if original_crop is not None and original_crop.size > 0:
      filename_original = f"track_{track_id:04d}_{timestamp}_conf{confidence:.3f}_original.jpg"
      filepath_original = os.path.join(output_dir, filename_original)
      original_crop_bgr = cv2.cvtColor(original_crop, cv2.COLOR_RGB2BGR)
      cv2.imwrite(filepath_original, original_crop_bgr)

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
      while True:
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_path = self._save_snapshot(frame, self.frame_count, timestamp)
            print(f"[Auto-snapshot] Saved: {os.path.basename(snapshot_path)}")
            self.last_snapshot_time = current_time
  
        if display:
          display_frame = self._draw_display(frame, result, current_fps)
          cv2.imshow('Live Face Recognition', display_frame)
          key = cv2.waitKey(1) & 0xFF
        else:
          key = 0xFF

        if key == ord('q'):
          print("\nStopping...")
          break
        elif key == ord('s'):
          snapshot_path = os.path.join(self.session_dir, 
                                      f'snapshot_{self.frame_count:06d}.jpg')
          cv2.imwrite(snapshot_path, frame)
          print(f"Snapshot saved: {snapshot_path}")
        
        if max_frames and self.frame_count >= max_frames:
          print(f"\nReached max frames ({max_frames})")
          break
    
    finally:
      cap.release()
      if display:
        cv2.destroyAllWindows()
      
      self._finalize_session()
  
  def _draw_display(self, frame: np.ndarray, result: Dict, fps: float) -> np.ndarray:
    display = frame.copy()

    for track_id, track_data in self.active_tracks.items():
      face = track_data.get('face')
      if face is None:
        continue
      
      bbox = face['bbox']
      x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
      if track_id in self.tracker.recognized_tracks:
        student_info = self.tracker.recognized_tracks[track_id]
        color = (0, 255, 0)
        label = f"{student_info['name']} ({student_info['confidence']:.2f})"
      elif self.tracker.recognition_attempts.get(track_id, 0) >= self.max_recognition_attempts:
        color = (0, 0, 255)
        label = f"Unknown (track_{track_id:04d})"
      else:
        color = (0, 255, 255)
        attempts = self.tracker.recognition_attempts.get(track_id, 0)
        label = f"Detecting... ({attempts}/{self.max_recognition_attempts})"
    
      cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

      (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(display, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
      cv2.putText(display, label, (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    stats_y = 30
    stats = [
      f"Frame: {self.frame_count}",
      f"FPS: {fps:.1f}",
      f"Faces: {result['faces_detected']}",
      f"Active Tracks: {result['active_tracks']}",
      f"Recognized: {len(self.tracker.recognized_tracks)}"
    ]

    if self.perf_monitor and 'performance' in result:
      perf = result["performance"]
      current_stats = self.perf_monitor.get_current_stats()
      stats.extend([
        f"Latency: {perf.get('latency_e2e_ms', 0):.1f}ms",
        f"RAM: {current_stats.get('current_cpu_ram_mb', 0):.0f}MB"
      ])
      if self.perf_monitor.enable_gpu_monitoring:
          current_stats = self.perf_monitor.get_current_stats()
          stats.append(f"VRAM: {current_stats.get('current_gpu_vram_mb', 0):.0f}MB")
    
    for stat in stats:
      cv2.putText(display, stat, (10, stats_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      stats_y += 25
    
    return display
  
  def _finalize_session(self):
    print("\n" + "="*70)
    print("FINALIZING SESSION")
    print("="*70)
    
    session_end = datetime.now()
    duration = (session_end - self.session_start).total_seconds()

    if self.perf_monitor:
      perf_report = self.perf_monitor.finalize_session()
  
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
      'unrecognized_tracks': len(attendance_data['unrecognized']),
      'average_fps': self.frame_count / duration if duration > 0 else 0
    }
    
    self._write_session(session_data)
    print(f"\nSession: {self.session_name}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Frames processed: {self.frame_count}")
    print(f"Average FPS: {session_data['statistics']['average_fps']:.1f}")

    if self.auto_snapshot:
      print(f"Snapshots captured: {self.snapshot_count}")


    print(f"\nRecognized students: {len(attendance_data['recognized'])}")
    for student in attendance_data['recognized']:
      print(f"{student['name']} ({student['student_id']}) - confidence: {student['confidence']:.3f}")
    
    print(f"\nUnrecognized tracks: {len(attendance_data['unrecognized'])}")
    for track in attendance_data['unrecognized']:
      best = track['best_match']
      print(f"{track['track_id']} - best match: {best['name']} ({best['confidence']:.3f})")
    
    print(f"\nOutput directory: {self.session_dir}")
    print("="*70 + "\n")

def main():
  parser = argparse.ArgumentParser(
    description='Real-time face recognition for classroom attendance'
  )
  parser.add_argument(
    '--camera',
    type=int,
    default=0,
    help='Camera index (default: 0)'
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
    default=0.5,
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
    '--no_display',
    action='store_true',
    help='Run without video display'
  )
  parser.add_argument(
    '--max_frames',
    type=int,
    default=None,
    help='Maximum frames to process (for testing)'
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
    default=5.0,
    help='Interval between auto-snapshots in seconds (default: 5.0)'
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

  recognizer = LiveFaceRecognition(
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
    auto_snapshot=args.auto_snapshot,
    snapshot_interval=args.snapshot_interval
  )
  
  recognizer.run(
    camera_index=args.camera,
    display=not args.no_display,
    max_frames=args.max_frames
  )


if __name__ == '__main__':
  main()