import cv2
import numpy as np
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime
from collections import defaultdict, deque

from face_recognition import FaceProcessor

class SimpleTracker:
  def __init__(self, max_disappeared=30, max_distance=50):
    self.next_track_id = 1
    self.tracks = {}
    self.max_disappeared = max_disappeared
    self.max_distance = max_distance

  def compute_centroid(self, bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
  
  def compute_iou(self, bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
  
  def update(self, detections):
    if len(detections) == 0:
      for track_id in list(self.tracks.keys()):
        self.tracks[track_id]['disappeared'] += 1
        if self.tracks[track_id]['disappeared'] > self.max_disappeared:
          del self.tracks[track_id]
      return []
    
    if len(self.tracks) == 0:
      results = []
      for detection in detections:
        track_id = self.next_track_id
        self.next_track_id += 1
        
        centroid = self.compute_centroid(detection['bbox'])
        self.tracks[track_id] = {
            'bbox': detection['bbox'],
            'centroid': centroid,
            'disappeared': 0,
            'last_seen': datetime.now()
        }
        results.append((track_id, detection))
      return results

    track_ids = list(self.tracks.keys())
    track_centroids = np.array([self.tracks[tid]['centroid'] for tid in track_ids])
    detection_centroids = np.array([self.compute_centroid(d['bbox']) for d in detections])
    
    from scipy.spatial import distance as dist
    distances = dist.cdist(track_centroids, detection_centroids)
    
    matched_tracks = set()
    matched_detections = set()
    results = []
    
    while distances.size > 0 and distances.min() < self.max_distance:
      min_idx = distances.argmin()
      track_idx = min_idx // len(detections)
      det_idx = min_idx % len(detections)
      
      if track_idx in matched_tracks or det_idx in matched_detections:
        distances[track_idx, det_idx] = np.inf
        continue
      
      track_id = track_ids[track_idx]
      detection = detections[det_idx]
      
      centroid = self.compute_centroid(detection['bbox'])
      self.tracks[track_id].update({
        'bbox': detection['bbox'],
        'centroid': centroid,
        'disappeared': 0,
        'last_seen': datetime.now()
      })
      
      results.append((track_id, detection))
      matched_tracks.add(track_idx)
      matched_detections.add(det_idx)
      distances[track_idx, det_idx] = np.inf
  
    for idx, track_id in enumerate(track_ids):
      if idx not in matched_tracks:
        self.tracks[track_id]['disappeared'] += 1
        if self.tracks[track_id]['disappeared'] > self.max_disappeared:
          del self.tracks[track_id]
    
    for idx, detection in enumerate(detections):
      if idx not in matched_detections:
        track_id = self.next_track_id
        self.next_track_id += 1
        
        centroid = self.compute_centroid(detection['bbox'])
        self.tracks[track_id] = {
          'bbox': detection['bbox'],
          'centroid': centroid,
          'disappeared': 0,
          'last_seen': datetime.now()
        }
        results.append((track_id, detection))
    
    return results
  
class FrameAccumulator:
  def __init__(self, 
                target_frames=12,
                min_quality_score=0.5,
                output_dir='output/camera_captures'):
    self.target_frames = target_frames
    self.min_quality_score = min_quality_score
    self.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    self.accumulated_frames = defaultdict(list)
    self.completed_tracks = set()
    self.metadata = {}

  def compute_quality_score(self, face_dict):
    metrics = face_dict['quality_metrics']
    
    det_score = face_dict['det_score']
    blur_score = metrics.get('blur_score', 0)
    
    normalized_blur = min(blur_score / 200.0, 1.0)

    yaw = abs(metrics.get('yaw', 0))
    pitch = abs(metrics.get('pitch', 0))
    roll = abs(metrics.get('roll', 0))
    
    pose_score = 1.0 - (yaw / 90.0 + pitch / 90.0 + roll / 90.0) / 3.0
    pose_score = max(0, pose_score)

    quality = (det_score * 0.4 + normalized_blur * 0.3 + pose_score * 0.3)
    return quality
  
  def add_frame(self, track_id, face_dict, frame_rgb):
    if track_id in self.completed_tracks:
      return True
    
    quality = self.compute_quality_score(face_dict)
    
    if quality < self.min_quality_score:
      return False
    
    frame_data = {
      'aligned_face': face_dict['aligned_face'],
      'quality_score': quality,
      'det_score': face_dict['det_score'],
      'metrics': face_dict['quality_metrics'],
      'timestamp': datetime.now().isoformat()
    }
    
    self.accumulated_frames[track_id].append(frame_data)

    if len(self.accumulated_frames[track_id]) >= self.target_frames:
      if track_id not in self.completed_tracks:
        self.save_track(track_id)
      return True
    
    return False
  
  def save_track(self, track_id):
    if track_id in self.completed_tracks:
      return
    
    frames = self.accumulated_frames[track_id]
    if len(frames) == 0:
      return
    
    frames.sort(key=lambda x: x['quality_score'], reverse=True)
    frames_to_save = frames[:self.target_frames]
    
    track_dir = os.path.join(self.output_dir, f'track_{track_id:03d}')
    os.makedirs(track_dir, exist_ok=True)
    
    saved_files = []
    for idx, frame_data in enumerate(frames_to_save):
      filename = f'frame_{idx:03d}.jpg'
      filepath = os.path.join(track_dir, filename)

      aligned_bgr = cv2.cvtColor(frame_data['aligned_face'], cv2.COLOR_RGB2BGR)
      cv2.imwrite(filepath, aligned_bgr)
      saved_files.append(filename)
    
    metadata = {
      'track_id': track_id,
      'num_frames': len(frames_to_save),
      'avg_quality': float(np.mean([f['quality_score'] for f in frames_to_save])),
      'avg_det_score': float(np.mean([f['det_score'] for f in frames_to_save])),
      'saved_at': datetime.now().isoformat(),
      'files': saved_files
    }
    
    metadata_path = os.path.join(track_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
      json.dump(metadata, f, indent=2)
    
    self.metadata[track_id] = metadata
    self.completed_tracks.add(track_id)
    
    print(f"\nSaved {len(frames_to_save)} frames for track_{track_id:03d}")
    print(f"  Avg quality: {metadata['avg_quality']:.3f}")
    print(f"  Location: {track_dir}")
  
  def get_status(self, track_id):
    if track_id in self.completed_tracks:
        return 'completed'
    count = len(self.accumulated_frames[track_id])
    return f'{count}/{self.target_frames}'
  
class CameraFaceCapture:
  def __init__(self,
              camera_id=0,
              output_dir='output/camera_captures',
              skip_frames=5,
              target_frames_per_person=12,
              min_quality_score=0.5):
    
    self.camera_id = camera_id
    self.skip_frames = skip_frames
    self.frame_count = 0
    
    print("Initializing face processor...")
    self.processor = FaceProcessor(
      output_size=224,
      det_size=(640, 640),
      det_thresh=0.4,
      quality_filter_config={
        'min_det_score': 0.6,
        'min_face_size': 60,
        'max_yaw': 45,
        'max_pitch': 30,
        'max_roll': 30,
        'check_blur': True,
        'blur_threshold': 100
      }
    )
    
    print("Initializing tracker...")
    self.tracker = SimpleTracker(max_disappeared=30, max_distance=80)
    
    print("Initializing frame accumulator...")
    self.accumulator = FrameAccumulator(
      target_frames=target_frames_per_person,
      min_quality_score=min_quality_score,
      output_dir=output_dir
    )

    self.fps = 0
    self.last_time = datetime.now()

  def process_frame(self, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    if self.frame_count % self.skip_frames == 0:
      faces = self.processor.process_numpy(frame_rgb, return_all=True)
      
      tracked_faces = self.tracker.update(faces)
      
      for track_id, face in tracked_faces:
        if face['is_valid']:
          self.accumulator.add_frame(track_id, face, frame_rgb)
  
    return frame_rgb
  
  def draw_visualizations(self, frame_bgr):
    for track_id, track_data in self.tracker.tracks.items():
      bbox = track_data['bbox']
      x1, y1, x2, y2 = bbox

      status = self.accumulator.get_status(track_id)
      
      if status == 'completed':
        color = (0, 255, 0)
        status_text = "SAVED"
      else:
        color = (255, 165, 0)
        status_text = f"Collecting {status}"

      cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
      
      label = f"ID:{track_id} | {status_text}"
      (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(frame_bgr, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
      
      cv2.putText(frame_bgr, label, (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    stats_text = [
      f"FPS: {self.fps:.1f}",
      f"Active Tracks: {len(self.tracker.tracks)}",
      f"Completed: {len(self.accumulator.completed_tracks)}",
      f"Press 'q' to quit, 's' to force save all"
    ]
    
    y_offset = 30
    for text in stats_text:
      cv2.putText(frame_bgr, text, (10, y_offset),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
      y_offset += 30
    
    return frame_bgr
  

  def run(self):
    print(f"\nStarting camera {self.camera_id}...")
    cap = cv2.VideoCapture(self.camera_id)
    
    if not cap.isOpened():
      raise RuntimeError(f"Could not open camera {self.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*60)
    print("CAMERA FACE CAPTURE SYSTEM")
    print("="*60)
    print("Controls:")
    print("  q - Quit and save")
    print("  s - Force save all current tracks")
    print("  r - Reset (clear all tracks)")
    print("="*60 + "\n")
    
    try:
      while True:
        ret, frame = cap.read()
        if not ret:
          print("Failed to read frame")
          break
        
        self.process_frame(frame)
        
        display_frame = self.draw_visualizations(frame.copy())
        
        now = datetime.now()
        time_diff = (now - self.last_time).total_seconds()
        if time_diff > 0:
          self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.last_time = now

        cv2.imshow('Face Capture System', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          print("\nQuitting...")
          break
        elif key == ord('s'):
          print("\nForce saving all tracks...")
          for track_id in list(self.accumulator.accumulated_frames.keys()):
            if track_id not in self.accumulator.completed_tracks:
              self.accumulator.save_track(track_id)
        elif key == ord('r'):
          print("\nResetting all tracks...")
          self.tracker.tracks.clear()
          self.accumulator.accumulated_frames.clear()
          self.accumulator.completed_tracks.clear()
        
        self.frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        self.save_summary()
    
  def save_summary(self):
      summary = {
        'session_end': datetime.now().isoformat(),
        'total_frames_processed': self.frame_count,
        'total_tracks': self.tracker.next_track_id - 1,
        'completed_tracks': len(self.accumulator.completed_tracks),
        'tracks': self.accumulator.metadata
      }
      
      summary_path = os.path.join(self.accumulator.output_dir, 'session_summary.json')
      with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
      
      print("\n" + "="*60)
      print("SESSION SUMMARY")
      print("="*60)
      print(f"Total tracks: {summary['total_tracks']}")
      print(f"Completed tracks: {summary['completed_tracks']}")
      print(f"Frames processed: {summary['total_frames_processed']}")
      print(f"Output directory: {self.accumulator.output_dir}")
      print(f"Summary saved to: {summary_path}")
      print("="*60 + "\n")

if __name__ == '__main__':
  capture_system = CameraFaceCapture(
    camera_id=0,
    output_dir='output/camera_captures',
    skip_frames=5,
    target_frames_per_person=12,
    min_quality_score=0.5
  )
  
  capture_system.run()