import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import threading

class PerformanceMonitorClient:
  def __init__(self,
                session_name: str,
                output_dir: str,
                latency_window_size: int = 100):
    self.session_name = session_name
    self.output_dir = output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    self.session_start = datetime.now()
    self.session_end = None
  
    self.total_frames = 0
    self.total_faces_detected = 0
    self.total_network_requests = 0

    self.latency_capture = deque(maxlen=latency_window_size)
    self.latency_detection = deque(maxlen=latency_window_size)
    self.latency_network_send = deque(maxlen=latency_window_size)
    self.latency_e2e_client = deque(maxlen=latency_window_size)
    
    self.fps_start_time = time.time()
    self.fps_frame_count = 0
    self.current_fps = 0.0
    self.fps_history = []

    self.process = psutil.Process()
    self.baseline_cpu_ram_mb = self.get_cpu_ram_usage()
    self.peak_cpu_ram_mb = self.baseline_cpu_ram_mb
    
    self.detailed_frame_logs = []
    self.log_detailed_frames = False
    self.lock = threading.Lock()
    
    print(f"Client Performance Monitor initialized")
    print(f"  Session: {session_name}")
    print(f"  Baseline CPU RAM: {self.baseline_cpu_ram_mb:.2f} MB")
  
  def get_cpu_ram_usage(self) -> float:
    return self.process.memory_info().rss / (1024 * 1024)
  
  def start_frame(self) -> Dict[str, float]:
    return {
      'frame_start': time.perf_counter(),
      'capture_start': time.perf_counter()
    }
  
  def mark_capture_end(self, timings: Dict[str, float]):
    timings['capture_end'] = time.perf_counter()
    timings['detection_start'] = time.perf_counter()
  
  def mark_detection_end(self, timings: Dict[str, float]):
    timings['detection_end'] = time.perf_counter()
  
  def mark_network_start(self, timings: Dict[str, float]):
    timings['network_start'] = time.perf_counter()
  
  def mark_network_end(self, timings: Dict[str, float]):
    timings['network_end'] = time.perf_counter()
  
  def end_frame(self,
              timings: Dict[str, float],
              num_faces_detected: int = 0,
              network_request_sent: bool = False) -> Dict[str, float]:
    with self.lock:
      frame_end = time.perf_counter()
      
      latency_capture_ms = (timings.get('capture_end', timings['frame_start']) - 
                          timings.get('capture_start', timings['frame_start'])) * 1000
      
      latency_detection_ms = (timings.get('detection_end', frame_end) - 
                              timings.get('detection_start', frame_end)) * 1000
    
      latency_network_send_ms = 0
      if timings.get('network_start') and timings.get('network_end'):
        latency_network_send_ms = (timings['network_end'] - timings['network_start']) * 1000
        self.latency_network_send.append(latency_network_send_ms)
    
      latency_e2e_client_ms = (frame_end - timings['frame_start']) * 1000
      
      self.latency_capture.append(latency_capture_ms)
      self.latency_detection.append(latency_detection_ms)
      self.latency_e2e_client.append(latency_e2e_client_ms)

      self.total_frames += 1
      self.total_faces_detected += num_faces_detected
      if network_request_sent:
        self.total_network_requests += 1

      self.fps_frame_count += 1
      if self.fps_frame_count >= 30:
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        self.current_fps = self.fps_frame_count / elapsed
        self.fps_history.append({
          'timestamp': current_time,
          'fps': self.current_fps,
          'frame_number': self.total_frames
        })
        self.fps_start_time = current_time
        self.fps_frame_count = 0

      current_cpu_ram = self.get_cpu_ram_usage()
      self.peak_cpu_ram_mb = max(self.peak_cpu_ram_mb, current_cpu_ram)
      
      if self.log_detailed_frames:
        self.detailed_frame_logs.append({
          'frame_number': self.total_frames,
          'timestamp': datetime.now().isoformat(),
          'latency_e2e_client_ms': latency_e2e_client_ms,
          'latency_capture_ms': latency_capture_ms,
          'latency_detection_ms': latency_detection_ms,
          'latency_network_send_ms': latency_network_send_ms,
          'faces_detected': num_faces_detected,
          'cpu_ram_mb': current_cpu_ram
        })
      
      return {
        'latency_e2e_client_ms': latency_e2e_client_ms,
        'latency_capture_ms': latency_capture_ms,
        'latency_detection_ms': latency_detection_ms,
        'latency_network_send_ms': latency_network_send_ms,
        'current_fps': self.current_fps
      }

  def get_current_stats(self) -> Dict:
    with self.lock:
      avg_latency_capture = sum(self.latency_capture) / len(self.latency_capture) if self.latency_capture else 0
      avg_latency_detection = sum(self.latency_detection) / len(self.latency_detection) if self.latency_detection else 0
      avg_latency_network_send = sum(self.latency_network_send) / len(self.latency_network_send) if self.latency_network_send else 0
      avg_latency_e2e_client = sum(self.latency_e2e_client) / len(self.latency_e2e_client) if self.latency_e2e_client else 0
      
      return {
        'total_frames': self.total_frames,
        'total_faces_detected': self.total_faces_detected,
        'total_network_requests': self.total_network_requests,
        'current_fps': self.current_fps,
        'avg_latency_capture_ms': avg_latency_capture,
        'avg_latency_detection_ms': avg_latency_detection,
        'avg_latency_network_send_ms': avg_latency_network_send,
        'avg_latency_e2e_client_ms': avg_latency_e2e_client,
        'current_cpu_ram_mb': self.get_cpu_ram_usage(),
        'peak_cpu_ram_mb': self.peak_cpu_ram_mb
      }

  def finalize_session(self) -> Dict:
    self.session_end = datetime.now()
    duration_seconds = (self.session_end - self.session_start).total_seconds()

    avg_latency_capture = sum(self.latency_capture) / len(self.latency_capture) if self.latency_capture else 0
    avg_latency_detection = sum(self.latency_detection) / len(self.latency_detection) if self.latency_detection else 0
    avg_latency_network_send = sum(self.latency_network_send) / len(self.latency_network_send) if self.latency_network_send else 0
    avg_latency_e2e_client = sum(self.latency_e2e_client) / len(self.latency_e2e_client) if self.latency_e2e_client else 0
    
    max_latency_detection = max(self.latency_detection) if self.latency_detection else 0
    min_latency_detection = min(self.latency_detection) if self.latency_detection else 0
    
    avg_fps = self.total_frames / duration_seconds if duration_seconds > 0 else 0
    
    performance_data = {
      'session_info': {
        'session_name': self.session_name,
        'start_time': self.session_start.isoformat(),
        'end_time': self.session_end.isoformat(),
        'duration_seconds': duration_seconds,
        'component': 'client'
      },
      'frame_statistics': {
        'total_frames_processed': self.total_frames,
        'total_faces_detected': self.total_faces_detected,
        'total_network_requests': self.total_network_requests,
        'avg_faces_per_frame': self.total_faces_detected / self.total_frames if self.total_frames > 0 else 0
      },
      'fps_metrics': {
        'average_fps': avg_fps,
        'current_fps': self.current_fps,
        'fps_history': self.fps_history
      },
      'latency_metrics': {
        'capture': {
          'average_ms': avg_latency_capture,
          'unit': 'milliseconds'
        },
        'detection': {
          'average_ms': avg_latency_detection,
          'max_ms': max_latency_detection,
          'min_ms': min_latency_detection,
          'unit': 'milliseconds'
        },
        'network_send': {
          'average_ms': avg_latency_network_send,
          'unit': 'milliseconds'
        },
        'end_to_end_client': {
          'average_ms': avg_latency_e2e_client,
          'unit': 'milliseconds'
        }
      },
      'memory_usage': {
        'cpu_ram': {
          'baseline_mb': self.baseline_cpu_ram_mb,
          'peak_mb': self.peak_cpu_ram_mb,
          'delta_mb': self.peak_cpu_ram_mb - self.baseline_cpu_ram_mb,
          'unit': 'megabytes'
        }
      },
      'system_info': {
        'cpu_count': psutil.cpu_count(),
        'total_ram_gb': psutil.virtual_memory().total / (1024**3)
      }
    }

    temp_report_path = os.path.join(self.output_dir, 'performance_report_client_temp.json')
    with open(temp_report_path, 'w') as f:
      json.dump(performance_data, f, indent=2)

    if self.log_detailed_frames and self.detailed_frame_logs:
      detailed_path = os.path.join(self.output_dir, 'detailed_frame_logs_client.json')
      with open(detailed_path, 'w') as f:
        json.dump(self.detailed_frame_logs, f, indent=2)
      print(f"Detailed frame logs saved to: {detailed_path}")
    
    print("\n" + "="*70)
    print("CLIENT PERFORMANCE REPORT")
    print("="*70)
    print(f"\nSession: {self.session_name}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    
    print(f"\n--- Frame Statistics ---")
    print(f"Total frames processed: {self.total_frames}")
    print(f"Total faces detected: {self.total_faces_detected}")
    print(f"Total network requests: {self.total_network_requests}")
    print(f"Average faces per frame: {performance_data['frame_statistics']['avg_faces_per_frame']:.2f}")
    
    print(f"\n--- Performance Metrics ---")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average client E2E latency: {avg_latency_e2e_client:.2f} ms")
    print(f"  - Capture: {avg_latency_capture:.2f} ms")
    print(f"  - Detection: {avg_latency_detection:.2f} ms")
    print(f"  - Network send: {avg_latency_network_send:.2f} ms")
    print(f"Detection latency range: {min_latency_detection:.2f} - {max_latency_detection:.2f} ms")
    
    print(f"\n--- Memory Usage ---")
    print(f"CPU RAM:")
    print(f"  Baseline: {self.baseline_cpu_ram_mb:.2f} MB")
    print(f"  Peak: {self.peak_cpu_ram_mb:.2f} MB")
    print(f"  Delta: {self.peak_cpu_ram_mb - self.baseline_cpu_ram_mb:.2f} MB")
    
    print(f"\nClient report will be sent to server")
    print("="*70 + "\n")
    
    return performance_data