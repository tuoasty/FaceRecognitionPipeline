import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import threading

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class PerformanceMonitor:
  def __init__(self, 
              model_identifier: str,
              session_name: str,
              output_dir: str,
              enable_gpu_monitoring: bool = True,
              latency_window_size: int = 100):
    self.model_identifier = model_identifier
    self.session_name = session_name
    self.output_dir = output_dir
    self.enable_gpu_monitoring = enable_gpu_monitoring and NVML_AVAILABLE
  
    os.makedirs(output_dir, exist_ok=True)
 
    self.session_start = datetime.now()
    self.session_end = None

    self.total_frames = 0
    self.total_faces_detected = 0
    self.total_faces_recognized = 0
    self.total_faces_unknown = 0
    self.latency_window = deque(maxlen=latency_window_size)
    self.latency_capture = deque(maxlen=latency_window_size)
    self.latency_detection = deque(maxlen=latency_window_size)
    self.latency_recognition = deque(maxlen=latency_window_size)
    self.latency_e2e = deque(maxlen=latency_window_size)

    self.fps_start_time = time.time()
    self.fps_frame_count = 0
    self.current_fps = 0.0
    self.fps_history = []

    self.process = psutil.Process()
    self.baseline_cpu_ram_mb = self.get_cpu_ram_usage()
    self.peak_cpu_ram_mb = self.baseline_cpu_ram_mb
    self.baseline_gpu_vram_mb = 0.0
    self.peak_gpu_vram_mb = 0.0

    self.gpu_handle = None
    if self.enable_gpu_monitoring:
      try:
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline_gpu_vram_mb = self.get_gpu_vram_usage()
        print(f"GPU monitoring enabled. Baseline VRAM: {self.baseline_gpu_vram_mb:.2f} MB")
      except Exception as e:
        print(f"Failed to initialize GPU monitoring: {e}")
        self.enable_gpu_monitoring = False

    self.detailed_frame_logs = []
    self.log_detailed_frames = False 
    self.lock = threading.Lock()
    
    print(f"Performance Monitor initialized")
    print(f"  Model: {model_identifier}")
    print(f"  Session: {session_name}")
    print(f"  Baseline CPU RAM: {self.baseline_cpu_ram_mb:.2f} MB")
    if self.enable_gpu_monitoring:
      print(f"  Baseline GPU VRAM: {self.baseline_gpu_vram_mb:.2f} MB")

  def get_cpu_ram_usage(self) -> float:
    return self.process.memory_info().rss / (1024 * 1024)

  def get_gpu_vram_usage(self) -> float:
    if not self.enable_gpu_monitoring or self.gpu_handle is None:
      return 0.0
    try:
      info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
      return info.used / (1024 * 1024)
    except Exception as e:
      print(f"Error reading GPU memory: {e}")
      return 0.0

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
    timings['recognition_start'] = time.perf_counter()

  def mark_recognition_end(self, timings: Dict[str, float]):
    timings['recognition_end'] = time.perf_counter()

  def end_frame(self, 
                timings: Dict[str, float],
                num_faces_detected: int = 0,
                num_faces_recognized: int = 0,
                num_faces_unknown: int = 0) -> Dict[str, float]:
    with self.lock:
      frame_end = time.perf_counter()
      latency_capture_ms = (timings.get('capture_end', timings['frame_start']) - 
                            timings.get('capture_start', timings['frame_start'])) * 1000
      
      latency_detection_ms = (timings.get('detection_end', frame_end) - 
                              timings.get('detection_start', frame_end)) * 1000
      
      latency_recognition_ms = (timings.get('recognition_end', frame_end) - 
                                timings.get('recognition_start', frame_end)) * 1000
      
      latency_e2e_ms = (frame_end - timings['frame_start']) * 1000
      self.latency_capture.append(latency_capture_ms)
      self.latency_detection.append(latency_detection_ms)
      self.latency_recognition.append(latency_recognition_ms)
      self.latency_e2e.append(latency_e2e_ms)
      self.total_frames += 1
      self.total_faces_detected += num_faces_detected
      self.total_faces_recognized += num_faces_recognized
      self.total_faces_unknown += num_faces_unknown

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
      
      if self.enable_gpu_monitoring:
        current_gpu_vram = self.get_gpu_vram_usage()
        self.peak_gpu_vram_mb = max(self.peak_gpu_vram_mb, current_gpu_vram)

      if self.log_detailed_frames:
        self.detailed_frame_logs.append({
          'frame_number': self.total_frames,
          'timestamp': datetime.now().isoformat(),
          'latency_e2e_ms': latency_e2e_ms,
          'latency_capture_ms': latency_capture_ms,
          'latency_detection_ms': latency_detection_ms,
          'latency_recognition_ms': latency_recognition_ms,
          'faces_detected': num_faces_detected,
          'faces_recognized': num_faces_recognized,
          'faces_unknown': num_faces_unknown,
          'cpu_ram_mb': current_cpu_ram,
          'gpu_vram_mb': self.get_gpu_vram_usage() if self.enable_gpu_monitoring else 0
        })
      
      return {
        'latency_e2e_ms': latency_e2e_ms,
        'latency_capture_ms': latency_capture_ms,
        'latency_detection_ms': latency_detection_ms,
        'latency_recognition_ms': latency_recognition_ms,
        'current_fps': self.current_fps
      }

  def get_current_stats(self) -> Dict:
    with self.lock:
      avg_latency_e2e = sum(self.latency_e2e) / len(self.latency_e2e) if self.latency_e2e else 0
      avg_latency_capture = sum(self.latency_capture) / len(self.latency_capture) if self.latency_capture else 0
      avg_latency_detection = sum(self.latency_detection) / len(self.latency_detection) if self.latency_detection else 0
      avg_latency_recognition = sum(self.latency_recognition) / len(self.latency_recognition) if self.latency_recognition else 0
      
      return {
        'total_frames': self.total_frames,
        'total_faces_detected': self.total_faces_detected,
        'total_faces_recognized': self.total_faces_recognized,
        'total_faces_unknown': self.total_faces_unknown,
        'current_fps': self.current_fps,
        'avg_latency_e2e_ms': avg_latency_e2e,
        'avg_latency_capture_ms': avg_latency_capture,
        'avg_latency_detection_ms': avg_latency_detection,
        'avg_latency_recognition_ms': avg_latency_recognition,
        'current_cpu_ram_mb': self.get_cpu_ram_usage(),
        'peak_cpu_ram_mb': self.peak_cpu_ram_mb,
        'current_gpu_vram_mb': self.get_gpu_vram_usage() if self.enable_gpu_monitoring else 0,
        'peak_gpu_vram_mb': self.peak_gpu_vram_mb if self.enable_gpu_monitoring else 0
      }

  def finalize_session(self) -> Dict:
      self.session_end = datetime.now()
      duration_seconds = (self.session_end - self.session_start).total_seconds()
      avg_latency_e2e = sum(self.latency_e2e) / len(self.latency_e2e) if self.latency_e2e else 0
      avg_latency_capture = sum(self.latency_capture) / len(self.latency_capture) if self.latency_capture else 0
      avg_latency_detection = sum(self.latency_detection) / len(self.latency_detection) if self.latency_detection else 0
      avg_latency_recognition = sum(self.latency_recognition) / len(self.latency_recognition) if self.latency_recognition else 0
      
      max_latency_e2e = max(self.latency_e2e) if self.latency_e2e else 0
      min_latency_e2e = min(self.latency_e2e) if self.latency_e2e else 0
      
      avg_fps = self.total_frames / duration_seconds if duration_seconds > 0 else 0
      
      performance_data = {
        'session_info': {
          'session_name': self.session_name,
          'model_identifier': self.model_identifier,
          'start_time': self.session_start.isoformat(),
          'end_time': self.session_end.isoformat(),
          'duration_seconds': duration_seconds
        },
        'frame_statistics': {
          'total_frames_processed': self.total_frames,
          'total_faces_detected': self.total_faces_detected,
          'total_faces_recognized': self.total_faces_recognized,
          'total_faces_unknown': self.total_faces_unknown,
          'avg_faces_per_frame': self.total_faces_detected / self.total_frames if self.total_frames > 0 else 0,
          'recognition_rate': self.total_faces_recognized / self.total_faces_detected if self.total_faces_detected > 0 else 0
        },
        'fps_metrics': {
          'average_fps': avg_fps,
          'current_fps': self.current_fps,
          'fps_history': self.fps_history
        },
        'latency_metrics': {
          'end_to_end': {
            'average_ms': avg_latency_e2e,
            'max_ms': max_latency_e2e,
            'min_ms': min_latency_e2e,
            'unit': 'milliseconds'
          },
          'capture': {
            'average_ms': avg_latency_capture,
            'unit': 'milliseconds'
          },
          'detection': {
            'average_ms': avg_latency_detection,
            'unit': 'milliseconds'
          },
          'recognition': {
            'average_ms': avg_latency_recognition,
            'unit': 'milliseconds'
          }
        },
        'memory_usage': {
          'cpu_ram': {
            'baseline_mb': self.baseline_cpu_ram_mb,
            'peak_mb': self.peak_cpu_ram_mb,
            'delta_mb': self.peak_cpu_ram_mb - self.baseline_cpu_ram_mb,
            'unit': 'megabytes'
          },
          'gpu_vram': {
            'baseline_mb': self.baseline_gpu_vram_mb,
            'peak_mb': self.peak_gpu_vram_mb,
            'delta_mb': self.peak_gpu_vram_mb - self.baseline_gpu_vram_mb,
            'unit': 'megabytes',
            'available': self.enable_gpu_monitoring
          }
        },
        'system_info': {
          'cpu_count': psutil.cpu_count(),
          'total_ram_gb': psutil.virtual_memory().total / (1024**3),
          'gpu_available': self.enable_gpu_monitoring
        }
      }

      report_path = os.path.join(self.output_dir, 'performance_report.json')
      with open(report_path, 'w') as f:
        json.dump(performance_data, f, indent=2)
  
      if self.log_detailed_frames and self.detailed_frame_logs:
        detailed_path = os.path.join(self.output_dir, 'detailed_frame_logs.json')
        with open(detailed_path, 'w') as f:
          json.dump(self.detailed_frame_logs, f, indent=2)
        print(f"Detailed frame logs saved to: {detailed_path}")

      print("\n" + "="*70)
      print("PERFORMANCE REPORT")
      print("="*70)
      print(f"\nModel: {self.model_identifier}")
      print(f"Session: {self.session_name}")
      print(f"Duration: {duration_seconds:.2f} seconds")
      
      print(f"\n--- Frame Statistics ---")
      print(f"Total frames processed: {self.total_frames}")
      print(f"Total faces detected: {self.total_faces_detected}")
      print(f"Total faces recognized: {self.total_faces_recognized}")
      print(f"Total faces unknown: {self.total_faces_unknown}")
      print(f"Recognition rate: {performance_data['frame_statistics']['recognition_rate']*100:.2f}%")
      
      print(f"\n--- Performance Metrics ---")
      print(f"Average FPS: {avg_fps:.2f}")
      print(f"Average E2E latency: {avg_latency_e2e:.2f} ms")
      print(f"  - Capture: {avg_latency_capture:.2f} ms")
      print(f"  - Detection: {avg_latency_detection:.2f} ms")
      print(f"  - Recognition: {avg_latency_recognition:.2f} ms")
      print(f"Latency range: {min_latency_e2e:.2f} - {max_latency_e2e:.2f} ms")
      
      print(f"\n--- Memory Usage ---")
      print(f"CPU RAM:")
      print(f"  Baseline: {self.baseline_cpu_ram_mb:.2f} MB")
      print(f"  Peak: {self.peak_cpu_ram_mb:.2f} MB")
      print(f"  Delta: {self.peak_cpu_ram_mb - self.baseline_cpu_ram_mb:.2f} MB")
      
      if self.enable_gpu_monitoring:
        print(f"GPU VRAM:")
        print(f"  Baseline: {self.baseline_gpu_vram_mb:.2f} MB")
        print(f"  Peak: {self.peak_gpu_vram_mb:.2f} MB")
        print(f"  Delta: {self.peak_gpu_vram_mb - self.baseline_gpu_vram_mb:.2f} MB")
      
      print(f"\nReport saved to: {report_path}")
      print("="*70 + "\n")
      
      if self.enable_gpu_monitoring:
        try:
          pynvml.nvmlShutdown()
        except:
          pass
      
      return performance_data

def __del__(self):
  if self.enable_gpu_monitoring:
    try:
      pynvml.nvmlShutdown()
    except:
      pass