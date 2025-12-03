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

class PerformanceMonitorServer:
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
      self.total_requests = 0
      self.total_faces_processed = 0
      self.total_faces_recognized = 0
      self.total_faces_unknown = 0
      
      self.latency_recognition = deque(maxlen=latency_window_size)
      self.latency_network = deque(maxlen=latency_window_size)
      self.latency_e2e_server = deque(maxlen=latency_window_size) 

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
      
      self.detailed_request_logs = []
      self.log_detailed_requests = False
      self.lock = threading.Lock()
      
      print(f"Server Performance Monitor initialized")
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
  
  def start_request(self) -> Dict[str, float]:
    return {
      'request_start': time.perf_counter(),
      'recognition_start': None
    }
  
  def mark_recognition_start(self, timings: Dict[str, float]):
    timings['recognition_start'] = time.perf_counter()
  
  def mark_recognition_end(self, timings: Dict[str, float]):
    timings['recognition_end'] = time.perf_counter()
  
  def end_request(self,
                  timings: Dict[str, float],
                  num_faces_processed: int = 0,
                  num_faces_recognized: int = 0,
                  num_faces_unknown: int = 0) -> Dict[str, float]:
    with self.lock:
      request_end = time.perf_counter()
    
      latency_recognition_ms = 0
      if timings.get('recognition_start') and timings.get('recognition_end'):
        latency_recognition_ms = (timings['recognition_end'] - timings['recognition_start']) * 1000
        self.latency_recognition.append(latency_recognition_ms)
      
      latency_e2e_server_ms = (request_end - timings['request_start']) * 1000
      self.latency_e2e_server.append(latency_e2e_server_ms)

      latency_network_ms = latency_e2e_server_ms - latency_recognition_ms
      self.latency_network.append(latency_network_ms)

      self.total_requests += 1
      self.total_faces_processed += num_faces_processed
      self.total_faces_recognized += num_faces_recognized
      self.total_faces_unknown += num_faces_unknown

      current_cpu_ram = self.get_cpu_ram_usage()
      self.peak_cpu_ram_mb = max(self.peak_cpu_ram_mb, current_cpu_ram)
      
      if self.enable_gpu_monitoring:
        current_gpu_vram = self.get_gpu_vram_usage()
        self.peak_gpu_vram_mb = max(self.peak_gpu_vram_mb, current_gpu_vram)

      if self.log_detailed_requests:
        self.detailed_request_logs.append({
          'request_number': self.total_requests,
          'timestamp': datetime.now().isoformat(),
          'latency_e2e_server_ms': latency_e2e_server_ms,
          'latency_recognition_ms': latency_recognition_ms,
          'latency_network_ms': latency_network_ms,
          'faces_processed': num_faces_processed,
          'faces_recognized': num_faces_recognized,
          'faces_unknown': num_faces_unknown,
          'cpu_ram_mb': current_cpu_ram,
          'gpu_vram_mb': self.get_gpu_vram_usage() if self.enable_gpu_monitoring else 0
        })
      
      return {
        'latency_e2e_server_ms': latency_e2e_server_ms,
        'latency_recognition_ms': latency_recognition_ms,
        'latency_network_ms': latency_network_ms
      }
  
  def get_current_stats(self) -> Dict:
    with self.lock:
      avg_latency_recognition = sum(self.latency_recognition) / len(self.latency_recognition) if self.latency_recognition else 0
      avg_latency_network = sum(self.latency_network) / len(self.latency_network) if self.latency_network else 0
      avg_latency_e2e_server = sum(self.latency_e2e_server) / len(self.latency_e2e_server) if self.latency_e2e_server else 0
      
      return {
        'total_requests': self.total_requests,
        'total_faces_processed': self.total_faces_processed,
        'total_faces_recognized': self.total_faces_recognized,
        'total_faces_unknown': self.total_faces_unknown,
        'avg_latency_recognition_ms': avg_latency_recognition,
        'avg_latency_network_ms': avg_latency_network,
        'avg_latency_e2e_server_ms': avg_latency_e2e_server,
        'current_cpu_ram_mb': self.get_cpu_ram_usage(),
        'peak_cpu_ram_mb': self.peak_cpu_ram_mb,
        'current_gpu_vram_mb': self.get_gpu_vram_usage() if self.enable_gpu_monitoring else 0,
        'peak_gpu_vram_mb': self.peak_gpu_vram_mb if self.enable_gpu_monitoring else 0
      }
  
  def finalize_session(self, client_report: Optional[Dict] = None) -> Dict:
    self.session_end = datetime.now()
    duration_seconds = (self.session_end - self.session_start).total_seconds()

    avg_latency_recognition = sum(self.latency_recognition) / len(self.latency_recognition) if self.latency_recognition else 0
    avg_latency_network = sum(self.latency_network) / len(self.latency_network) if self.latency_network else 0
    avg_latency_e2e_server = sum(self.latency_e2e_server) / len(self.latency_e2e_server) if self.latency_e2e_server else 0
    
    max_latency_recognition = max(self.latency_recognition) if self.latency_recognition else 0
    min_latency_recognition = min(self.latency_recognition) if self.latency_recognition else 0
    
    performance_data = {
      'session_info': {
        'session_name': self.session_name,
        'model_identifier': self.model_identifier,
        'start_time': self.session_start.isoformat(),
        'end_time': self.session_end.isoformat(),
        'duration_seconds': duration_seconds,
        'component': 'server'
      },
      'request_statistics': {
        'total_requests_processed': self.total_requests,
        'total_faces_processed': self.total_faces_processed,
        'total_faces_recognized': self.total_faces_recognized,
        'total_faces_unknown': self.total_faces_unknown,
        'avg_faces_per_request': self.total_faces_processed / self.total_requests if self.total_requests > 0 else 0,
        'recognition_rate': self.total_faces_recognized / self.total_faces_processed if self.total_faces_processed > 0 else 0,
        'requests_per_second': self.total_requests / duration_seconds if duration_seconds > 0 else 0
      },
      'latency_metrics': {
        'recognition': {
          'average_ms': avg_latency_recognition,
          'max_ms': max_latency_recognition,
          'min_ms': min_latency_recognition,
          'unit': 'milliseconds'
        },
        'network_overhead': {
          'average_ms': avg_latency_network,
          'unit': 'milliseconds'
        },
        'end_to_end_server': {
          'average_ms': avg_latency_e2e_server,
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
    
    report_path = os.path.join(self.output_dir, 'performance_report_server.json')
    with open(report_path, 'w') as f:
      json.dump(performance_data, f, indent=2)

    if client_report:
      client_report_path = os.path.join(self.output_dir, 'performance_report_client.json')
      with open(client_report_path, 'w') as f:
        json.dump(client_report, f, indent=2)
      print(f"Client performance report saved to: {client_report_path}")

    if self.log_detailed_requests and self.detailed_request_logs:
      detailed_path = os.path.join(self.output_dir, 'detailed_request_logs_server.json')
      with open(detailed_path, 'w') as f:
        json.dump(self.detailed_request_logs, f, indent=2)
      print(f"Detailed request logs saved to: {detailed_path}")

    print("\n" + "="*70)
    print("SERVER PERFORMANCE REPORT")
    print("="*70)
    print(f"\nModel: {self.model_identifier}")
    print(f"Session: {self.session_name}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    
    print(f"\n--- Request Statistics ---")
    print(f"Total requests processed: {self.total_requests}")
    print(f"Total faces processed: {self.total_faces_processed}")
    print(f"Total faces recognized: {self.total_faces_recognized}")
    print(f"Total faces unknown: {self.total_faces_unknown}")
    print(f"Recognition rate: {performance_data['request_statistics']['recognition_rate']*100:.2f}%")
    print(f"Requests per second: {performance_data['request_statistics']['requests_per_second']:.2f}")
    
    print(f"\n--- Performance Metrics ---")
    print(f"Average recognition latency: {avg_latency_recognition:.2f} ms")
    print(f"Average network overhead: {avg_latency_network:.2f} ms")
    print(f"Average total server processing: {avg_latency_e2e_server:.2f} ms")
    print(f"Recognition latency range: {min_latency_recognition:.2f} - {max_latency_recognition:.2f} ms")
    
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
    
    print(f"\nServer report saved to: {report_path}")
    print("="*70 + "\n")
    
    # Cleanup GPU monitoring
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