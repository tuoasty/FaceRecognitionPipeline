import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import os

from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceDetector:
  def __init__(self, det_size=(640, 640), det_thresh=0.5, 
               providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    self.app = FaceAnalysis(
      name='buffalo_l',
      providers=providers
      )
    self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
  
  def detect(self, image:np.ndarray) -> List[Dict]:
    if len(image.shape) == 2:
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = self.app.get(image)
    results = []
    for face in faces:
      results.append({
        'bbox': face.bbox.astype(np.int32),
        'landmarks': face.kps.astype(np.float32),
        'det_score': float(face.det_score),
        'pose': getattr(face, 'pose', None),
        'age': getattr(face, 'age', None),
        'gender': getattr(face, 'gender', None)
      })
    
    return results
  
class FaceAligner:
  def __init__(self, output_size=112):
    self.output_size = output_size
    self.template = np.array([
        [0.34 * output_size, 0.46 * output_size],
        [0.66 * output_size, 0.46 * output_size],
        [0.50 * output_size, 0.61 * output_size],
        [0.37 * output_size, 0.74 * output_size],
        [0.63 * output_size, 0.74 * output_size],
    ], dtype=np.float32)

  def align(self, image: np.ndarray, landmarks: np.ndarray, method='similarity') -> np.ndarray:
    landmarks = landmarks.astype(np.float32)
    if method == 'similarity':
        tform = cv2.estimateAffinePartial2D(landmarks, self.template)[0]
    else:
        tform = cv2.getAffineTransform(landmarks[:3], self.template[:3])
    aligned = cv2.warpAffine(
        image, 
        tform, 
        (self.output_size, self.output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return aligned
  
class FaceQualityFilter:
  def __init__(self,
                min_det_score=0.6,
                min_face_size=60,
                max_yaw=45,
                max_pitch=30,
                max_roll=30,
                check_blur=True,
                blur_threshold=100):
      self.min_det_score = min_det_score
      self.min_face_size = min_face_size
      self.max_yaw = max_yaw
      self.max_pitch = max_pitch
      self.max_roll = max_roll
      self.check_blur = check_blur
      self.blur_threshold = blur_threshold
  
  def compute_blur_score(self, face_image: np.ndarray) -> float:
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_image
    return cv2.Laplacian(gray, cv2.CV_64F).var()
  
  def compute_pose_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    eye_center = (left_eye + right_eye) / 2
    eye_delta = right_eye - left_eye
    roll = np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))
    
    nose_offset_x = nose[0] - eye_center[0]
    eye_distance = np.linalg.norm(eye_delta)
    yaw = np.degrees(np.arcsin(np.clip(nose_offset_x / eye_distance, -1, 1))) * 2
    
    mouth_center = (left_mouth + right_mouth) / 2
    face_height = mouth_center[1] - eye_center[1]
    nose_offset_y = nose[1] - eye_center[1]
    pitch = (nose_offset_y / face_height - 0.5) * 60 
    
    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
  
  def is_valid(self, 
                face_dict: Dict,
                face_image: Optional[np.ndarray] = None) -> Tuple[bool, Dict]:
    metrics = {}
    
    det_score = face_dict['det_score']
    metrics['det_score'] = det_score
    if det_score < self.min_det_score:
        return False, metrics
    
    bbox = face_dict['bbox']
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    face_size = min(face_width, face_height)
    metrics['face_size'] = face_size
    if face_size < self.min_face_size:
        return False, metrics

    landmarks = face_dict['landmarks']
    pose = self.compute_pose_angles(landmarks)
    metrics.update(pose)
    
    if abs(pose['yaw']) > self.max_yaw:
        return False, metrics
    if abs(pose['pitch']) > self.max_pitch:
        return False, metrics
    if abs(pose['roll']) > self.max_roll:
        return False, metrics
    
    if self.check_blur and face_image is not None:
        blur_score = self.compute_blur_score(face_image)
        metrics['blur_score'] = blur_score
        if blur_score < self.blur_threshold:
            return False, metrics
    
    return True, metrics
    
class FaceProcessor:
  def __init__(self,
                output_size=224,
                det_size=(640, 640),
                det_thresh=0.5,
                quality_filter_config: Optional[Dict] = None):
    self.detector = FaceDetector(det_size=det_size, det_thresh=det_thresh)
    self.aligner = FaceAligner(output_size=output_size)
    
    if quality_filter_config is None:
        quality_filter_config = {}
    self.quality_filter = FaceQualityFilter(**quality_filter_config)
    
  def process_image(self, 
                    image_path: str,
                    return_all: bool = False) -> List[Dict]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return self.process_numpy(image_rgb, return_all)
  
  def process_numpy(self,
                    image_rgb: np.ndarray,
                    return_all: bool = False) -> List[Dict]:
    faces = self.detector.detect(image_rgb)
    
    if len(faces) == 0:
        return []
    
    results = []
    for face in faces:
      aligned_face = self.aligner.align(image_rgb, face['landmarks'])
      
      is_valid, quality_metrics = self.quality_filter.is_valid(face, aligned_face)
      
      if is_valid or return_all:
          results.append({
              'aligned_face': aligned_face,
              'bbox': face['bbox'],
              'landmarks': face['landmarks'],
              'det_score': face['det_score'],
              'quality_metrics': quality_metrics,
              'is_valid': is_valid
          })
    
    results.sort(
      key=lambda x: x['det_score'] * x['quality_metrics'].get('blur_score', 1000),
      reverse=True
    )
    
    if not return_all and len(results) > 0:
      return [results[0]]
    
    return results
  
def visualize_detections(image_path: str, faces: List[Dict], save_path: str = None):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches

  image = cv2.imread(image_path)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  fig, ax = plt.subplots(1, figsize=(15, 10))
  ax.imshow(image_rgb)
  
  for idx, face in enumerate(faces):
    bbox = face['bbox']
    landmarks = face['landmarks']
    is_valid = face['is_valid']
    
    color = 'green' if is_valid else 'red'
    
    rect = patches.Rectangle(
      (bbox[0], bbox[1]), 
      bbox[2] - bbox[0], 
      bbox[3] - bbox[1],
      linewidth=2, 
      edgecolor=color, 
      facecolor='none'
    )
    ax.add_patch(rect)
    
    for lm_idx, (x, y) in enumerate(landmarks):
      ax.plot(x, y, 'ro', markersize=4)
    
    metrics = face['quality_metrics']
    label = f"Face {idx+1}\n"
    label += f"Score: {face['det_score']:.2f}\n"
    if 'blur_score' in metrics:
      label += f"Blur: {metrics['blur_score']:.0f}\n"
    if 'yaw' in metrics:
      label += f"Yaw: {metrics['yaw']:.1f}°"
    ax.text(
        bbox[0], bbox[1] - 10,
        label,
        color=color,
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
  
  ax.axis('off')
  plt.title(f"{os.path.basename(image_path)} - Detected {len(faces)} faces", 
            fontsize=14, fontweight='bold')
  plt.tight_layout()
  
  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
  
  plt.show()

def process_classroom_images(input_dir: str, output_dir: str, visualize: bool = True):
  aligned_dir = os.path.join(output_dir, 'aligned_faces')
  viz_dir = os.path.join(output_dir, 'visualizations')
  os.makedirs(aligned_dir, exist_ok=True)
  if visualize:
    os.makedirs(viz_dir, exist_ok=True)
  
  processor = FaceProcessor(
    output_size=224,
    det_size=(640, 640),
    det_thresh=0.5,
    quality_filter_config={
      'min_det_score': 0.5,
      'min_face_size': 40,
      'max_yaw': 60,
      'max_pitch': 45,
      'check_blur': True,
      'blur_threshold': 50
    }
  )
  
  valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

  total_images = 0
  total_faces_detected = 0
  total_faces_valid = 0

  for filename in sorted(os.listdir(input_dir)):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_extensions:
      continue
    
    input_path = os.path.join(input_dir, filename)
    total_images += 1
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    try:
      faces = processor.process_image(input_path, return_all=True)
      
      if len(faces) == 0:
          print("No faces detected")
          continue
      
      total_faces_detected += len(faces)
      valid_faces = [f for f in faces if f['is_valid']]
      total_faces_valid += len(valid_faces)
      
      print(f"Detected {len(faces)} faces ({len(valid_faces)} valid)")
      
      for idx, face in enumerate(faces):
        if face['is_valid']:
            output_filename = f"{os.path.splitext(filename)[0]}_face{idx:02d}.jpg"
            output_path = os.path.join(aligned_dir, output_filename)
            
            aligned_bgr = cv2.cvtColor(face['aligned_face'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, aligned_bgr)
            metrics = face['quality_metrics']
            print(f"  Face {idx:2d}: score={face['det_score']:.3f}, "
                  f"blur={metrics.get('blur_score', 0):.0f}, "
                  f"yaw={metrics['yaw']:.1f}°, "
                  f"pitch={metrics['pitch']:.1f}°, "
                  f"roll={metrics['roll']:.1f}° "
                  f"→ {output_filename}")
      if visualize:
        viz_path = os.path.join(viz_dir, f"{os.path.splitext(filename)[0]}_detection.jpg")
        visualize_detections(input_path, faces, save_path=viz_path)
        
    except Exception as e:
      print(f"Error processing {filename}: {e}")
      import traceback
      traceback.print_exc()
  
  print(f"\n{'='*60}")
  print(f"SUMMARY")
  print(f"{'='*60}")
  print(f"Images processed: {total_images}")
  print(f"Total faces detected: {total_faces_detected}")
  print(f"Valid faces (saved): {total_faces_valid}")
  print(f"Average faces per image: {total_faces_detected/max(total_images,1):.1f}")
  print(f"Valid face ratio: {total_faces_valid/max(total_faces_detected,1)*100:.1f}%")
  print(f"\nAligned faces saved to: {aligned_dir}")
  if visualize:
    print(f"Visualizations saved to: {viz_dir}")

if __name__ == '__main__':
  input_directory = 'samples/classroom'
  output_directory = 'output/classroom_detection'
  
  process_classroom_images(
    input_dir=input_directory,
    output_dir=output_directory,
    visualize=True
  )