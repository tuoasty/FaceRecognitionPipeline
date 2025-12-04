import cv2
import numpy as np
import os
import json
from typing import Dict, List
from pathlib import Path
import re

from face_recognition import FaceProcessor

class DatasetPreprocessor:
  def __init__(self, 
                output_size=224,
                det_size=(640, 640),
                det_thresh=0.3,
                quality_filter_config: Dict = None):
    print("Initializing FaceProcessor...")
    if quality_filter_config is None:
      quality_filter_config = {
        'min_det_score': 0.3,
        'min_face_size': 30,
        'max_yaw': 90,
        'max_pitch': 90,
        'max_roll': 90,
        'check_blur': True,
        'blur_threshold': 100
      }
    
    self.processor = FaceProcessor(
      output_size=output_size,
      det_size=det_size,
      det_thresh=det_thresh,
      quality_filter_config=quality_filter_config,
      providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
  
  def standardize_filename(self, original_path: str, class_id: str, angle: str, image_idx: int) -> str:
    return f"{class_id}_{angle}_{image_idx:03d}.jpg"
  
  def process_single_image(self, 
                          image_path: str,
                          class_id: str,
                          angle: str,
                          standardized_name: str,
                          output_dir: str,
                          metadata_list: List[Dict]) -> int:
    original_filename = os.path.basename(image_path)
    
    print(f"  Processing: {original_filename} → {standardized_name}")
    
    try:
      faces = self.processor.process_image(image_path, return_all=True)
      
      if len(faces) == 0:
        print(f"No faces detected")
        return 0
      
      saved_count = 0
      for face_idx, face in enumerate(faces):
        output_filename = f"{standardized_name}_face{face_idx}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        aligned_bgr = cv2.cvtColor(face['aligned_face'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, aligned_bgr)

        metrics = face['quality_metrics']

        metadata_entry = {
          'filename': output_filename,
          'class_id': class_id,
          'source_image': original_filename,
          'standardized_name': f"{standardized_name}.jpg",
          'face_index': face_idx,
          'angle': angle,
          'det_score': float(face['det_score']),
          'yaw': float(metrics.get('yaw', 0.0)),
          'pitch': float(metrics.get('pitch', 0.0)),
          'roll': float(metrics.get('roll', 0.0)),
          'blur_score': float(metrics.get('blur_score', 0.0)),
          'face_size': int(metrics.get('face_size', 0)),
          'bbox': face['bbox'].tolist()
        }
        
        metadata_list.append(metadata_entry)
        
        print(f"Face {face_idx}: det={face['det_score']:.3f}, "
              f"blur={metrics.get('blur_score', 0):.0f}, "
              f"yaw={metrics.get('yaw', 0):.1f}°, "
              f"pitch={metrics.get('pitch', 0):.1f}° → {output_filename}")
        saved_count += 1
    
      return saved_count
      
    except Exception as e:
      print(f"    ✗ Error: {e}")
      import traceback
      traceback.print_exc()
      return 0

  def process_dataset(self, 
                      input_dir: str,
                      output_dir: str,
                      probe_dir_name: str = 'probe_positive',
                      metadata_filename: str = 'probe_positive_metadata.json'):
      probe_dir = os.path.join(output_dir, probe_dir_name)
      os.makedirs(probe_dir, exist_ok=True)
      
      print("\n" + "="*70)
      print("DATASET PREPROCESSING")
      print("="*70)
      print(f"Input directory: {input_dir}")
      print(f"Output directory: {output_dir}")
      print(f"Probe directory: {probe_dir}")
      print("="*70 + "\n")

      all_metadata = []
      
      total_classes = 0
      total_images = 0
      total_faces = 0

      valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

      angle_folders = ['center', 'left', 'right']
      
      class_dirs = sorted([d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d))])
      
      if len(class_dirs) == 0:
        print("No class directories found!")
        return

      for class_id in class_dirs:
        class_path = os.path.join(input_dir, class_id)
        print(f"\n{'='*70}")
        print(f"Processing Class: {class_id}")
        print(f"{'='*70}")
        
        total_classes += 1
        class_image_count = 0
        class_face_count = 0
    
        has_angle_folders = any(
          os.path.isdir(os.path.join(class_path, angle)) 
          for angle in angle_folders
        )
        
        if has_angle_folders:
          for angle in angle_folders:
            angle_path = os.path.join(class_path, angle)
            
            if not os.path.isdir(angle_path):
              print(f"Missing angle folder: {angle}")
              continue
            
            print(f"\n  Angle: {angle}")
            print(f"  {'-'*66}")
        
            image_files = sorted([
              f for f in os.listdir(angle_path)
              if os.path.splitext(f)[1].lower() in valid_extensions
            ])
            
            if len(image_files) == 0:
              print(f"No images found in {angle} folder")
              continue

            for img_idx, image_file in enumerate(image_files, start=1):
              image_path = os.path.join(angle_path, image_file)
              
              standardized_name = self.standardize_filename(
                image_path, class_id, angle, img_idx
              )
              standardized_name = os.path.splitext(standardized_name)[0]
  
              num_faces = self.process_single_image(
                image_path=image_path,
                class_id=class_id,
                angle=angle,
                standardized_name=standardized_name,
                output_dir=probe_dir,
                metadata_list=all_metadata
              )
              
              total_images += 1
              class_image_count += 1
              total_faces += num_faces
              class_face_count += num_faces
      
        else:
          print(f"No angle folders found, processing images directly")
          print(f"  {'-'*66}")
          
          image_files = sorted([
            f for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in valid_extensions
          ])
          
          for img_idx, image_file in enumerate(image_files, start=1):
            image_path = os.path.join(class_path, image_file)
            
            filename_lower = image_file.lower()
            if 'left' in filename_lower:
              angle = 'left'
            elif 'right' in filename_lower:
              angle = 'right'
            else:
              angle = 'center'
            
            standardized_name = self.standardize_filename(
              image_path, class_id, angle, img_idx
            )
            standardized_name = os.path.splitext(standardized_name)[0]
            
            num_faces = self.process_single_image(
              image_path=image_path,
              class_id=class_id,
              angle=angle,
              standardized_name=standardized_name,
              output_dir=probe_dir,
              metadata_list=all_metadata
            )
            total_images += 1
            class_image_count += 1
            total_faces += num_faces
            class_face_count += num_faces
        
        print(f"\n  Class Summary: {class_image_count} images, {class_face_count} faces detected")
      
      metadata_path = os.path.join(output_dir, metadata_filename)
      with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
      
      print("\n" + "="*70)
      print("PREPROCESSING COMPLETE")
      print("="*70)
      print(f"Total classes processed: {total_classes}")
      print(f"Total images processed: {total_images}")
      print(f"Total faces detected: {total_faces}")
      print(f"Average faces per image: {total_faces/max(total_images,1):.2f}")
      print(f"\nProbe images saved to: {probe_dir}")
      print(f"Metadata saved to: {metadata_path}")
      print("="*70 + "\n")

      angle_counts = {}
      for entry in all_metadata:
        angle = entry['angle']
        angle_counts[angle] = angle_counts.get(angle, 0) + 1
      
      if angle_counts:
        print("Angle Distribution:")
        for angle, count in sorted(angle_counts.items()):
          print(f"  {angle}: {count} faces")
        print()

def main():
  import argparse
  
  parser = argparse.ArgumentParser(
    description='Preprocess classroom dataset for face recognition evaluation'
  )
  parser.add_argument(
    '--input_dir',
    type=str,
    default='samples/classroom',
    help='Input directory containing class subdirectories'
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default='output/preprocessed',
    help='Output directory for processed data'
  )
  parser.add_argument(
    '--probe_dir',
    type=str,
    default='probe_positive',
    help='Name of subdirectory for probe images'
  )
  parser.add_argument(
    '--metadata_file',
    type=str,
    default='probe_positive_metadata.json',
    help='Name of metadata JSON file'
  )
  parser.add_argument(
    '--output_size',
    type=int,
    default=224,
    help='Size of aligned face output'
  )
  parser.add_argument(
    '--det_thresh',
    type=float,
    default=0.3,
    help='Detection threshold (lower = more detections)'
  )
  
  args = parser.parse_args()

  preprocessor = DatasetPreprocessor(
    output_size=args.output_size,
    det_thresh=args.det_thresh
  )
  
  preprocessor.process_dataset(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    probe_dir_name=args.probe_dir,
    metadata_filename=args.metadata_file
  )


if __name__ == '__main__':
  main()