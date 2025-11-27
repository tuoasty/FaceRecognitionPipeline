import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

from face_recognition import FaceProcessor
from face_embedder import FaceEmbedder
from gallery_manager import GalleryManager

def augment_face_for_enrollment(face_image: np.ndarray, 
                                num_augmentations: int = 8) -> List[np.ndarray]:
  augmented = [face_image.copy()]
  
  augmented.append(cv2.flip(face_image, 1))
  
  for angle in [-10, -5, 5, 10]:
    center = (face_image.shape[1] // 2, face_image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(face_image, M, 
                              (face_image.shape[1], face_image.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)
    augmented.append(rotated)
  for beta in [-20, -10, 10, 20]:
    adjusted = np.clip(face_image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    augmented.append(adjusted)

  for alpha in [0.85, 0.92, 1.08, 1.15]:
    adjusted = np.clip(face_image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    augmented.append(adjusted)
  
  blurred = cv2.GaussianBlur(face_image, (3, 3), 0.5)
  augmented.append(blurred)
  
  noise = np.random.normal(0, 3, face_image.shape).astype(np.float32)
  noisy = np.clip(face_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
  augmented.append(noisy)
  
  return augmented[:num_augmentations]

class StudentEnrollment:
  def __init__(self,
                gallery_path='gallery/students.pkl',
                min_faces_per_student=3,
                max_faces_per_student=10,
                limit_images=0,
                image_indices=None):
    self.min_faces = min_faces_per_student
    self.max_faces = max_faces_per_student
    self.limit_images = limit_images
    self.image_indices = image_indices
    
    print("Initializing enrollment system...")
    print("="*60)
    print("\n1. Loading face detection & alignment...")
    self.face_processor = FaceProcessor(
      output_size=224,
      det_size=(640, 640),
      det_thresh=0.5,
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

    print("\n2. Loading AdaFace model...")
    self.embedder = FaceEmbedder(
      architecture='ir_101'
    )

    print("\n3. Loading gallery database...")
    self.gallery = GalleryManager(
        gallery_path=gallery_path,
        aggregation_method='mean'
    )
    
    print("\n" + "="*60)
    print("Enrollment system ready!")
    print("="*60 + "\n")
    
  def process_student_directory(self, 
                                student_dir: str,
                                student_id: str = None) -> Tuple[bool, Dict]:
      student_name = os.path.basename(student_dir)
      if student_id is None:
        existing_count = len(self.gallery.get_all_students())
        student_id = f"STU{existing_count + 1:04d}"
      
      print(f"\n{'='*60}")
      print(f"Processing: {student_name}")
      print(f"Student ID: {student_id}")
      print(f"{'='*60}")
      
      image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
      image_files = []
      
      for file in sorted(os.listdir(student_dir)):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
          image_files.append(os.path.join(student_dir, file))

      image_files = sorted(image_files)
      
      if len(image_files) == 0:
        print(f"No images found in {student_dir}")
        return False, {'error': 'no_images'}
      
      if self.image_indices:
        selected = []
        for idx in self.image_indices:
          if 1 <= idx <= len(image_files):
            selected.append(image_files[idx - 1])
          else:
            print(f"Warning: image index {idx} out of range (1-{len(image_files)})")

        image_files = selected
        print(f"Using explicit image indices: {self.image_indices}")

      elif self.limit_images > 0:
        image_files = image_files[:self.limit_images]
        print(f"Limiting to first {self.limit_images} images")
      
      print(f"Found {len(image_files)} images")
      all_faces = []
      valid_faces = []
      
      for idx, img_path in enumerate(image_files, 1):
        print(f"\n  [{idx}/{len(image_files)}] Processing {os.path.basename(img_path)}...", end=' ')
        
        try:
          faces = self.face_processor.process_image(img_path, return_all=True)
          
          if len(faces) == 0:
              print("No faces detected")
              continue
          
          best_face = faces[0]
          all_faces.append(best_face)
          
          if best_face['is_valid']:
            valid_faces.append(best_face)
            quality = best_face['quality_metrics']
            print(f"✓ Valid face (score: {best_face['det_score']:.3f}, "
                f"blur: {quality.get('blur_score', 0):.0f})")
          else:
            print("Low quality face")
            
        except Exception as e:
          print(f"Error: {e}")
          continue
      
      print(f"\n  Summary: {len(valid_faces)}/{len(all_faces)} valid faces")
      if len(valid_faces) < self.min_faces:
          print(f"\nInsufficient valid faces ({len(valid_faces)} < {self.min_faces})")
          print(f"   Please provide more high-quality images for {student_name}")
          return False, {
            'error': 'insufficient_faces',
            'valid_faces': len(valid_faces),
            'required': self.min_faces
          }
    
      if len(valid_faces) > self.max_faces:
        print(f"  Using top {self.max_faces} faces (sorted by quality)")
        valid_faces.sort(
          key=lambda x: x['det_score'] * x['quality_metrics'].get('blur_score', 1000),
          reverse=True
        )
        valid_faces = valid_faces[:self.max_faces]
      
      print(f"\n  Applying augmentation to {len(valid_faces)} faces...")
      all_face_images = []
      augmentation_per_face = 8

      for face in valid_faces:
        original_face = face['aligned_face']
        augmented_faces = augment_face_for_enrollment(original_face, 
                                                        num_augmentations=augmentation_per_face)
        all_face_images.extend(augmented_faces)

      print(f"  Generated {len(all_face_images)} augmented faces "
            f"({augmentation_per_face} per original)")

      print(f"  Extracting embeddings...")
      embeddings = self.embedder.extract_embeddings_batch(all_face_images, normalize=True)

      print(f"  Extracted {len(embeddings)} embeddings (512-dim)")

      similarities = np.dot(embeddings, embeddings.T)
      avg_similarity = (np.sum(similarities) - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))
      print(f"  Average intra-class similarity: {avg_similarity:.4f}")
      
      if avg_similarity < 0.3:
        print(f"\nWarning: Low intra-class similarity ({avg_similarity:.4f})")
        print(f"   Images may contain different people or very different poses")

      success = self.gallery.add_student(
        student_id=student_id,
        name=student_name,
        embeddings=embeddings,
        metadata={
          'num_images': len(image_files),
          'num_valid_faces': len(valid_faces),
          'num_augmented_faces': len(all_face_images),
          'augmentation_per_face': augmentation_per_face,
          'avg_similarity': float(avg_similarity),
          'source_directory': student_dir
        },
        overwrite=True
      )
      
      if success:
        print(f"\nSuccessfully enrolled {student_name} ({student_id})")
      
      return success, {
        'student_id': student_id,
        'name': student_name,
        'num_images': len(image_files),
        'num_valid_faces': len(valid_faces),
        'num_embeddings': len(embeddings),
        'avg_similarity': float(avg_similarity)
      }
  
  def enroll_from_directory(self, enrollment_dir: str) -> Dict:
      if not os.path.exists(enrollment_dir):
        raise ValueError(f"Enrollment directory not found: {enrollment_dir}")
      
      print("\n" + "="*60)
      print(f"ENROLLMENT FROM: {enrollment_dir}")
      print("="*60)
      
      student_dirs = []
      for item in sorted(os.listdir(enrollment_dir)):
        item_path = os.path.join(enrollment_dir, item)
        if os.path.isdir(item_path):
          student_dirs.append(item_path)
      
      if len(student_dirs) == 0:
        print("No student directories found!")
        return {'error': 'no_directories'}
      
      print(f"Found {len(student_dirs)} student directories")
      
      results = []
      successful = 0
      failed = 0
      
      for student_dir in student_dirs:
        success, info = self.process_student_directory(student_dir)
        
        if success:
          successful += 1
        else:
          failed += 1
        
        results.append({
          'directory': student_dir,
          'success': success,
          'info': info
        })
    
      print("\n" + "="*60)
      print("Saving gallery...")
      self.gallery.save()
      
      print("\n" + "="*60)
      print("ENROLLMENT SUMMARY")
      print("="*60)
      print(f"Total students processed: {len(student_dirs)}")
      print(f"Successfully enrolled: {successful}")
      print(f"Failed: {failed}")
      
      if successful > 0:
        print("\nEnrolled students:")
        for result in results:
          if result['success']:
            info = result['info']
            print(f"  • {info['name']} ({info['student_id']}): "
                f"{info['num_embeddings']} embeddings "
                f"(from {info.get('num_valid_faces', '?')} faces) " 
                f"(similarity: {info['avg_similarity']:.3f})")

      if failed > 0:
        print("\nFailed enrollments:")
        for result in results:
          if not result['success']:
            name = os.path.basename(result['directory'])
            error = result['info'].get('error', 'unknown')
            print(f"  • {name}: {error}")

      stats = self.gallery.get_statistics()
      print(f"\nGallery statistics:")
      print(f"  Total students: {stats['num_students']}")
      print(f"  Total embeddings: {stats['total_embeddings']}")
      print(f"  Avg per student: {stats['avg_embeddings_per_student']:.1f}")
      
      print("\n" + "="*60)

      if successful > 0:
        self.verify_enrollment()
      
      return {
        'total': len(student_dirs),
        'successful': successful,
        'failed': failed,
        'results': results,
        'gallery_stats': stats
      }
  
  def verify_enrollment(self):
    print("\n" + "="*60)
    print("ENROLLMENT VERIFICATION")
    print("="*60)
    
    students = self.gallery.get_all_students()
    if len(students) < 2:
      print("Need at least 2 students for verification")
      return
    
    print(f"\nTesting {len(students)} enrolled students...")
    correct_matches = 0
    total_tests = 0
    inter_class_sims = []
    
    for student_id, student in students.items():
      query_embedding = student.embeddings[0]
      results = self.gallery.search(query_embedding, top_k=3)
      top_match_id = results[0][0]
      top_match_name = results[0][1]
      top_match_score = results[0][2]
      
      if top_match_name == student.name:
          correct_matches += 1
      else:
          print(f"  ✗ Invalid {student.name}: Matched to {top_match_name} (score: {top_match_score:.3f})")
      
      total_tests += 1
      
      for match_id, match_name, score in results[1:]:
        inter_class_sims.append(score)
    
    accuracy = correct_matches / total_tests * 100
    avg_inter_class = np.mean(inter_class_sims) if inter_class_sims else 0
    max_inter_class = np.max(inter_class_sims) if inter_class_sims else 0
    
    print(f"\nVerification Results:")
    print(f"  Rank-1 Accuracy: {correct_matches}/{total_tests} ({accuracy:.1f}%)")
    print(f"  Avg inter-class similarity: {avg_inter_class:.3f}")
    print(f"  Max inter-class similarity: {max_inter_class:.3f}")
    
    if accuracy < 100:
      print(f"\nWarning: {total_tests - correct_matches} student(s) failed verification!")
      print("   Check if images contain the correct person")
    
    if max_inter_class > 0.6:
      print(f"\nWarning: High inter-class similarity detected ({max_inter_class:.3f})")
      print("   Some students may look very similar or be duplicates")
    
    if accuracy == 100 and max_inter_class < 0.5:
      print("\nAll students verified successfully!")
    
    print("="*60)


def main():
  parser = argparse.ArgumentParser(
    description='Enroll students from directory structure'
  )
  parser.add_argument(
    '--enrollment_dir',
    type=str,
    default='samples/enrollment',
    help='Directory containing student subdirectories'
  )
  parser.add_argument(
    '--gallery_path',
    type=str,
    default='gallery/students.pkl',
    help='Path to gallery database'
  )
  parser.add_argument(
    '--min_faces',
    type=int,
    default=1,
    help='Minimum valid faces required per student'
  )
  parser.add_argument(
    '--max_faces',
    type=int,
    default=10,
    help='Maximum faces to use per student'
  )
  parser.add_argument(
    '--limit_images',
    type=int,
    default=0,
    help='Limit how many images per student to process (0 = all)'
  ) 

  parser.add_argument(
    '--image_indices',
    type=int,
    nargs='*',
    default=None,
    help='Explicit list of image numbers to use (1-based). Example: 2 3 4'
  )
    
  args = parser.parse_args()
  
  enrollment = StudentEnrollment(
    gallery_path=args.gallery_path,
    min_faces_per_student=args.min_faces,
    max_faces_per_student=args.max_faces,
    limit_images=args.limit_images,
    image_indices=args.image_indices
  )
  
  summary = enrollment.enroll_from_directory(args.enrollment_dir)

  if summary.get('successful', 0) > 0:
    print("\nCreating backup...")
    backup_dir = os.path.join(os.path.dirname(args.gallery_path), 'backups')
    enrollment.gallery.export_for_backup(backup_dir)
    print(f"Backup saved to {backup_dir}")


if __name__ == '__main__':
  main()