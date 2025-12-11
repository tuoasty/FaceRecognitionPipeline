import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import argparse

from face_embedder import FaceEmbedder
from face_recognition import FaceProcessor

def get_project_root():
  current = Path(__file__).resolve()
  return current.parent.parent.parent

PROJECT_ROOT = get_project_root()

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

class EmbeddingGenerator:
  def __init__(self, model_type='adaface', architecture='ir_101',dataset_root=None,
             output_root=None):
    self.model_type = model_type
    self.architecture = architecture
    self.model_name = f"{model_type}_{architecture}"

    if dataset_root is None:
      self.dataset_root = PROJECT_ROOT / 'dataset'
    else:
      self.dataset_root = Path(dataset_root)
    
    if output_root is None:
      self.output_root = PROJECT_ROOT / 'output' / 'v0'
    else:
      self.output_root = Path(output_root)
    
    print(f"\n{'='*60}")
    print(f"Initializing Embedding Generator")
    print(f"Model: {self.model_name}")
    print(f"{'='*60}\n")
    
    self.embedder = FaceEmbedder(
      architecture=architecture,
      model_type=model_type
    )
    self.face_processor = FaceProcessor(
      output_size=112,  # Match model input size
      det_size=(640, 640),
      det_thresh=0.5,
      quality_filter_config={
        'min_det_score': 0.5,
        'min_face_size': 40,
      },
      providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    self.output_dir = self.output_root / 'embeddings' / self.model_name
    self.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Embeddings will be saved to: {self.output_dir}\n")

    
  def extract_name_from_filename(self, filename: str) -> str:
    name = Path(filename).stem
    parts = name.split('_')
    name_parts = []
    for part in parts:
      if part.isdigit():
        break
      name_parts.append(part)
    
    return '_'.join(name_parts) if name_parts else parts[0]
  
  def save_embeddings_json(self, data: Dict, output_path: Path):
    def convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
      elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
      elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
      elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
      else:
        return obj
    
    json_data = convert_to_serializable(data)
    
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
      json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_path}")
  
  def load_image(self, image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
      raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  def process_gallery_enrollment(self, enrollment_type: str = 'one-shot', 
                               use_augmentation: bool = False) -> Dict:
    print(f"\n{'='*60}")
    suffix = 'augmented' if use_augmentation else 'base'
    print(f"Processing Gallery: {enrollment_type}-{suffix}")
    print(f"{'='*60}\n")
    
    gallery_dir = self.dataset_root / 'enrollment' / enrollment_type
    
    if not gallery_dir.exists():
      print(f"Warning: Gallery directory not found: {gallery_dir}")
      return {}
    
    gallery_embeddings = {}
    person_dirs = sorted([d for d in gallery_dir.iterdir() if d.is_dir()])
    
    for person_dir in tqdm(person_dirs, desc="Processing persons"):
      person_name = person_dir.name

      image_files = sorted(list(person_dir.glob('*.jpg')) + 
                          list(person_dir.glob('*.png')) +
                          list(person_dir.glob('*.jpeg')))
      
      if len(image_files) == 0:
        print(f"Warning: No images found for {person_name}")
        continue
      
      all_face_images = []
      valid_files = []
      
      for img_path in image_files:
        try:
          # Detect and align face
          faces = self.face_processor.process_image(str(img_path), return_all=True)
          
          if len(faces) == 0:
            print(f"  No face detected in {img_path.name}")
            continue
          
          aligned_face = faces[0]['aligned_face']
          
          if use_augmentation:
            # Generate augmentations
            augmented = augment_face_for_enrollment(aligned_face, num_augmentations=8)
            all_face_images.extend(augmented)
          else:
            # Just use the aligned face
            all_face_images.append(aligned_face)
          
          valid_files.append(img_path.name)
          
        except Exception as e:
          print(f"Error processing {img_path}: {e}")
          continue
      
      if len(all_face_images) > 0:
        if self.model_type == 'arcface':
            embeddings = self.embedder.extract_embeddings_batch(all_face_images, normalize=True, batch_size=1)
        else:
            embeddings = self.embedder.extract_embeddings_batch(all_face_images, normalize=True, batch_size=32)
        
        gallery_embeddings[person_name] = {
          'embeddings': embeddings,
          'num_images': len(valid_files),
          'num_embeddings': len(embeddings),
          'image_files': valid_files,
          'enrollment_type': enrollment_type,
          'augmented': use_augmentation
        }
        
        print(f"  {person_name}: {len(embeddings)} embeddings from {len(valid_files)} images")
    
    print(f"\nTotal persons enrolled: {len(gallery_embeddings)}")
    
    output_path = self.output_dir / f'gallery_{enrollment_type}_{suffix}.pkl'
    with open(output_path, 'wb') as f:
      pickle.dump(gallery_embeddings, f)
    self.save_embeddings_json(gallery_embeddings, output_path)
    
    print(f"Saved to: {output_path}")
    
    return gallery_embeddings

  def process_probe_positive(self, segmented: bool = False) -> Dict:
    print(f"\n{'='*60}")
    print(f"Processing Probe Positive: {'Segmented' if segmented else 'Unsegmented'}")
    print(f"{'='*60}\n")
    
    if segmented:
      probe_dir = self.output_root / 'probe_labeled' / 'segmented'
      categories = ['high_quality', 'blur_blurry', 'blur_sharp', 'face_large', 
                    'face_medium', 'face_small', 'pose_easy', 'pose_medium', 'pose_hard', 'low_quality']
    else:
      probe_dir = self.output_root / 'probe_labeled' / 'positive'
      categories = ['.']
    
    if not probe_dir.exists():
      print(f"Warning: Probe directory not found: {probe_dir}")
      return {}
    
    probe_embeddings = {}
    
    for category in categories:
      if category == '.':
        category_dir = probe_dir
        category_name = 'all'
      else:
        category_dir = probe_dir / category
        category_name = category
      
      if not category_dir.exists():
        print(f"Warning: Category directory not found: {category_dir}")
        continue
      
      print(f"\nProcessing category: {category_name}")
      
      image_files = sorted(list(category_dir.glob('*.jpg')) + 
                          list(category_dir.glob('*.png')) +
                          list(category_dir.glob('*.jpeg')))
      
      if len(image_files) == 0:
        print(f"  No images found")
        continue
      
      category_data = {}
      
      for img_path in tqdm(image_files, desc=f"  {category_name}"):
        try:
          person_name = self.extract_name_from_filename(img_path.name)
          
          # Load image directly (already aligned from DatasetPreprocessor)
          img = self.load_image(img_path)
          
          # Resize to 112x112 if needed (DatasetPreprocessor uses 224x224)
          if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))
          
          embedding = self.embedder.extract_embedding(img, normalize=True)

          if person_name not in category_data:
            category_data[person_name] = {
              'embeddings': [],
              'filenames': []
            }
          
          category_data[person_name]['embeddings'].append(embedding)
          category_data[person_name]['filenames'].append(img_path.name)
            
        except Exception as e:
          print(f"Error processing {img_path.name}: {e}")
          continue

      for person_name in category_data:
        category_data[person_name]['embeddings'] = np.array(
          category_data[person_name]['embeddings']
        )
      
      probe_embeddings[category_name] = category_data
      print(f"  Processed {len(category_data)} persons, "
          f"{sum(len(d['embeddings']) for d in category_data.values())} total images")

    suffix = 'segmented' if segmented else 'unsegmented'
    output_path = self.output_dir / f'probe_positive_{suffix}.pkl'
    with open(output_path, 'wb') as f:
      pickle.dump(probe_embeddings, f)
    self.save_embeddings_json(probe_embeddings, output_path)
    
    print(f"\nSaved to: {output_path}")
    
    return probe_embeddings
  
  def process_probe_negative(self) -> Dict:
    print(f"\n{'='*60}")
    print(f"Processing Probe Negative")
    print(f"{'='*60}\n")
    
    probe_dir = self.output_root / 'probe_labeled' / 'negative'
    
    if not probe_dir.exists():
      print(f"Warning: Probe directory not found: {probe_dir}")
      return {}
    
    negative_embeddings = {
      'real': {'embeddings': [], 'filenames': []},
      'lfw': {'embeddings': [], 'filenames': []}
    }
    
    image_files = sorted(list(probe_dir.glob('*.jpg')) + 
                        list(probe_dir.glob('*.png')) +
                        list(probe_dir.glob('*.jpeg')))
    
    print(f"Found {len(image_files)} negative probe images")
    
    for img_path in tqdm(image_files, desc="Processing negatives"):
      try:
        # Load image directly (already aligned from DatasetPreprocessor)
        img = self.load_image(img_path)
        
        # Resize to 112x112 if needed (DatasetPreprocessor uses 224x224)
        if img.shape[0] != 112 or img.shape[1] != 112:
          img = cv2.resize(img, (112, 112))
        
        embedding = self.embedder.extract_embedding(img, normalize=True)
        
        if 'lfw' in img_path.name.lower() or 'lfw' in str(img_path.parent).lower():
          category = 'lfw'
        else:
          category = 'real'
        
        negative_embeddings[category]['embeddings'].append(embedding)
        negative_embeddings[category]['filenames'].append(img_path.name)
          
      except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        continue

    for category in negative_embeddings:
      if len(negative_embeddings[category]['embeddings']) > 0:
        negative_embeddings[category]['embeddings'] = np.array(
          negative_embeddings[category]['embeddings']
        )
        print(f"  {category}: {len(negative_embeddings[category]['embeddings'])} images")

    output_path = self.output_dir / 'probe_negative.pkl'
    with open(output_path, 'wb') as f:
      pickle.dump(negative_embeddings, f)
    self.save_embeddings_json(negative_embeddings, output_path)
    
    print(f"\nSaved to: {output_path}")
    
    return negative_embeddings

  def generate_all_embeddings(self):
    print(f"\n{'#'*60}")
    print(f"# GENERATING ALL EMBEDDINGS - {self.model_name}")
    print(f"{'#'*60}\n")
    
    start_time = datetime.now()
    print("\n[1/7] Gallery One-Shot BASE")
    gallery_oneshot_base = self.process_gallery_enrollment('one-shot', use_augmentation=False)
    
    print("\n[2/7] Gallery One-Shot AUGMENTED")
    gallery_oneshot_aug = self.process_gallery_enrollment('one-shot', use_augmentation=True)
    
    print("\n[3/7] Gallery Few-Shot BASE")
    gallery_fewshot_base = self.process_gallery_enrollment('few-shot', use_augmentation=False)
    
    print("\n[4/7] Gallery Few-Shot AUGMENTED")
    gallery_fewshot_aug = self.process_gallery_enrollment('few-shot', use_augmentation=True)
    
    print("\n[5/7] Probe Positive (Unsegmented)")
    probe_positive_unseg = self.process_probe_positive(segmented=False)
    
    print("\n[6/7] Probe Positive (Segmented)")
    probe_positive_seg = self.process_probe_positive(segmented=True)
    
    print("\n[7/7] Probe Negative")
    probe_negative = self.process_probe_negative()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    summary = {
      'model_type': self.model_type,
      'architecture': self.architecture,
      'model_name': self.model_name,
      'timestamp': datetime.now().isoformat(),
      'duration_seconds': duration,
      'gallery': {
        'one_shot_base_persons': len(gallery_oneshot_base),
        'one_shot_augmented_persons': len(gallery_oneshot_aug),
        'few_shot_base_persons': len(gallery_fewshot_base),
        'few_shot_augmented_persons': len(gallery_fewshot_aug)
      },
      'probe_positive': {
        'unsegmented_categories': list(probe_positive_unseg.keys()) if probe_positive_unseg else [],
        'segmented_categories': list(probe_positive_seg.keys()) if probe_positive_seg else []
      },
      'probe_negative': {
        'real_images': len(probe_negative.get('real', {}).get('embeddings', [])),
        'lfw_images': len(probe_negative.get('lfw', {}).get('embeddings', []))
      },
      'output_directory': str(self.output_dir)
    }
    summary_path = self.output_dir / 'generation_summary.json'
    with open(summary_path, 'w') as f:
      json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"EMBEDDING GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {self.model_name}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nGallery:")
    print(f"  One-shot base: {summary['gallery']['one_shot_base_persons']} persons")
    print(f"  One-shot augmented: {summary['gallery']['one_shot_augmented_persons']} persons")
    print(f"  Few-shot base: {summary['gallery']['few_shot_base_persons']} persons")
    print(f"  Few-shot augmented: {summary['gallery']['few_shot_augmented_persons']} persons")
    print(f"\nProbe Positive:")
    print(f"  Unsegmented categories: {len(summary['probe_positive']['unsegmented_categories'])}")
    print(f"  Segmented categories: {len(summary['probe_positive']['segmented_categories'])}")
    print(f"\nProbe Negative:")
    print(f"  Real images: {summary['probe_negative']['real_images']}")
    print(f"  LFW images: {summary['probe_negative']['lfw_images']}")
    print(f"\nOutput directory: {self.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

def main():
  parser = argparse.ArgumentParser(
    description='Generate face embeddings for evaluation using multiple models'
  )
  parser.add_argument(
    '--model_type',
    type=str,
    default='all',
    choices=['adaface', 'arcface', 'all'],
    help='Model type to use (default: all models)'
  )
  parser.add_argument(
    '--architecture',
    type=str,
    default='all',
    choices=['ir_50', 'ir_101', 'all'],
    help='Model architecture (default: all architectures)'
  )
  parser.add_argument(
    '--dataset_root',
    type=str,
    default=None,
    help='Root directory for dataset (default: PROJECT_ROOT/dataset)'
  )
  parser.add_argument(
    '--output_root',
    type=str,
    default=None,
    help='Root directory for outputs (default: PROJECT_ROOT/output/v0)'
  )
    
  args = parser.parse_args()

  model_types = ['adaface', 'arcface'] if args.model_type == 'all' else [args.model_type]
  architectures = ['ir_50', 'ir_101'] if args.architecture == 'all' else [args.architecture]

  total_runs = len(model_types) * len(architectures)
  current_run = 0
  
  print(f"\n{'#'*60}")
  print(f"# EMBEDDING GENERATION PIPELINE")
  print(f"# Total models to process: {total_runs}")
  print(f"{'#'*60}\n")
  
  for model_type in model_types:
    for architecture in architectures:
      current_run += 1
      print(f"\n{'#'*60}")
      print(f"# RUN {current_run}/{total_runs}")
      print(f"{'#'*60}\n")
      
      try:
        generator = EmbeddingGenerator(
          model_type=model_type,
          architecture=architecture,
          dataset_root=args.dataset_root,
          output_root=args.output_root
        )
        generator.generate_all_embeddings()
      except Exception as e:
        print(f"\nERROR in {model_type}_{architecture}: {e}")
        import traceback
        traceback.print_exc()
        continue
  
  print(f"\n{'#'*60}")
  print(f"# ALL EMBEDDING GENERATION COMPLETE")
  print(f"# Processed {total_runs} model configurations")
  print(f"{'#'*60}\n")


if __name__ == '__main__':
  main()