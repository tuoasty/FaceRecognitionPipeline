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

def get_project_root():
  current = Path(__file__).resolve()
  return current.parent.parent.parent

PROJECT_ROOT = get_project_root()

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
  
  def load_image(self, image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
      raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  def process_gallery_enrollment(self, enrollment_type: str = 'one-shot') -> Dict:
    print(f"\n{'='*60}")
    print(f"Processing Gallery: {enrollment_type}")
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
      
      embeddings = []
      valid_files = []
      
      for img_path in image_files:
        try:
          img = self.load_image(img_path)
          embedding = self.embedder.extract_embedding(img, normalize=True)
          embeddings.append(embedding)
          valid_files.append(img_path.name)
        except Exception as e:
          print(f"Error processing {img_path}: {e}")
          continue
      
      if len(embeddings) > 0:
        embeddings_array = np.array(embeddings)
        gallery_embeddings[person_name] = {
          'embeddings': embeddings_array,
          'num_images': len(embeddings),
          'image_files': valid_files,
          'enrollment_type': enrollment_type
        }
        
        print(f"  {person_name}: {len(embeddings)} embeddings")
    
    print(f"\nTotal persons enrolled: {len(gallery_embeddings)}")
    
    output_path = self.output_dir / f'gallery_{enrollment_type}.pkl'
    with open(output_path, 'wb') as f:
      pickle.dump(gallery_embeddings, f)
    
    print(f"Saved to: {output_path}")
    
    return gallery_embeddings

  def process_probe_positive(self, segmented: bool = False) -> Dict:
    print(f"\n{'='*60}")
    print(f"Processing Probe Positive: {'Segmented' if segmented else 'Unsegmented'}")
    print(f"{'='*60}\n")
    
    if segmented:
      probe_dir = self.output_root / 'probe_labeled' / 'probe_segmented'
      categories = ['baseline', 'left', 'center', 'right', 
                    'high_pitch', 'high_yaw', 'blur', 'low_quality']
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
          
          img = self.load_image(img_path)
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
        img = self.load_image(img_path)
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
    
    print(f"\nSaved to: {output_path}")
    
    return negative_embeddings

  def generate_all_embeddings(self):
    print(f"\n{'#'*60}")
    print(f"# GENERATING ALL EMBEDDINGS - {self.model_name}")
    print(f"{'#'*60}\n")
    
    start_time = datetime.now()
    print("\n[1/5] Gallery One-Shot")
    gallery_oneshot = self.process_gallery_enrollment('one-shot')
    
    print("\n[2/5] Gallery Few-Shot")
    gallery_fewshot = self.process_gallery_enrollment('few-shot')
    print("\n[3/5] Probe Positive (Unsegmented)")
    probe_positive_unseg = self.process_probe_positive(segmented=False)
    
    print("\n[4/5] Probe Positive (Segmented)")
    probe_positive_seg = self.process_probe_positive(segmented=True)
    
    print("\n[5/5] Probe Negative")
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
        'one_shot_persons': len(gallery_oneshot),
        'few_shot_persons': len(gallery_fewshot)
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
    print(f"  One-shot: {summary['gallery']['one_shot_persons']} persons")
    print(f"  Few-shot: {summary['gallery']['few_shot_persons']} persons")
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