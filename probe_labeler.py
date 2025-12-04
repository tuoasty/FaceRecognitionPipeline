import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
from datetime import datetime
import shutil

from face_embedder import FaceEmbedder
from gallery_manager import GalleryManager

def get_script_dir():
  return Path(__file__).resolve().parent

SCRIPT_DIR = get_script_dir()

class ProbeLabeler:
  def __init__(self,
              gallery_path=None,
              model_type='adaface',
              architecture='ir_101',
              sure_threshold=0.5,
              unsure_threshold=0.4):
    self.sure_threshold = sure_threshold
    self.unsure_threshold = unsure_threshold
    self.model_type = model_type
    self.architecture = architecture

    if gallery_path is None:
      gallery_path = str(SCRIPT_DIR / 'gallery' / 'students.pkl')
    
    print("Initializing Probe Labeler...")
    print("="*70)
    print(f"Model: {model_type} - {architecture}")
    print(f"Sure threshold: {sure_threshold}")
    print(f"Unsure threshold: {unsure_threshold}")
    print("="*70)

    self.embedder = FaceEmbedder(architecture=architecture, model_type=model_type)
    self.gallery = GalleryManager(gallery_path=gallery_path)
    
    num_students = len(self.gallery.get_all_students())
    if num_students == 0:
      print("\nWARNING: Gallery is empty! Please enroll students first.")
    else:
      print(f"   Loaded {num_students} enrolled students")
    
    print("\n" + "="*70)
    print("Probe Labeler ready!")
    print("="*70 + "\n")
  
  def determine_label(self, confidence: float) -> str:
    if confidence >= self.sure_threshold:
      return "SURE"
    elif confidence >= self.unsure_threshold:
      return "UNSURE"
    else:
      return "IMPOSTOR"
  
  def match_face(self, face_image: np.ndarray, top_k: int = 3) -> Tuple[str, str, float, str, List]:
    embedding = self.embedder.extract_embedding(face_image, normalize=True)
    results = self.gallery.search(embedding, top_k=top_k)
    
    if len(results) == 0:
      return None, "UNKNOWN", 0.0, "IMPOSTOR", []
    
    student_id, name, confidence = results[0]
    label = self.determine_label(confidence)
    
    top_matches = [
      {'student_id': sid, 'name': n, 'score': float(s), 'rank': i+1}
      for i, (sid, n, s) in enumerate(results)
    ]
    
    return student_id, name, float(confidence), label, top_matches
  
  def process_probe_directory(self,
                            probe_dir: str,
                            output_dir: str = None,
                            metadata_file: str = None,
                            copy_files: bool = True,
                            top_k: int = 3) -> Dict:
    if not os.path.exists(probe_dir):
      raise ValueError(f"Probe directory not found: {probe_dir}")
    
    if output_dir is None:
      output_dir = probe_dir + '_labeled'
    
    print("\n" + "="*70)
    print("LABELING PROBE FACES")
    print("="*70)
    print(f"Input directory: {probe_dir}")
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    
    if copy_files:
      sure_dir = os.path.join(output_dir, 'SURE')
      unsure_dir = os.path.join(output_dir, 'UNSURE')
      impostor_dir = os.path.join(output_dir, 'IMPOSTOR')
      
      os.makedirs(sure_dir, exist_ok=True)
      os.makedirs(unsure_dir, exist_ok=True)
      os.makedirs(impostor_dir, exist_ok=True)
  
    input_metadata = {}
    if metadata_file and os.path.exists(metadata_file):
      print(f"Loading metadata from: {metadata_file}")
      with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
        for entry in metadata_list:
          input_metadata[entry['filename']] = entry
      print(f"Loaded metadata for {len(input_metadata)} images\n")

    image_files = sorted([
      f for f in os.listdir(probe_dir)
      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    
    if len(image_files) == 0:
      print("No image files found in probe directory!")
      return {'error': 'no_images'}
    
    print(f"Found {len(image_files)} images to process\n")

    results = []
    label_counts = {'SURE': 0, 'UNSURE': 0, 'IMPOSTOR': 0}
    
    for idx, image_file in enumerate(image_files, 1):
      image_path = os.path.join(probe_dir, image_file)
      
      print(f"[{idx}/{len(image_files)}] Processing: {image_file}")
      
      face_image = cv2.imread(image_path)
      if face_image is None:
        print(f"Failed to read image")
        continue
      
      face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
      
      student_id, name, confidence, label, top_matches = self.match_face(
        face_rgb, top_k=top_k
      )
      
      label_counts[label] += 1

      marker = "✓" if label == "SURE" else "⚠" if label == "UNSURE" else "✗"
      print(f"  {marker} {label}: {name} ({student_id}) - confidence: {confidence:.3f}")

      if len(top_matches) > 0:
        print(f"     Top {len(top_matches)} matches:")
        for match in top_matches:
          print(f"       {match['rank']}. {match['name']} ({match['student_id']}): {match['score']:.3f}")

      original_meta = input_metadata.get(image_file, {})

      result = {
        'filename': image_file,
        'matched_student_id': student_id,
        'matched_name': name,
        'confidence': confidence,
        'label': label,
        'top_matches': top_matches,
        'original_metadata': original_meta
      }
      
      results.append(result)

      if copy_files:
        new_filename = f"{name}_{image_file}"
        
        if label == "SURE":
          dest_path = os.path.join(sure_dir, new_filename)
        elif label == "UNSURE":
          dest_path = os.path.join(unsure_dir, new_filename)
        else:
          dest_path = os.path.join(impostor_dir, new_filename)
        
        shutil.copy2(image_path, dest_path)
        result['labeled_path'] = dest_path

    summary = {
      'total_images': len(image_files),
      'processed': len(results),
      'label_distribution': label_counts,
      'sure_percentage': label_counts['SURE'] / len(results) * 100 if results else 0,
      'unsure_percentage': label_counts['UNSURE'] / len(results) * 100 if results else 0,
      'impostor_percentage': label_counts['IMPOSTOR'] / len(results) * 100 if results else 0,
      'settings': {
        'model_type': self.model_type,
        'architecture': self.architecture,
        'sure_threshold': self.sure_threshold,
        'unsure_threshold': self.unsure_threshold
      },
      'timestamp': datetime.now().isoformat()
    }

    results_file = os.path.join(output_dir, 'labeling_results.json')
    with open(results_file, 'w') as f:
      json.dump({
        'summary': summary,
        'results': results
      }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

    self._print_summary(summary, output_dir if copy_files else None)
    
    return summary
  
  def _print_summary(self, summary: Dict, output_dir: str = None):
    print("\n" + "="*70)
    print("LABELING SUMMARY")
    print("="*70)
    print(f"Total images processed: {summary['processed']}/{summary['total_images']}")
    print(f"\nLabel Distribution:")
    print(f"  SURE:     {summary['label_distribution']['SURE']:3d} ({summary['sure_percentage']:.1f}%)")
    print(f"  UNSURE:   {summary['label_distribution']['UNSURE']:3d} ({summary['unsure_percentage']:.1f}%)")
    print(f"  IMPOSTOR: {summary['label_distribution']['IMPOSTOR']:3d} ({summary['impostor_percentage']:.1f}%)")
    
    print(f"\nThresholds:")
    print(f"  Sure threshold:   ≥ {summary['settings']['sure_threshold']}")
    print(f"  Unsure threshold: ≥ {summary['settings']['unsure_threshold']}")
    
    if output_dir:
      print(f"\nLabeled images saved to:")
      print(f"  {os.path.join(output_dir, 'SURE')}")
      print(f"  {os.path.join(output_dir, 'UNSURE')}")
      print(f"  {os.path.join(output_dir, 'IMPOSTOR')}")
    
    print("="*70 + "\n")


def main():
  parser = argparse.ArgumentParser(
      description='Label probe faces by matching against gallery'
  )
  parser.add_argument(
    '--probe_dir',
    type=str,
    default=str(SCRIPT_DIR / 'output' / 'preprocessed' / 'probe_positive'),
    help='Directory containing probe face images'
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default=None,
    help='Output directory (default: probe_dir + "_labeled")'
  )
  parser.add_argument(
    '--metadata_file',
    type=str,
    default=None,
    help='Path to metadata JSON file from preprocessing'
  )
  parser.add_argument(
    '--gallery_path',
    type=str,
    default=str(SCRIPT_DIR / 'gallery' / 'students.pkl'),
    help='Path to student gallery database'
  )
  parser.add_argument(
    '--model_type',
    type=str,
    default='adaface',
    choices=['adaface', 'arcface'],
    help='Type of face recognition model to use'
  )
  parser.add_argument(
    '--architecture',
    type=str,
    default='ir_101',
    choices=['ir_50', 'ir_101'],
    help='Model architecture'
  )
  parser.add_argument(
    '--sure_threshold',
    type=float,
    default=0.5,
    help='Confidence threshold for SURE label'
  )
  parser.add_argument(
    '--unsure_threshold',
    type=float,
    default=0.3,
    help='Confidence threshold for UNSURE label (below = IMPOSTOR)'
  )
  parser.add_argument(
    '--top_k',
    type=int,
    default=3,
    help='Number of top matches to record'
  )
  parser.add_argument(
    '--no_copy',
    action='store_true',
    help='Do not copy files to labeled subdirectories'
  )
  
  args = parser.parse_args()

  if args.metadata_file is None:
    probe_parent = os.path.dirname(args.probe_dir)
    potential_metadata = os.path.join(probe_parent, 'probe_positive_metadata.json')
    if os.path.exists(potential_metadata):
      args.metadata_file = potential_metadata

  labeler = ProbeLabeler(
    gallery_path=args.gallery_path,
    model_type=args.model_type,
    architecture=args.architecture,
    sure_threshold=args.sure_threshold,
    unsure_threshold=args.unsure_threshold
  )

  summary = labeler.process_probe_directory(
    probe_dir=args.probe_dir,
    output_dir=args.output_dir,
    metadata_file=args.metadata_file,
    copy_files=not args.no_copy,
    top_k=args.top_k
  )


if __name__ == '__main__':
  main()