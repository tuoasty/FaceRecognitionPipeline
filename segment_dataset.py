import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
import argparse

class ProbeSegmenter:
  def __init__(self, 
              pose_easy_threshold: float = 15.0,
              pose_medium_threshold: float = 30.0,
              face_large_threshold: int = 150,
              face_medium_threshold: int = 80,
              blur_sharp_percentile: float = 50.0,
              blur_blurry_percentile: float = 20.0,
              det_score_threshold: float = 0.7):
    self.pose_easy_threshold = pose_easy_threshold
    self.pose_medium_threshold = pose_medium_threshold
    self.face_large_threshold = face_large_threshold
    self.face_medium_threshold = face_medium_threshold
    self.blur_sharp_percentile = blur_sharp_percentile
    self.blur_blurry_percentile = blur_blurry_percentile
    self.det_score_threshold = det_score_threshold
    self.blur_sharp_threshold = None
    self.blur_blurry_threshold = None
    self.categories = [
      'baseline',
      'pose_easy',
      'pose_medium',
      'pose_hard',
      'face_large',
      'face_medium',
      'face_small',
      'blur_sharp',
      'blur_blurry',
      'low_quality'
    ]
  def compute_blur_thresholds(self, metadata_list: List[Dict]):
    blur_scores = [m.get('blur_score', 0.0) for m in metadata_list]
    blur_scores.sort()

    sharp_idx = int(len(blur_scores) * (1 - self.blur_sharp_percentile / 100.0))
    self.blur_sharp_threshold = blur_scores[sharp_idx] if sharp_idx < len(blur_scores) else 0

    blurry_idx = int(len(blur_scores) * (self.blur_blurry_percentile / 100.0))
    self.blur_blurry_threshold = blur_scores[blurry_idx] if blurry_idx < len(blur_scores) else 0

    print(f"  Blur sharp threshold (top 50%): {self.blur_sharp_threshold:.2f}")
    print(f"  Blur blurry threshold (bottom 20%): {self.blur_blurry_threshold:.2f}")

  def categorize_face(self, metadata: Dict) -> List[str]:
    categories = []
    
    yaw = abs(metadata.get('yaw', 0.0))
    pitch = abs(metadata.get('pitch', 0.0))
    blur_score = metadata.get('blur_score', 0.0)
    det_score = metadata.get('det_score', 1.0)
    face_size = metadata.get('face_size', 0)
    
    pose_angle = (yaw ** 2 + pitch ** 2) ** 0.5

    is_baseline = (
      pose_angle <= self.pose_easy_threshold and
      face_size >= self.face_medium_threshold and
      blur_score >= self.blur_sharp_threshold and
      det_score >= 0.7
    )
    
    if is_baseline:
      categories.append('baseline')

    if pose_angle <= self.pose_easy_threshold:
      categories.append('pose_easy')
    elif pose_angle <= self.pose_medium_threshold:
      categories.append('pose_medium')
    else:
      categories.append('pose_hard')

    if face_size >= self.face_large_threshold:
      categories.append('face_large')
    elif face_size >= self.face_medium_threshold:
      categories.append('face_medium')
    else:
      categories.append('face_small')

    if blur_score >= self.blur_sharp_threshold:
      categories.append('blur_sharp')
    if blur_score <= self.blur_blurry_threshold:
      categories.append('blur_blurry')

    if det_score < self.det_score_threshold:
      categories.append('low_quality')
    
    return categories
  def build_filename_mapping(self, input_dir: str, metadata_list: List[Dict]) -> Dict[str, str]:
    actual_files = set(os.listdir(input_dir))
    
    filename_mapping = {}
    unmatched_metadata = []
    
    for metadata in metadata_list:
      original_filename = metadata['filename']
      
      matched = False
      for actual_file in actual_files:
        if actual_file.endswith(original_filename):
          filename_mapping[original_filename] = actual_file
          matched = True
          break
      
      if not matched:
        unmatched_metadata.append(original_filename)
    
    if unmatched_metadata:
      print(f"\nWarning: {len(unmatched_metadata)} metadata entries without matching files:")
      for fname in unmatched_metadata[:10]:
        print(f"  - {fname}")
      if len(unmatched_metadata) > 10:
        print(f"  ... and {len(unmatched_metadata) - 10} more")
    
    print(f"\nSuccessfully mapped {len(filename_mapping)}/{len(metadata_list)} files")
    
    return filename_mapping

  def segment_dataset(self,
                      input_dir: str,
                      metadata_file: str,
                      output_dir: str,
                      copy_files: bool = True):
    print("\n" + "="*70)
    print("PROBE DATASET SEGMENTATION")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Metadata file: {metadata_file}")
    print(f"Output directory: {output_dir}")
    print(f"File operation: {'Copy' if copy_files else 'Symlink'}")
    print("="*70)
    print("="*70 + "\n")

    with open(metadata_file, 'r') as f:
      metadata_list = json.load(f)
    
    print(f"Loaded metadata for {len(metadata_list)} faces\n")
    print("Computing blur thresholds from dataset...")
    self.compute_blur_thresholds(metadata_list)
    print()

    print("Building filename mapping...")
    filename_mapping = self.build_filename_mapping(input_dir, metadata_list)

    category_dirs = {}
    for category in self.categories:
      category_path = os.path.join(output_dir, category)
      os.makedirs(category_path, exist_ok=True)
      category_dirs[category] = category_path

    category_counts = {cat: 0 for cat in self.categories}
    processed_count = 0
    skipped_count = 0

    for metadata in metadata_list:
      original_filename = metadata['filename']

      if original_filename not in filename_mapping:
        print(f"Warning: No matching file for metadata: {original_filename}")
        skipped_count += 1
        continue
      
      actual_filename = filename_mapping[original_filename]
      source_path = os.path.join(input_dir, actual_filename)

      if not os.path.exists(source_path):
        print(f"Warning: File not found: {actual_filename}")
        skipped_count += 1
        continue
      
      categories = self.categorize_face(metadata)
      
      for category in categories:
        dest_path = os.path.join(category_dirs[category], actual_filename)
        
        try:
          if copy_files:
            shutil.copy2(source_path, dest_path)
          else:
            rel_source = os.path.relpath(source_path, category_dirs[category])
            if os.path.exists(dest_path):
              os.remove(dest_path)
            os.symlink(rel_source, dest_path)
          
          category_counts[category] += 1
        except Exception as e:
          print(f"Error processing {actual_filename} -> {category}: {e}")
      
      processed_count += 1
 
      if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{len(metadata_list)} faces...")

    for category in self.categories:
      category_metadata = []
      for metadata in metadata_list:
        categories = self.categorize_face(metadata)
        if category in categories:
          metadata_copy = metadata.copy()
          original_filename = metadata['filename']
          if original_filename in filename_mapping:
            metadata_copy['labeled_filename'] = filename_mapping[original_filename]
          category_metadata.append(metadata_copy)
    
      metadata_path = os.path.join(
        category_dirs[category], 
        f'{category}_metadata.json'
      )
      with open(metadata_path, 'w') as f:
        json.dump(category_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE")
    print("="*70)
    print(f"Total faces processed: {processed_count}")
    print(f"Skipped (not found): {skipped_count}")
    print("\nCategory Distribution:")
    print("-" * 70)
    
    for category in self.categories:
      count = category_counts[category]
      percentage = (count / processed_count * 100) if processed_count > 0 else 0
      print(f"  {category:15s}: {count:5d} faces ({percentage:5.1f}%)")
  
    print("="*70 + "\n")

    self.print_quality_insights(metadata_list)
  
  def print_quality_insights(self, metadata_list: List[Dict]):
    print("Quality Insights:")
    print("-" * 70)

    total = len(metadata_list)

    baseline_count = sum(
      1 for m in metadata_list
      if ((abs(m.get('yaw', 0)) ** 2 + abs(m.get('pitch', 0)) ** 2) ** 0.5) <= self.pose_easy_threshold and
        m.get('face_size', 0) >= self.face_medium_threshold and
        m.get('blur_score', 0) >= self.blur_sharp_threshold and
        m.get('det_score', 1) >= 0.7
    )
    print(f"  Baseline (clean) faces: {baseline_count}/{total} ({baseline_count/total*100:.1f}%)")

    pose_easy = sum(1 for m in metadata_list 
                    if ((abs(m.get('yaw', 0)) ** 2 + abs(m.get('pitch', 0)) ** 2) ** 0.5) <= self.pose_easy_threshold)
    pose_medium = sum(1 for m in metadata_list 
                      if self.pose_easy_threshold < ((abs(m.get('yaw', 0)) ** 2 + abs(m.get('pitch', 0)) ** 2) ** 0.5) <= self.pose_medium_threshold)
    pose_hard = sum(1 for m in metadata_list 
                    if ((abs(m.get('yaw', 0)) ** 2 + abs(m.get('pitch', 0)) ** 2) ** 0.5) > self.pose_medium_threshold)

    print(f"\n  Pose Distribution:")
    print(f"    Easy (0-15°): {pose_easy}/{total} ({pose_easy/total*100:.1f}%)")
    print(f"    Medium (15-30°): {pose_medium}/{total} ({pose_medium/total*100:.1f}%)")
    print(f"    Hard (>30°): {pose_hard}/{total} ({pose_hard/total*100:.1f}%)")

    face_large = sum(1 for m in metadata_list if m.get('face_size', 0) >= self.face_large_threshold)
    face_medium = sum(1 for m in metadata_list if self.face_medium_threshold <= m.get('face_size', 0) < self.face_large_threshold)
    face_small = sum(1 for m in metadata_list if m.get('face_size', 0) < self.face_medium_threshold)

    print(f"\n  Face Size Distribution:")
    print(f"    Large (≥150px): {face_large}/{total} ({face_large/total*100:.1f}%)")
    print(f"    Medium (80-150px): {face_medium}/{total} ({face_medium/total*100:.1f}%)")
    print(f"    Small (<80px): {face_small}/{total} ({face_small/total*100:.1f}%)")

    blur_sharp = sum(1 for m in metadata_list if m.get('blur_score', 0) >= self.blur_sharp_threshold)
    blur_blurry = sum(1 for m in metadata_list if m.get('blur_score', 0) <= self.blur_blurry_threshold)

    print(f"\n  Blur Distribution:")
    print(f"    Sharp (top 50%): {blur_sharp}/{total} ({blur_sharp/total*100:.1f}%)")
    print(f"    Blurry (bottom 20%): {blur_blurry}/{total} ({blur_blurry/total*100:.1f}%)")

    low_det_count = sum(1 for m in metadata_list if m.get('det_score', 1) < self.det_score_threshold)
    print(f"\n  Low detection score: {low_det_count}/{total} ({low_det_count/total*100:.1f}%)")

    print("="*70 + "\n")


def main():
  parser = argparse.ArgumentParser(
    description='Segment probe dataset based on quality metrics for evaluation'
  )
  parser.add_argument(
    '--input_dir',
    type=str,
    default='output/preprocessed/probe_positive',
    help='Directory containing probe images'
  )
  parser.add_argument(
    '--metadata_file',
    type=str,
    default='output/preprocessed/probe_positive_metadata.json',
    help='Path to metadata JSON file'
  )
  parser.add_argument(
    '--output_dir',
    type=str,
    default='output/preprocessed/segmented',
    help='Output directory for segmented images'
  )
  parser.add_argument(
    '--yaw_threshold',
    type=float,
    default=35.0,
    help='Absolute yaw angle threshold for high_yaw category (degrees)'
  )
  parser.add_argument(
    '--pitch_threshold',
    type=float,
    default=25.0,
    help='Absolute pitch angle threshold for high_pitch category (degrees)'
  )
  parser.add_argument(
    '--blur_threshold',
    type=float,
    default=50.0,
    help='Blur score threshold (below = blurry)'
  )
  parser.add_argument(
    '--symlink',
    action='store_true',
    help='Create symlinks instead of copying files (saves space)'
  )
  parser.add_argument(
  '--pose_easy_threshold',
  type=float,
  default=15.0,
  help='Pose angle threshold for easy category (degrees)'
)
  parser.add_argument(
  '--pose_medium_threshold',
  type=float,
  default=30.0,
  help='Pose angle threshold for medium category (degrees)'
  )
  parser.add_argument(
  '--face_large_threshold',
  type=int,
  default=150,
  help='Face size threshold for large faces (pixels)'
  )
  parser.add_argument(
  '--face_medium_threshold',
  type=int,
  default=80,
  help='Face size threshold for medium faces (pixels)'
  )
  parser.add_argument(
  '--blur_sharp_percentile',
  type=float,
  default=50.0,
  help='Top percentile for sharp faces (default: 50%)'
  )
  parser.add_argument(
  '--blur_blurry_percentile',
  type=float,
  default=20.0,
  help='Bottom percentile for blurry faces (default: 20%)'
  )
  parser.add_argument(
  '--det_score_threshold',
  type=float,
  default=0.7,
  help='Detection score threshold (below = low quality)'
  )
  
  args = parser.parse_args()
  
  segmenter = ProbeSegmenter(
    pose_easy_threshold=args.pose_easy_threshold,
    pose_medium_threshold=args.pose_medium_threshold,
    face_large_threshold=args.face_large_threshold,
    face_medium_threshold=args.face_medium_threshold,
    blur_sharp_percentile=args.blur_sharp_percentile,
    blur_blurry_percentile=args.blur_blurry_percentile,
    det_score_threshold=args.det_score_threshold
  )
    
  segmenter.segment_dataset(
    input_dir=args.input_dir,
    metadata_file=args.metadata_file,
    output_dir=args.output_dir,
    copy_files=not args.symlink
  )


if __name__ == '__main__':
  main()