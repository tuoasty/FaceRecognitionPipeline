import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
import argparse

class ProbeSegmenter:
  def __init__(self, 
                yaw_threshold: float = 35.0,
                pitch_threshold: float = 30.0,
                blur_threshold: float = 70.0,
                det_score_threshold: float = 0.7):
    self.yaw_threshold = yaw_threshold
    self.pitch_threshold = pitch_threshold
    self.blur_threshold = blur_threshold
    self.det_score_threshold = det_score_threshold
    self.categories = [
      'baseline',
      'center',
      'left',
      'right',
      'high_yaw',
      'high_pitch',
      'blur',
      'low_quality'
    ]
  
  def categorize_face(self, metadata: Dict) -> List[str]:
    categories = []
    
    yaw = abs(metadata.get('yaw', 0.0))
    pitch = abs(metadata.get('pitch', 0.0))
    blur_score = metadata.get('blur_score', 0.0)
    det_score = metadata.get('det_score', 1.0)
    angle = metadata.get('angle', 'center')

    is_baseline = (
      yaw <= self.yaw_threshold and
      pitch <= self.pitch_threshold and
      blur_score >= self.blur_threshold and
      det_score >= self.det_score_threshold
    )
    
    if is_baseline:
      categories.append('baseline')
    
    if angle == 'center':
      categories.append('center')
    elif angle == 'left':
      categories.append('left')
    elif angle == 'right':
      categories.append('right')
    if yaw > self.yaw_threshold:
      categories.append('high_yaw')
    
    if pitch > self.pitch_threshold:
      categories.append('high_pitch')
    
    if blur_score < self.blur_threshold:
      categories.append('blur')
    
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
    print("\nThresholds:")
    print(f"  Yaw: {self.yaw_threshold}°")
    print(f"  Pitch: {self.pitch_threshold}°")
    print(f"  Blur: {self.blur_threshold}")
    print(f"  Detection score: {self.det_score_threshold}")
    print("="*70 + "\n")

    with open(metadata_file, 'r') as f:
      metadata_list = json.load(f)
    
    print(f"Loaded metadata for {len(metadata_list)} faces\n")

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
        if abs(m.get('yaw', 0)) <= self.yaw_threshold and
          abs(m.get('pitch', 0)) <= self.pitch_threshold and
          m.get('blur_score', 0) >= self.blur_threshold and
          m.get('det_score', 1) >= self.det_score_threshold
      )
      print(f"  Baseline (clean) faces: {baseline_count}/{total} ({baseline_count/total*100:.1f}%)")
      
      high_yaw_count = sum(1 for m in metadata_list if abs(m.get('yaw', 0)) > self.yaw_threshold)
      high_pitch_count = sum(1 for m in metadata_list if abs(m.get('pitch', 0)) > self.pitch_threshold)
      blur_count = sum(1 for m in metadata_list if m.get('blur_score', 0) < self.blur_threshold)
      low_det_count = sum(1 for m in metadata_list if m.get('det_score', 1) < self.det_score_threshold)
      
      print(f"  High yaw: {high_yaw_count}/{total} ({high_yaw_count/total*100:.1f}%)")
      print(f"  High pitch: {high_pitch_count}/{total} ({high_pitch_count/total*100:.1f}%)")
      print(f"  Blurry: {blur_count}/{total} ({blur_count/total*100:.1f}%)")
      print(f"  Low detection score: {low_det_count}/{total} ({low_det_count/total*100:.1f}%)")

      angles = {}
      for m in metadata_list:
        angle = m.get('angle', 'center')
        angles[angle] = angles.get(angle, 0) + 1
      
      print(f"\nAngle Distribution:")
      for angle, count in sorted(angles.items()):
        print(f"  {angle}: {count}/{total} ({count/total*100:.1f}%)")
      
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
    default='output/preprocessed/probe_positive_segmented',
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
    '--det_score_threshold',
    type=float,
    default=0.7,
    help='Detection score threshold (below = low quality)'
  )
  parser.add_argument(
    '--symlink',
    action='store_true',
    help='Create symlinks instead of copying files (saves space)'
  )
  
  args = parser.parse_args()
  
  segmenter = ProbeSegmenter(
    yaw_threshold=args.yaw_threshold,
    pitch_threshold=args.pitch_threshold,
    blur_threshold=args.blur_threshold,
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