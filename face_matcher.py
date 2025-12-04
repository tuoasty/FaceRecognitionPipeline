import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

from face_embedder import FaceEmbedder
from gallery_manager import GalleryManager

def get_script_dir():
    return Path(__file__).resolve().parent

SCRIPT_DIR = get_script_dir()

class FaceMatcher:
  def __init__(self,
                gallery_path=None,
                similarity_threshold=0.5,
                aggregation_method='majority_vote',
                model_type='adaface',
                architecture='ir_101'):
    self.similarity_threshold = similarity_threshold
    self.aggregation_method = aggregation_method
    self.model_type = model_type
    self.architecture = architecture

    if gallery_path is None:
     gallery_path = str(SCRIPT_DIR / 'gallery' / 'students.pkl')
    
    print("Initializing Face Matcher...")
    print("="*60)

    print("\n1. Loading AdaFace model...")
    self.embedder = FaceEmbedder(architecture=architecture, model_type=model_type)
    print("\n2. Loading student gallery...")
    self.gallery = GalleryManager(gallery_path=gallery_path)
    
    num_students = len(self.gallery.get_all_students())
    if num_students == 0:
        print("\nWARNING: Gallery is empty! Please enroll students first.")
    else:
        print(f"   Loaded {num_students} enrolled students")
    
    print("\n" + "="*60)
    print("Face Matcher ready!")
    print("="*60 + "\n")

  def match_single_face(self, 
                        face_image: np.ndarray,
                        top_k: int = 5) -> List[Tuple[str, str, float]]:
    embedding = self.embedder.extract_embedding(face_image, normalize=True)
    results = self.gallery.search(embedding, top_k=top_k)
    
    return results
  
  def match_track(self, 
                  track_dir: str,
                  top_k: int = 3) -> Optional[Dict]:
    track_id = os.path.basename(track_dir)
    metadata_path = os.path.join(track_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
      print(f"No metadata found for {track_id}")
      return None
    
    with open(metadata_path, 'r') as f:
      metadata = json.load(f)
    face_files = sorted([f for f in os.listdir(track_dir) if f.endswith('.jpg')])
    
    if len(face_files) == 0:
      print(f"No face images found in {track_id}")
      return None
    
    print(f"\nProcessing {track_id}: {len(face_files)} frames")
    
    frame_matches = []
    all_scores = {}
    
    for face_file in face_files:
      face_path = os.path.join(track_dir, face_file)
      
      face_image = cv2.imread(face_path)
      if face_image is None:
        continue
      face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
      matches = self.match_single_face(face_rgb, top_k=top_k)
      
      if len(matches) > 0:
        top_match = matches[0]
        student_id, name, score = top_match
        
        frame_matches.append({
          'frame': face_file,
          'student_id': student_id,
          'name': name,
          'score': float(score),
          'top_k_matches': [
            {'student_id': sid, 'name': n, 'score': float(s)}
            for sid, n, s in matches
          ]
        })
        
        if student_id not in all_scores:
          all_scores[student_id] = []
        all_scores[student_id].append(score)
    
    if len(frame_matches) == 0:
      print(f"No valid matches found")
      return None
    final_match = self._aggregate_matches(frame_matches, all_scores)
    
    if final_match is None:
      best_candidate = self._get_best_candidate(frame_matches, all_scores)
      print(f"Below threshold - Best candidate: {best_candidate['name']} ({best_candidate['student_id']}) - confidence: {best_candidate['confidence']:.3f}")
      
      return {
        'track_id': track_id,
        'recognized': False,
        'reason': 'below_threshold',
        'best_candidate': best_candidate,
        'frame_matches': frame_matches,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat()
      }
      
    student_id = final_match['student_id']
    name = final_match['name']
    confidence = final_match['confidence']
    
    print(f"  ✓ Identified: {name} ({student_id}) - confidence: {confidence:.3f}")
    
    return {
      'track_id': track_id,
      'recognized': True,
      'student_id': student_id,
      'name': name,
      'confidence': confidence,
      'method': self.aggregation_method,
      'num_frames': len(frame_matches),
      'frame_matches': frame_matches,
      'metadata': metadata,
      'timestamp': datetime.now().isoformat()
    }
  
  def match_single_image(self,
                       image_path: str,
                       top_k: int = 5,
                       save_visualization: bool = True) -> Dict:
    print(f"\n{'='*60}")
    print(f"MATCHING IMAGE: {image_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
      raise ValueError(f"Image not found: {image_path}")
    
    from face_recognition import FaceProcessor
    processor = FaceProcessor(
      output_size=112,
      det_size=(640, 640),
      det_thresh=0.5,
      quality_filter_config={
        'min_det_score': 0.5,
        'min_face_size': 40,
        'max_yaw': 60,
        'max_pitch': 45,
        'max_roll': 45,
        'check_blur': True,
        'blur_threshold': 50
      },
      providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    faces = processor.process_image(image_path, return_all=True)
    
    if len(faces) == 0:
      print("No faces detected in image")
      return {
        'image_path': image_path,
        'num_faces': 0,
        'matches': [],
        'timestamp': datetime.now().isoformat()
      }
    
    print(f"Detected {len(faces)} face(s)")

    matches = []
    for idx, face in enumerate(faces):
      print(f"\nFace {idx + 1}:")
      print(f"  Detection score: {face['det_score']:.3f}")
      print(f"  Valid: {face['is_valid']}")
      
      if face['quality_metrics'].get('blur_score'):
        print(f"  Blur score: {face['quality_metrics']['blur_score']:.0f}")
      
      face_rgb = face['aligned_face']
      results = self.match_single_face(face_rgb, top_k=top_k)
      
      if len(results) > 0:
        top_match = results[0]
        student_id, name, score = top_match
        
        recognized = score >= self.similarity_threshold
        
        if recognized:
          print(f"  ✓ Recognized: {name} ({student_id}) - confidence: {score:.3f}")
        else:
          print(f"  ⚠ Below threshold: {name} ({student_id}) - confidence: {score:.3f} (threshold: {self.similarity_threshold})")
        
        match_result = {
          'face_index': idx,
          'bbox': face['bbox'].tolist(),
          'recognized': recognized,
          'confidence': float(score),
          'quality_metrics': face['quality_metrics'],
          'top_matches': [
            {
              'student_id': sid,
              'name': n,
              'score': float(s),
              'rank': rank + 1
            }
            for rank, (sid, n, s) in enumerate(results)
          ]
        }
        
        if recognized:
          match_result['student_id'] = student_id
          match_result['name'] = name
        else:
          match_result['best_candidate'] = {
            'student_id': student_id,
            'name': name,
            'confidence': float(score)
          }
        
        matches.append(match_result)

        print(f"  Top {min(top_k, len(results))} matches:")
        for rank, (sid, n, s) in enumerate(results[:top_k], 1):
          marker = "→" if rank == 1 else " "
          print(f"    {marker} {rank}. {n} ({sid}): {s:.3f}")
      else:
        print(f"  No matches found in gallery")
        matches.append({
          'face_index': idx,
          'bbox': face['bbox'].tolist(),
          'recognized': False,
          'reason': 'no_gallery_matches',
          'quality_metrics': face['quality_metrics']
        })
    if save_visualization:
      output_path = self._save_match_visualization(image_path, faces, matches)
      print(f"\nVisualization saved to: {output_path}")
    
    result = {
      'image_path': image_path,
      'num_faces': len(faces),
      'num_recognized': sum(1 for m in matches if m.get('recognized', False)),
      'matches': matches,
      'threshold': self.similarity_threshold,
      'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n{'='*60}")
    print(f"Summary: {result['num_recognized']}/{result['num_faces']} faces recognized")
    print(f"{'='*60}\n")
    
    return result
  
  def _save_match_visualization(self, 
                               image_path: str,
                               faces: List[Dict],
                               matches: List[Dict]) -> str:
    import cv2
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for face, match in zip(faces, matches):
      bbox = face['bbox']
      x1, y1, x2, y2 = bbox

      if match.get('recognized', False):
        color = (0, 255, 0)
        label = f"{match['name']}\n{match['confidence']:.3f}"
      elif 'best_candidate' in match:
        color = (255, 165, 0) 
        cand = match['best_candidate']
        label = f"{cand['name']}?\n{cand['confidence']:.3f}"
      else:
        color = (255, 0, 0)
        label = "Unknown"
      
      cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 3)

      label_lines = label.split('\n')
      y_offset = y1 - 10
      for line in reversed(label_lines):
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y_offset -= (h + 5)
        cv2.rectangle(image_rgb, (x1, y_offset), (x1 + w, y_offset + h + 5), color, -1)
        cv2.putText(image_rgb, line, (x1, y_offset + h), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    base_dir = os.path.dirname(image_path)
    gallery_name = Path(self.gallery.gallery_path).stem
    output_dir_name = f'{gallery_name}_match_results'
    output_dir = os.path.join(base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f'matched_{filename}')
    
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    
    return output_path

  def _aggregate_matches(self, 
                      frame_matches: List[Dict],
                      all_scores: Dict[str, List[float]]) -> Optional[Dict]:
    min_quality = 0.55 
    min_frames = 3

    quality_matches = [m for m in frame_matches if m['score'] >= min_quality]
    
    if len(quality_matches) < min_frames:
      return None

    votes = Counter([m['student_id'] for m in quality_matches])
    total_votes = len(quality_matches)
    most_common = votes.most_common(2)
    
    winner_id, winner_count = most_common[0]
    winner_ratio = winner_count / total_votes
    
    strong_consensus = winner_ratio > 0.5
    
    if not strong_consensus and len(most_common) > 1:
      second_count = most_common[1][1]
      strong_consensus = winner_ratio > 0.4 and winner_count >= 2 * second_count
    
    if not strong_consensus:
      return None

    winner_scores = [m['score'] for m in quality_matches if m['student_id'] == winner_id]
    avg_score = np.mean(winner_scores)
    
    if avg_score < self.similarity_threshold:
      return None
    
    name = next(m['name'] for m in quality_matches if m['student_id'] == winner_id)
    
    return {
      'student_id': winner_id,
      'name': name,
      'confidence': float(avg_score),
      'consensus_strength': float(winner_ratio),
      'num_quality_frames': len(winner_scores),
      'total_frames_evaluated': len(frame_matches)
    }
  
  def _get_best_candidate(self, frame_matches, all_scores):
    min_quality = 0.55
    quality_matches = [m for m in frame_matches if m['score'] >= min_quality]

    if not quality_matches:
      quality_matches = frame_matches

    votes = Counter([m['student_id'] for m in quality_matches])
    student_id = votes.most_common(1)[0][0]

    quality_scores = [m['score'] for m in quality_matches if m['student_id'] == student_id]
    avg_score = np.mean(quality_scores)
    
    name = next(m['name'] for m in quality_matches if m['student_id'] == student_id)
    
    return {
      'student_id': student_id,
      'name': name,
      'confidence': float(avg_score),
      'num_quality_frames': len(quality_scores)
    }
  
  def process_capture_directory(self, 
                                capture_dir: str,
                                save_results: bool = True) -> Dict:
    if not os.path.exists(capture_dir):
      raise ValueError(f"Capture directory not found: {capture_dir}")
    
    print("\n" + "="*60)
    print(f"PROCESSING CAMERA CAPTURES")
    print(f"Directory: {capture_dir}")
    print("="*60)

    track_dirs = []
    for item in sorted(os.listdir(capture_dir)):
      item_path = os.path.join(capture_dir, item)
      if os.path.isdir(item_path) and item.startswith('track_'):
        track_dirs.append(item_path)
  
    if len(track_dirs) == 0:
      print("No track directories found!")
      return {'error': 'no_tracks'}
    
    print(f"Found {len(track_dirs)} tracks to process\n")

    results = []
    recognized_count = 0
    unrecognized_count = 0
    
    for track_dir in track_dirs:
      result = self.match_track(track_dir, top_k=3)
      
      if result is not None:
        results.append(result)
        
        if result['recognized']:
          recognized_count += 1
        else:
          unrecognized_count += 1
        
        if save_results:
          result_path = os.path.join(track_dir, 'recognition_result.json')
          with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    summary = self._generate_summary(results, recognized_count, unrecognized_count)

    if save_results:
      results_dir_name = f'{self.model_type}_{self.architecture}_results'
      results_dir = os.path.join(capture_dir, results_dir_name)
      os.makedirs(results_dir, exist_ok=True)
      
      summary_path = os.path.join(results_dir, 'recognition_summary.json')
      with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
      print(f"\nSummary saved to: {summary_path}")
    
    self._print_summary(summary)
    
    return summary
  
  def _generate_summary(self, 
                        results: List[Dict],
                        recognized: int,
                        unrecognized: int) -> Dict:
      student_counts = Counter()
      for result in results:
        if result['recognized']:
          student_counts[result['name']] += 1
      
      confidences = [r['confidence'] for r in results if r['recognized']]
      avg_confidence = np.mean(confidences) if confidences else 0

      below_threshold = []
      for result in results:
        if not result['recognized'] and 'best_candidate' in result:
          below_threshold.append(result['best_candidate'])
      
      return {
        'total_tracks': len(results),
        'recognized': recognized,
        'unrecognized': unrecognized,
        'recognition_rate': recognized / len(results) * 100 if results else 0,
        'avg_confidence': float(avg_confidence),
        'student_appearances': dict(student_counts.most_common()),
        'below_threshold_candidates': below_threshold,
        'unique_students': len(student_counts),
        'timestamp': datetime.now().isoformat(),
        'settings': {
          'similarity_threshold': self.similarity_threshold,
          'aggregation_method': self.aggregation_method
        }
      }
  
  def _print_summary(self, summary: Dict):
    """Print summary to console"""
    print("\n" + "="*60)
    print("RECOGNITION SUMMARY")
    print("="*60)
    print(f"Total tracks: {summary['total_tracks']}")
    print(f"Recognized: {summary['recognized']} ({summary['recognition_rate']:.1f}%)")
    print(f"Unrecognized: {summary['unrecognized']}")
    print(f"Average confidence: {summary['avg_confidence']:.3f}")
    print(f"Unique students detected: {summary['unique_students']}")
    
    if summary['student_appearances']:
      print("\nStudent appearances:")
      for name, count in summary['student_appearances'].items():
        print(f"  • {name}: {count} track(s)")

    if summary.get('below_threshold_candidates'):
      print("\nBelow threshold (not recognized):")
      for candidate in summary['below_threshold_candidates']:
          print(f"  • {candidate['name']}: confidence {candidate['confidence']:.3f} (needed {self.similarity_threshold})")
    
    print("="*60 + "\n")


def main():
  parser = argparse.ArgumentParser(
    description='Match detected faces against student gallery'
  )
  parser.add_argument(
    '--capture_dir',
    type=str,
    default=str(SCRIPT_DIR / 'output' / 'camera_captures'),
    help='Directory containing camera capture tracks'
  )
  parser.add_argument(
    '--gallery_path',
    type=str,
    default=str(SCRIPT_DIR / 'gallery' / 'students.pkl'),
    help='Path to student gallery database'
  )
  parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Similarity threshold for positive match (0-1)'
  )
  parser.add_argument(
    '--aggregation',
    type=str,
    default='consensus',
    choices=['consensus', 'majority_vote', 'avg_similarity', 'max_similarity'],  # ADD 'consensus'
    help='Method to aggregate multi-frame matches'
  )
  parser.add_argument(
    '--no_save',
    action='store_true',
    help='Do not save recognition results to files'
  )
  parser.add_argument(
    '--single_image',
    type=str,
    default=None,
    help='Path to single image to match (instead of processing capture directory)'
  )
  parser.add_argument(
    '--top_k',
    type=int,
    default=5,
    help='Number of top matches to show per face'
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
    help='Model architecture (ir_50 or ir_101)'
  )
    
  args = parser.parse_args()

  matcher = FaceMatcher(
    gallery_path=args.gallery_path,
    similarity_threshold=args.threshold,
    aggregation_method=args.aggregation,
    model_type=args.model_type,
    architecture=args.architecture
  )

  summary = matcher.process_capture_directory(
    capture_dir=args.capture_dir,
    save_results=not args.no_save
  )

  if args.single_image:
    result = matcher.match_single_image(
        image_path=args.single_image,
        top_k=args.top_k,
        save_visualization=True
    )
  else:
      summary = matcher.process_capture_directory(
          capture_dir=args.capture_dir,
          save_results=not args.no_save
      )

if __name__ == '__main__':
  main()