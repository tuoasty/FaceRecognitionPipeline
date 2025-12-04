import json
import pickle
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil
from pathlib import Path

def get_script_dir():
    return Path(__file__).resolve().parent

SCRIPT_DIR = get_script_dir()

@dataclass
class StudentRecord:
  student_id: str
  name: str
  embeddings: np.ndarray
  template_embedding: np.ndarray
  num_samples: int
  enrollment_date: str
  last_updated: str
  metadata: Dict = None
    
  def to_dict(self):
    return {
      'student_id': self.student_id,
      'name': self.name,
      'embeddings': self.embeddings.tolist(),
      'template_embedding': self.template_embedding.tolist(),
      'num_samples': self.num_samples,
      'enrollment_date': self.enrollment_date,
      'last_updated': self.last_updated,
      'metadata': self.metadata or {}
    }
  
    @classmethod
    def from_dict(cls, data: Dict):
      return cls(
        student_id=data['student_id'],
        name=data['name'],
        embeddings=np.array(data['embeddings']),
        template_embedding=np.array(data['template_embedding']),
        num_samples=data['num_samples'],
        enrollment_date=data['enrollment_date'],
        last_updated=data['last_updated'],
        metadata=data.get('metadata', {})
      )


class GalleryManager:
    def __init__(self, 
                 gallery_path=None,
                 aggregation_method='mean'):
      if gallery_path is None:
        gallery_path = str(SCRIPT_DIR / 'gallery' / 'students.pkl')
      self.gallery_path = gallery_path
      self.aggregation_method = aggregation_method
      self.students: Dict[str, StudentRecord] = {}
      
      os.makedirs(os.path.dirname(gallery_path) or '.', exist_ok=True)
      
      if os.path.exists(gallery_path):
        self.load()
        print(f"Loaded gallery with {len(self.students)} students")
      else:
        print("Initialized empty gallery")
    
    def add_student(self,
                   student_id: str,
                   name: str,
                   embeddings: np.ndarray,
                   metadata: Optional[Dict] = None,
                   overwrite: bool = False) -> bool:
      if student_id in self.students and not overwrite:
        print(f"Student {student_id} already exists. Use overwrite=True to replace.")
        return False
    
      if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
      
      template = self._aggregate_embeddings(embeddings)
      
      now = datetime.now().isoformat()
      student = StudentRecord(
        student_id=student_id,
        name=name,
        embeddings=embeddings,
        template_embedding=template,
        num_samples=len(embeddings),
        enrollment_date=now,
        last_updated=now,
        metadata=metadata or {}
      )
      
      self.students[student_id] = student
      print(f"{'Updated' if overwrite else 'Added'} student: {name} ({student_id}) "
          f"with {len(embeddings)} embeddings")
      
      return True
    
    def _filter_quality_embeddings(self, 
                               embeddings: np.ndarray,
                               min_similarity: float = 0.70) -> np.ndarray:
        if len(embeddings) <= 2:
            return embeddings
        
        similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarities, 0)
        avg_similarities = np.mean(similarities, axis=1)

        mask = avg_similarities >= min_similarity
        filtered = embeddings[mask]
        
        if len(filtered) < 2:
            top_indices = np.argsort(avg_similarities)[-2:]
            filtered = embeddings[top_indices]
        
        print(f"    Quality filter: kept {len(filtered)}/{len(embeddings)} embeddings (threshold={min_similarity})")
        return filtered
        
    def update_embeddings(self,
                         student_id: str,
                         new_embeddings: np.ndarray,
                         mode: str = 'append') -> bool:
      if student_id not in self.students:
        print(f"Student {student_id} not found")
        return False
      
      student = self.students[student_id]

      if new_embeddings.ndim == 1:
        new_embeddings = new_embeddings.reshape(1, -1)
      
      if mode == 'append':
        updated_embeddings = np.vstack([student.embeddings, new_embeddings])
      
      elif mode == 'replace':
        updated_embeddings = new_embeddings
      
      elif mode == 'merge':
        all_embeddings = np.vstack([student.embeddings, new_embeddings])
        updated_embeddings = self._remove_outliers(all_embeddings)
      
      else:
        raise ValueError(f"Unknown mode: {mode}")
      
      student.embeddings = updated_embeddings
      student.template_embedding = self._aggregate_embeddings(updated_embeddings)
      student.num_samples = len(updated_embeddings)
      student.last_updated = datetime.now().isoformat()
      
      print(f"Updated embeddings for {student.name} ({student_id}): "
        f"{len(student.embeddings)} total embeddings")
      
      return True
    
    def delete_student(self, student_id: str) -> bool:
        if student_id not in self.students:
            print(f"Student {student_id} not found")
            return False
        
        student_name = self.students[student_id].name
        del self.students[student_id]
        print(f"Deleted student: {student_name} ({student_id})")
        
        return True
    
    def get_student(self, student_id: str) -> Optional[StudentRecord]:
        return self.students.get(student_id)
    
    def get_all_students(self) -> Dict[str, StudentRecord]:
        return self.students
    
    def get_gallery_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        if len(self.students) == 0:
            return np.array([]), []
        
        student_ids = list(self.students.keys())
        embeddings = np.vstack([
            self.students[sid].template_embedding 
            for sid in student_ids
        ])
        
        return embeddings, student_ids
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 5) -> List[Tuple[str, str, float]]:
        if len(self.students) == 0:
            return []
        embeddings, student_ids = self.get_gallery_embeddings()
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = np.dot(embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            sid = student_ids[idx]
            student = self.students[sid]
            results.append((sid, student.name, float(similarities[idx])))
        
        return results
    
    def save(self, path: Optional[str] = None):
        save_path = path or self.gallery_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.students, f)
        json_path = save_path.replace('.pkl', '.json')
        json_data = {
            'num_students': len(self.students),
            'last_saved': datetime.now().isoformat(),
            'students': {
                sid: {
                    'student_id': s.student_id,
                    'name': s.name,
                    'num_samples': s.num_samples,
                    'enrollment_date': s.enrollment_date,
                    'last_updated': s.last_updated,
                    'metadata': s.metadata
                }
                for sid, s in self.students.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Gallery saved to {save_path}")
        print(f"Metadata saved to {json_path}")
    
    def load(self, path: Optional[str] = None):
        load_path = path or self.gallery_path
        
        if not os.path.exists(load_path):
            print(f"Gallery file not found: {load_path}")
            return
        
        with open(load_path, 'rb') as f:
            self.students = pickle.load(f)
        
        print(f"Gallery loaded from {load_path}")
    
    def export_for_backup(self, backup_dir: str, backup_name: str = None):
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if backup_name:
            backup_path = os.path.join(backup_dir, f'{backup_name}_backup_{timestamp}.pkl')
            json_path = os.path.join(backup_dir, f'{backup_name}_backup_{timestamp}.json')
        else:
            backup_path = os.path.join(backup_dir, f'gallery_backup_{timestamp}.pkl')
            json_path = os.path.join(backup_dir, f'gallery_backup_{timestamp}.json')
        
        shutil.copy2(self.gallery_path, backup_path)
        
        json_data = {
            'backup_date': datetime.now().isoformat(),
            'backup_name': backup_name,
            'num_students': len(self.students),
            'students': {sid: s.to_dict() for sid, s in self.students.items()}
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Backup saved to {backup_dir}")
    
    def get_statistics(self) -> Dict:
        if len(self.students) == 0:
            return {
                'num_students': 0,
                'total_embeddings': 0,
                'avg_embeddings_per_student': 0
            }
        
        total_embeddings = sum(s.num_samples for s in self.students.values())
        
        return {
            'num_students': len(self.students),
            'total_embeddings': total_embeddings,
            'avg_embeddings_per_student': total_embeddings / len(self.students),
            'students': [
                {
                    'id': s.student_id,
                    'name': s.name,
                    'num_samples': s.num_samples,
                    'enrollment_date': s.enrollment_date
                }
                for s in self.students.values()
            ]
        }
    
    def _aggregate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 1:
            return embeddings[0]
        
        embeddings = self._filter_quality_embeddings(embeddings)
        
        if self.aggregation_method == 'mean':
            aggregated = np.mean(embeddings, axis=0)
        elif self.aggregation_method == 'median':
            aggregated = np.median(embeddings, axis=0)
        elif self.aggregation_method == 'weighted_mean':
            similarities = np.dot(embeddings, embeddings.T)
            weights = np.mean(similarities, axis=1)
            weights = weights / np.sum(weights)
            aggregated = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        else:
            aggregated = np.mean(embeddings, axis=0)

        aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)
        
        return aggregated
    
    def _remove_outliers(self, embeddings: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        if len(embeddings) <= 2:
            return embeddings
        
        similarities = np.dot(embeddings, embeddings.T)

        avg_similarities = np.mean(similarities, axis=1)

        median_sim = np.median(avg_similarities)
        mask = avg_similarities >= (median_sim * threshold)
        
        return embeddings[mask]


if __name__ == '__main__':
    print("Testing GalleryManager...")

    gallery = GalleryManager(gallery_path=str(SCRIPT_DIR / 'test_gallery' / 'students.pkl'))
    for i in range(3):
        embeddings = np.random.randn(5, 512)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        gallery.add_student(
            student_id=f'S{i+1:03d}',
            name=f'Student {i+1}',
            embeddings=embeddings,
            metadata={'class': '10A', 'section': 'A'}
        )
    
    print("\nTesting search...")
    query = np.random.randn(512)
    query = query / np.linalg.norm(query)
    results = gallery.search(query, top_k=3)
    print("Search results:")
    for sid, name, score in results:
        print(f"  {name} ({sid}): {score:.4f}")

    print("\nUpdating embeddings...")
    new_embeddings = np.random.randn(3, 512)
    new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    gallery.update_embeddings('S001', new_embeddings, mode='append')
    
    print("\nSaving gallery...")
    gallery.save()
    
    print("\nGallery statistics:")
    stats = gallery.get_statistics()
    print(f"  Total students: {stats['num_students']}")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Avg per student: {stats['avg_embeddings_per_student']:.1f}")

    print("\nCreating backup...")
    gallery.export_for_backup(str(SCRIPT_DIR / 'test_gallery' / 'backups'))
    
    print("\nGalleryManager test completed successfully!")