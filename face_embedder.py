import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
import cv2
from PIL import Image
import net

ADAFACE_MODELS = {
  'ir_50': 'pretrained/adaface_ir50_ms1mv2.ckpt',
  'ir_101': 'pretrained/adaface_ir101_ms1mv3.ckpt',
}

class FaceEmbedder:
  def __init__(self, 
                architecture='ir_101',
                model_path=None,
                device=None):
    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
      self.device = device

    if model_path is None:
      if architecture not in ADAFACE_MODELS:
        raise ValueError(f"Unknown architecture: {architecture}. "
                      f"Available: {list(ADAFACE_MODELS.keys())}")
      model_path = ADAFACE_MODELS[architecture]
  
    print(f"Loading AdaFace model ({architecture}) from {model_path}...")
    
    self.model = net.build_model(architecture)

    statedict = torch.load(model_path, map_location=self.device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    self.model.load_state_dict(model_statedict)
    
    self.model.to(self.device)
    self.model.eval()
    
    print(f"AdaFace model loaded successfully on {self.device}")
    self.input_size = (112, 112)
    self.mean = 0.5
    self.std = 0.5
    
  def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
    if face_image.shape[:2] != self.input_size:
      face_image = cv2.resize(face_image, self.input_size, 
                            interpolation=cv2.INTER_LINEAR)
    
    bgr_img = face_image[:, :, ::-1]
    
    bgr_img = (bgr_img / 255.0 - self.mean) / self.std
    
    face_tensor = torch.from_numpy(bgr_img.transpose(2, 0, 1)).float()
    
    face_tensor = face_tensor.unsqueeze(0)
    
    return face_tensor

  def extract_embedding(self, 
                        face_image: np.ndarray,
                        normalize=True) -> np.ndarray:
      face_tensor = self.preprocess(face_image).to(self.device)
      
      with torch.no_grad():
        features, norm = self.model(face_tensor)
        embedding = features
      
      embedding = embedding.cpu().numpy().squeeze()
      
      if normalize:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
      
      return embedding

  def extract_embeddings_batch(self,
                              face_images: List[np.ndarray],
                              normalize=True,
                              batch_size=32) -> np.ndarray:
    if len(face_images) == 0:
      return np.array([])
    
    all_embeddings = []
    
    for i in range(0, len(face_images), batch_size):
      batch = face_images[i:i + batch_size]
      
      batch_tensors = []
      for face_img in batch:
        face_tensor = self.preprocess(face_img)
        batch_tensors.append(face_tensor)
      batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
      
      with torch.no_grad():
        features, norm = self.model(batch_tensor)
        embeddings = features
      embeddings = embeddings.cpu().numpy()
      
      all_embeddings.append(embeddings)
    all_embeddings = np.vstack(all_embeddings)
    
    if normalize:
      norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
      all_embeddings = all_embeddings / (norms + 1e-8)
  
    return all_embeddings

  def compute_similarity(self, 
                        embedding1: np.ndarray, 
                        embedding2: np.ndarray) -> float:
    embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
    embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
    
    return np.dot(embedding1, embedding2)

  def compute_similarity_batch(self,
                              embedding: np.ndarray,
                              gallery_embeddings: np.ndarray) -> np.ndarray:
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    gallery_embeddings = gallery_embeddings / (norms + 1e-8)
    similarities = np.dot(gallery_embeddings, embedding)
    
    return similarities

  def aggregate_embeddings(self,
                          embeddings: np.ndarray,
                          method='mean') -> np.ndarray:
    if len(embeddings) == 0:
      raise ValueError("Cannot aggregate empty embeddings")
    
    if len(embeddings) == 1:
      return embeddings[0]
    
    if method == 'mean':
      aggregated = np.mean(embeddings, axis=0)
    elif method == 'median':
      aggregated = np.median(embeddings, axis=0)
    elif method == 'weighted_mean':
      similarities = np.dot(embeddings, embeddings.T)
      weights = np.mean(similarities, axis=1)
      weights = weights / np.sum(weights)
      aggregated = np.sum(embeddings * weights[:, np.newaxis], axis=0)
    else:
      raise ValueError(f"Unknown aggregation method: {method}")

    aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)
    
    return aggregated


if __name__ == '__main__':
  print("Testing FaceEmbedder...")
  print("\nTesting with IR-101 architecture...")
  embedder = FaceEmbedder(architecture='ir_101')
  dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
  embedding = embedder.extract_embedding(dummy_face)
  print(f"Embedding shape: {embedding.shape}")
  print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
  print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
  
  print("\nExtracting batch embeddings...")
  dummy_faces = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                  for _ in range(5)]
  embeddings = embedder.extract_embeddings_batch(dummy_faces)
  print(f"Embeddings shape: {embeddings.shape}")
  print("\nComputing similarities...")
  sim = embedder.compute_similarity(embeddings[0], embeddings[1])
  print(f"Similarity between emb[0] and emb[1]: {sim:.4f}")
  sims = embedder.compute_similarity_batch(embeddings[0], embeddings[1:])
  print(f"Similarities to all others: {sims}")
  print("\nAggregating embeddings...")
  aggregated = embedder.aggregate_embeddings(embeddings, method='mean')
  print(f"Aggregated embedding shape: {aggregated.shape}")
  print(f"Aggregated embedding norm: {np.linalg.norm(aggregated):.4f}")

  print("\nComputing similarity matrix...")
  similarity_matrix = np.dot(embeddings, embeddings.T)
  print(f"Similarity matrix shape: {similarity_matrix.shape}")
  print(f"Diagonal (self-similarity): {np.diag(similarity_matrix)}")
  
  print("\nFaceEmbedder test completed successfully!")