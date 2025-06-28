import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from collections import defaultdict, deque
import torchvision.transforms as transforms
from torchvision.models import resnet50
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import os
import sys
import warnings
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from google.colab import files
import time

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

@dataclass
class PlayerDetection:
    bbox: Tuple[int, int, int, int] 
    confidence: float
    frame_id: int
    camera_id: str
    player_id: Optional[int] = None
    features: Optional[np.ndarray] = None
    track_id: Optional[int] = None  

@dataclass
class MatchResult:
    tacticam_idx: int
    broadcast_idx: int
    similarity_score: float
    confidence: float
    tacticam_id: Optional[int] = None
    broadcast_id: Optional[int] = None

@dataclass
class PlayerTrack:
    player_id: int
    camera_id: str
    feature_history: List[np.ndarray]
    bbox_history: List[Tuple[int, int, int, int]]
    last_seen_frame: int
    confidence_history: List[float]
    
    def get_average_features(self) -> np.ndarray:
        if not self.feature_history:
            return np.zeros(2048)
        return np.mean(self.feature_history, axis=0)
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, 
               frame_id: int, confidence: float):
        self.bbox_history.append(bbox)
        self.feature_history.append(features)
        self.confidence_history.append(confidence)
        self.last_seen_frame = frame_id
        if len(self.feature_history) > 10:
            self.feature_history = self.feature_history[-10:]
            self.bbox_history = self.bbox_history[-10:]
            self.confidence_history = self.confidence_history[-10:]

class VisualFeatureExtractor:
    def __init__(self):
        try:
            self.backbone = resnet50(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Identity()
            self.backbone.eval()
            self.device = device
            self.backbone.to(self.device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            print(f" Feature extractor initialized on {self.device}")
        except Exception as e:
            print(f" Error initializing feature extractor: {e}")
            raise

    def extract_features(self, image_patch: np.ndarray) -> np.ndarray:
        if image_patch is None or image_patch.size == 0:
            return np.zeros(2048)
        try:
            if image_patch.shape[0] < 32 or image_patch.shape[1] < 32:
                return np.zeros(2048)
            if len(image_patch.shape) == 3 and image_patch.shape[2] == 3:
                image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
            else:
                return np.zeros(2048)
            tensor = self.transform(image_patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.backbone(tensor)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return np.zeros(2048)

class PlayerTracker:
    def __init__(self, max_disappeared: int = 5):
        self.tracks = {}  
        self.next_id = 1
        self.max_disappeared = max_disappeared
        
    def update(self, detections: List[PlayerDetection], frame_id: int, 
               frame: np.ndarray, camera_id: str, feature_extractor) -> List[PlayerDetection]:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                patch = frame[y1:y2, x1:x2]
                det.features = feature_extractor.extract_features(patch)
            else:
                det.features = np.zeros(2048)
        if not self.tracks:
            for i, det in enumerate(detections):
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = PlayerTrack(
                    player_id=track_id,
                    camera_id=camera_id,
                    feature_history=[det.features],
                    bbox_history=[det.bbox],
                    last_seen_frame=frame_id,
                    confidence_history=[det.confidence]
                )
                det.track_id = track_id
                det.player_id = track_id
            return detections
        active_tracks = {tid: track for tid, track in self.tracks.items() 
                        if frame_id - track.last_seen_frame <= self.max_disappeared}
        if not active_tracks:
            for det in detections:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = PlayerTrack(
                    player_id=track_id,
                    camera_id=camera_id,
                    feature_history=[det.features],
                    bbox_history=[det.bbox],
                    last_seen_frame=frame_id,
                    confidence_history=[det.confidence]
                )
                det.track_id = track_id
                det.player_id = track_id
            return detections
        track_ids = list(active_tracks.keys())
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = active_tracks[track_id]
                avg_features = track.get_average_features()
                if np.linalg.norm(det.features) > 0 and np.linalg.norm(avg_features) > 0:
                    similarity = np.dot(det.features, avg_features) / (
                        np.linalg.norm(det.features) * np.linalg.norm(avg_features)
                    )
                    similarity_matrix[i, j] = max(0, similarity)
                else:
                    similarity_matrix[i, j] = 0
        if similarity_matrix.size > 0:
            cost_matrix = 1.0 - similarity_matrix
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
            assigned_dets = set()
            assigned_tracks = set()
            for det_idx, track_idx in zip(det_indices, track_indices):
                if similarity_matrix[det_idx, track_idx] > 0.3:  
                    track_id = track_ids[track_idx]
                    track = active_tracks[track_id]
                    det = detections[det_idx]
                    track.update(det.bbox, det.features, frame_id, det.confidence)
                    det.track_id = track_id
                    det.player_id = track_id
                    assigned_dets.add(det_idx)
                    assigned_tracks.add(track_id)
            for i, det in enumerate(detections):
                if i not in assigned_dets:
                    track_id = self.next_id
                    self.next_id += 1
                    self.tracks[track_id] = PlayerTrack(
                        player_id=track_id,
                        camera_id=camera_id,
                        feature_history=[det.features],
                        bbox_history=[det.bbox],
                        last_seen_frame=frame_id,
                        confidence_history=[det.confidence]
                    )
                    det.track_id = track_id
                    det.player_id = track_id
        return detections

class PlayerMatcher:
    def __init__(self, similarity_threshold: float = 0.4):
        self.feature_extractor = VisualFeatureExtractor()
        self.similarity_threshold = similarity_threshold
        self.trackers = {
            'tacticam': PlayerTracker(),
            'broadcast': PlayerTracker()
        }
        self.cross_camera_matches = {}  
        self.global_id_counter = 1000  
        print(f" Player matcher initialized (threshold: {similarity_threshold})")

    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        try:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def match_players(self, tacticam_detections: List[PlayerDetection],
                     broadcast_detections: List[PlayerDetection],
                     tacticam_frame: np.ndarray,
                     broadcast_frame: np.ndarray,
                     frame_id: int) -> List[MatchResult]:
        tacticam_tracked = self.trackers['tacticam'].update(
            tacticam_detections, frame_id, tacticam_frame, 'tacticam', self.feature_extractor
        )
        broadcast_tracked = self.trackers['broadcast'].update(
            broadcast_detections, frame_id, broadcast_frame, 'broadcast', self.feature_extractor
        )
        if not tacticam_tracked or not broadcast_tracked:
            return []
        matches = []
        similarity_matrix = np.zeros((len(tacticam_tracked), len(broadcast_tracked)))
        for i, tac_det in enumerate(tacticam_tracked):
            for j, broad_det in enumerate(broadcast_tracked):
                if tac_det.features is not None and broad_det.features is not None:
                    similarity = self.calculate_similarity(tac_det.features, broad_det.features)
                    similarity_matrix[i, j] = similarity
        if similarity_matrix.size > 0:
            cost_matrix = 1.0 - similarity_matrix
            tac_indices, broad_indices = linear_sum_assignment(cost_matrix)
            for tac_idx, broad_idx in zip(tac_indices, broad_indices):
                score = similarity_matrix[tac_idx, broad_idx]
                if score >= self.similarity_threshold:
                    tac_det = tacticam_tracked[tac_idx]
                    broad_det = broadcast_tracked[broad_idx]
                    if tac_det.track_id in self.cross_camera_matches:
                        global_id = self.cross_camera_matches[tac_det.track_id]
                        broad_det.player_id = global_id
                        tac_det.player_id = global_id
                    elif broad_det.track_id in [v for v in self.cross_camera_matches.values()]:
                        for tac_id, broad_id in self.cross_camera_matches.items():
                            if broad_id == broad_det.track_id:
                                tac_det.player_id = broad_id
                                broad_det.player_id = broad_id
                                break
                    else:
                        global_id = self.global_id_counter
                        self.global_id_counter += 1
                        self.cross_camera_matches[tac_det.track_id] = global_id
                        tac_det.player_id = global_id
                        broad_det.player_id = global_id
                    matches.append(MatchResult(
                        tacticam_idx=tac_idx,
                        broadcast_idx=broad_idx,
                        similarity_score=score,
                        confidence=score,
                        tacticam_id=tac_det.player_id,
                        broadcast_id=broad_det.player_id
                    ))
        return matches

class PlayerReIDSystem:
    def __init__(self, model_path: str = "yolov8n.pt"):
        try:
            self.detector = YOLO(model_path)
            self.matcher = PlayerMatcher()
            print(" Player Re-ID System initialized successfully!")
        except Exception as e:
            print(f" Error initializing system: {e}")
            raise

    def detect_players(self, frame: np.ndarray, camera_id: str, frame_id: int) -> List[PlayerDetection]:
        try:
            results = self.detector(frame, verbose=False)
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            confidence = float(box.conf[0].cpu().numpy())
                            if x2 > x1 and y2 > y1 and confidence > 0.4:
                                width = x2 - x1
                                height = y2 - y1
                                if width > 30 and height > 50:  
                                    detections.append(PlayerDetection(
                                        bbox=(x1, y1, x2, y2),
                                        confidence=confidence,
                                        frame_id=frame_id,
                                        camera_id=camera_id
                                    ))
            return detections
        except Exception as e:
            print(f" Error in player detection frame {frame_id}: {e}")
            return []

    def process_frame_pair(self, tac_frame: np.ndarray, broad_frame: np.ndarray, frame_id: int) -> Dict:
        try:
            tac_dets = self.detect_players(tac_frame, "tacticam", frame_id)
            broad_dets = self.detect_players(broad_frame, "broadcast", frame_id)
            matches = self.matcher.match_players(
                tac_dets, broad_dets, tac_frame, broad_frame, frame_id
            )
            return {
                'frame_id': frame_id,
                'tacticam_detections': tac_dets,
                'broadcast_detections': broad_dets,
                'matches': matches,
                'num_matches': len(matches)
            }
        except Exception as e:
            print(f" Error processing frame pair {frame_id}: {e}")
            return {
                'frame_id': frame_id,
                'tacticam_detections': [],
                'broadcast_detections': [],
                'matches': [],
                'num_matches': 0
            }

    def visualize_results(self, tac_frame: np.ndarray, broad_frame: np.ndarray, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        try:
            tac_vis = tac_frame.copy()
            broad_vis = broad_frame.copy()
            colors = [
                (0, 255, 0),     
                (255, 0, 0),    
                (0, 0, 255),     
                (255, 255, 0),   
                (255, 0, 255),   
                (0, 255, 255),   
                (255, 165, 0),   
                (128, 0, 128),   
                (255, 192, 203), 
                (0, 128, 128),   
                (128, 128, 0),   
                (128, 0, 0),     
            ]
            for det in result['tacticam_detections']:
                if det.player_id is not None:
                    color = colors[det.player_id % len(colors)]
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(tac_vis, (x1, y1), (x2, y2), color, 3)
                    label = f"ID:{det.player_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(tac_vis, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
                    cv2.putText(tac_vis, label, (x1+5, y1-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for det in result['broadcast_detections']:
                if det.player_id is not None:
                    color = colors[det.player_id % len(colors)]
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(broad_vis, (x1, y1), (x2, y2), color, 3)
                    label = f"ID:{det.player_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(broad_vis, (x1, y1-25), (x1+label_size[0]+10, y1), color, -1)
                    cv2.putText(broad_vis, label, (x1+5, y1-8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_text = f"Tacticam - Frame {result['frame_id']} - Matches: {result['num_matches']}"
            cv2.rectangle(tac_vis, (5, 5), (600, 40), (0, 0, 0), -1)
            cv2.putText(tac_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_text = f"Broadcast - Frame {result['frame_id']} - Matches: {result['num_matches']}"
            cv2.rectangle(broad_vis, (5, 5), (600, 40), (0, 0, 0), -1)
            cv2.putText(broad_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return tac_vis, broad_vis
        except Exception as e:
            print(f" Error in visualization: {e}")
            return tac_frame, broad_frame

    def process_videos(self, tac_path: str, broad_path: str, output_path: str = None, max_frames: int = None) -> Dict:
        print(f" Tacticam: {tac_path}")
        print(f" Broadcast: {broad_path}")
        if not os.path.exists(tac_path):
            raise FileNotFoundError(f"Tacticam video not found: {tac_path}")
        if not os.path.exists(broad_path):
            raise FileNotFoundError(f"Broadcast video not found: {broad_path}")
        tac_cap = cv2.VideoCapture(tac_path)
        broad_cap = cv2.VideoCapture(broad_path)
        if not tac_cap.isOpened():
            raise ValueError(f" Could not open tacticam video: {tac_path}")
        if not broad_cap.isOpened():
            raise ValueError(f" Could not open broadcast video: {broad_path}")
        try:
            fps = int(tac_cap.get(cv2.CAP_PROP_FPS))
            width = int(tac_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(tac_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(tac_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f" Video properties: {width}x{height} @ {fps}fps")
            print(f" Total frames available: {total_frames}")
            if max_frames:
                print(f" Processing limited to: {max_frames} frames")
        except Exception as e:
            print(f" Error getting video properties: {e}")
            fps, width, height = 30, 640, 480
        out = None
        if output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
                if not out.isOpened():
                    print(f"Could not create output video at {output_path}")
                    out = None
                else:
                    print(f"Output video will be saved to: {output_path}")
            except Exception as e:
                print(f"Error creating output video: {e}")
                out = None
        results = []
        frame_id = 0
        successful_frames = 0
        total_players_detected = set()
        start_time = time.time()
        try:
            while True:
                if max_frames and frame_id >= max_frames:
                    break 
                ret1, tac_frame = tac_cap.read()
                ret2, broad_frame = broad_cap.read()
                if not ret1 or not ret2:
                    break
                result = self.process_frame_pair(tac_frame, broad_frame, frame_id)
                results.append(result)
                if result['num_matches'] > 0:
                    successful_frames += 1
                for det in result['tacticam_detections']:
                    if det.player_id is not None:
                        total_players_detected.add(det.player_id)
                for det in result['broadcast_detections']:
                    if det.player_id is not None:
                        total_players_detected.add(det.player_id)
                if out is not None:
                    try:
                        tac_vis, broad_vis = self.visualize_results(tac_frame, broad_frame, result)
                        if tac_vis.shape != broad_vis.shape:
                            broad_vis = cv2.resize(broad_vis, (tac_vis.shape[1], tac_vis.shape[0]))
                        combined = np.hstack([tac_vis, broad_vis])
                        out.write(combined)
                    except Exception as e:
                        print(f"Error writing frame {frame_id}: {e}")
                frame_id += 1
                if frame_id % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = frame_id / elapsed
                    print(f"Frame {frame_id} | Matches: {successful_frames} | Players: {len(total_players_detected)} | Speed: {fps_processing:.1f} fps")

        except KeyboardInterrupt:
            print("\n Processing interrupted by user")
        except Exception as e:
            print(f" Error during processing: {e}")
        finally:
            tac_cap.release()
            broad_cap.release()
            if out is not None:
                out.release()
        processing_time = time.time() - start_time
        print(f"Frames processed: {frame_id}")
        print(f"Total time: {processing_time:.1f} seconds")
        print(f"Processed {frame_id} frames with matches in {successful_frames} frames")
        print(f"Total unique players detected: {len(total_players_detected)}")
        total_matches = sum(r['num_matches'] for r in results)
        return {
            'total_frames': frame_id,
            'successful_frames': successful_frames,
            'total_matches': total_matches,
            'unique_players': len(total_players_detected),
            'processing_time': processing_time,
            'results': results
        }

def save_results(results: Dict, json_path: str = "results.json"):
    try:
        print(f"Saving results to {json_path}")
        
        serializable = []
        for result in results['results']:
            ser_result = {
                'frame_id': result['frame_id'],
                'num_matches': result['num_matches'],
                'tacticam_detections': [
                    {
                        'bbox': det.bbox, 
                        'confidence': det.confidence, 
                        'player_id': det.player_id,
                        'track_id': det.track_id,
                        'camera_id': det.camera_id
                    }
                    for det in result['tacticam_detections']
                ],
                'broadcast_detections': [
                    {
                        'bbox': det.bbox, 
                        'confidence': det.confidence, 
                        'player_id': det.player_id,
                        'track_id': det.track_id,
                        'camera_id': det.camera_id
                    }
                    for det in result['broadcast_detections']
                ],
                'matches': [
                    {
                        'tacticam_idx': m.tacticam_idx,
                        'broadcast_idx': m.broadcast_idx,
                        'similarity': m.similarity_score,
                        'confidence': m.confidence,
                        'tacticam_id': m.tacticam_id,
                        'broadcast_id': m.broadcast_id
                    }
                    for m in result['matches']
                ]
            }
            serializable.append(ser_result)
        summary = {
            'total_frames': results['total_frames'],
            'successful_frames': results.get('successful_frames', 0),
            'total_matches': results['total_matches'],
            'unique_players': results['unique_players'],
            'processing_time': results.get('processing_time', 0),
            'avg_matches_per_frame': results['total_matches'] / max(1, results['total_frames']),
            'success_rate': results.get('successful_frames', 0) / max(1, results['total_frames'])
        }
        output_data = {
            'summary': summary,
            'frame_results': serializable
        }
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Results saved successfully!")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def show_sample_results(results: Dict, num_samples: int = 5):
    try:
        print(f"\n sample results")
        frames_with_matches = [r for r in results['results'] if r['num_matches'] > 0]
        if not frames_with_matches:
            print("No frames with matches found!")
            return
        step = max(1, len(frames_with_matches) // num_samples)
        sample_frames = frames_with_matches[::step][:num_samples]
        print(f"Sample frames with matches: {[f['frame_id'] for f in sample_frames]}")

        for frame_result in sample_frames:
            print(f"\n Frame {frame_result['frame_id']}")
            print(f"Matches: {frame_result['num_matches']}")
            tac_ids = [det.player_id for det in frame_result['tacticam_detections'] if det.player_id]
            broad_ids = [det.player_id for det in frame_result['broadcast_detections'] if det.player_id]
            print(f"Tacticam player IDs: {tac_ids}")
            print(f"Broadcast player IDs: {broad_ids}")
            for match in frame_result['matches']:
                print(f"  Match: Tac_ID={match.tacticam_id} ↔ Broad_ID={match.broadcast_id} (similarity: {match.similarity_score:.3f})")
    except Exception as e:
        print(f"Error showing sample results: {e}")

def upload_videos():
    print("Please upload your video files:")
    uploaded = files.upload()
    video_files = []
    for filename in uploaded.keys():
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(filename)
            print(f"Uploaded: {filename}")   
    if len(video_files) < 2:
        print("Please upload at least 2 video files")
        return None, None   
    return video_files[0], video_files[1]

def main_colab(tac_video_path: str, broad_video_path: str, output_video_path: str = None, max_frames: int = 200):
    try:
        reid_system = PlayerReIDSystem("yolov8n.pt")
    except Exception as e:
        print(f" Failed to initialize system: {e}")
        return None
    try:
        results = reid_system.process_videos(
            tac_video_path,
            broad_video_path,
            output_video_path,
            max_frames
        )
    except Exception as e:
        print(f"Video processing failed: {e}")
        return None
    save_results(results)
    print(f"Total frames processed: {results['total_frames']}")
    print(f"Frames with matches: {results.get('successful_frames', 0)}")
    print(f"Total matches made: {results['total_matches']}")
    print(f"Unique players identified: {results['unique_players']}")
    print(f"Processing time: {results.get('processing_time', 0):.1f} seconds")
    if results['total_frames'] > 0:
        avg_matches = results['total_matches']/results['total_frames']
        success_rate = results.get('successful_frames', 0)/results['total_frames']
        print(f"Average matches per frame: {avg_matches:.2f}")
        print(f"Success rate: {success_rate:.2%}")
    if output_video_path:
        print(f"\nOutput video: {output_video_path}")
    print("Detailed results saved to: results.json")
    show_sample_results(results)
    return results

def debug_single_frame(tac_video_path: str, broad_video_path: str, frame_num: int = 50):
    print(f"Debug analysis for frame {frame_num}")
    try:
        reid_system = PlayerReIDSystem("yolov8n.pt")
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return None
    tac_cap = cv2.VideoCapture(tac_video_path)
    broad_cap = cv2.VideoCapture(broad_video_path)
    tac_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    broad_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret1, tac_frame = tac_cap.read()
    ret2, broad_frame = broad_cap.read()
    if not ret1 or not ret2:
        print("Could not read frames")
        return None
    result = reid_system.process_frame_pair(tac_frame, broad_frame, frame_num)
    print(f"\n Frame {frame_num} Analysis:")
    print(f"Tacticam detections: {len(result['tacticam_detections'])}")
    print(f"Broadcast detections: {len(result['broadcast_detections'])}")
    print(f"Cross-camera matches: {result['num_matches']}")
    print("\n Tacticam Detections:")
    for i, det in enumerate(result['tacticam_detections']):
        print(f"  {i}: ID={det.player_id}, TrackID={det.track_id}, Conf={det.confidence:.3f}, BBox={det.bbox}")
    print("\n Broadcast Detections:")
    for i, det in enumerate(result['broadcast_detections']):
        print(f"  {i}: ID={det.player_id}, TrackID={det.track_id}, Conf={det.confidence:.3f}, BBox={det.bbox}")
    print("\n Matches:")
    for match in result['matches']:
        print(f"Tac[{match.tacticam_idx}] ↔ Broad[{match.broadcast_idx}]: "
              f"IDs {match.tacticam_id}↔{match.broadcast_id}, similarity={match.similarity_score:.3f}")
    tac_vis, broad_vis = reid_system.visualize_results(tac_frame, broad_frame, result)
    cv2.imwrite(f'debug_frame_{frame_num}_tacticam.jpg', tac_vis)
    cv2.imwrite(f'debug_frame_{frame_num}_broadcast.jpg', broad_vis)
    print(f"\n Debug images saved: debug_frame_{frame_num}_tacticam.jpg, debug_frame_{frame_num}_broadcast.jpg")
    tac_cap.release()
    broad_cap.release()
    return result

tac_video, broad_video = upload_videos()
if tac_video and broad_video:
   results = main_colab(tac_video, broad_video, 'output.mp4', 300)
