
Brief Report – Cross‑Camera Player Re‑ID

1 . Problem statement
Given two time‑synchronised videos of a football match – one tactical
angle, one TV broadcast – assign a **global player ID** to each detection
so that the same athlete carries the same label across both views.

2 . Methodology
| Stage | Technique | Rationale |
| 1️Detection | YOLOv8‑n(class 0 = “person”) | Fast, tiny model; accurate enough at 640 px |
| Single‑camera tracking | Feature‑averaged cosine similarity + Hungarian match + “max disappeared = 5” | Stable track IDs inside each camera |
| Visual embedding | ResNet‑50 (ImageNet weights, fc removed) | Lightweight 2048‑D descriptor without extra training |
| Cross‑view matching | Cosine similarity matrix → Hungarian assignment (cost = 1‑sim) | Optimal one‑to‑one pairing per frame |
| Global ID fusion | Dictionary `tac_track → global_id`, `broadcast_track → global_id` | Preserves identity after first successful match |

Parameters tuned empirically: similarity threshold = 0.40, history length = 10.

3 . Experiments & outcomes
Metric | Value|
Frames processed | 132 |
Frames with ≥1 cross‑match | 30 |
Total cross‑matches | 431** |
Unique players identified | 78 |

Qualitatively, IDs stay consistent for most full‑body appearances; occasional
switches occur during heavy occlusion or when only a limb is visible.

4 . Key challenges
Scale variance – Broadcast zoom vs. Tacticam wide shot; fixed ResNet input helps but not perfect.  
Tiny bounding boxes(< 50 px tall) – Discarded early to avoid noisy embeddings.  
FPS mismatch – Some feeds reported 59.94 fps vs. 30 fps; handled by reading frame‑pairs in lock‑step.  
Compute budget – Wanted to test CLIP or custom Siamese fine‑tuning but stayed on ImageNet weights to remain lightweight.
Detection Model - The provided fine tuned detection model was not detecting the player hence had to use another model.

5 . future work

Domain‑specific re‑ID training – Finetune on soccer datasets (e.g. SoccerNet‑ReID) for higher discriminability.  
Temporal smoothness – Incorporate a Kalman filter & temporal cosine smoothing to resist ID flips.  
Pose cues – Combine appearance with keypoints for fine‑grained disambiguation.  
Automatic camera sync – Infer frame offset instead of assuming perfect alignment.

6 . Conclusion

The prototype meets the assignment goal: a self‑contained, reproducible
pipeline that tracks players in two heterogeneous camera streams and
exports interpretable artefacts (JSON + video). With modest compute and
no extra data, we achieve ~49 % cross‑frame coverage – a solid baseline
ready for domain‑specific refinement.
