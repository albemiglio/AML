# Enhancing 6D Object Pose Estimation: From RGB Baseline to Lightweight RGB-D Global Fusion

## Abstract
Accurate 6D object pose estimation is a critical challenge in computer vision, with applications ranging from robotic manipulation to augmented reality. In this paper, we present a comprehensive pipeline for estimating the 3D translation and 3D rotation of objects using the LineMod dataset. We propose a disjointed, phase-oriented architecture starting with a lightweight 2D object detector (YOLO11n) to isolate regions of interest, followed by a dual-head ResNet-50 baseline that predicts quaternions and translation vectors from RGB images. To overcome the inherent depth-ambiguity of monocular vision, we extend our pipeline with a lightweight RGB-D Global Fusion model inspired by DenseFusion. Furthermore, we replace naive rotation regression with a 6D continuous representation, ensuring the generation of mathematically valid orthogonal rotation matrices. Our results demonstrate a dramatic improvement in pose accuracy, leaping from a 1.8% accuracy in the RGB-only baseline to 98.4% when leveraging depth data and 6D continuous representations, proving the efficacy of our architectural choices.

---

## 1. Introduction
The ability to perceive the exact 6D pose — comprising a 3D translation vector and a 3x3 rotation matrix — of an object is a fundamental prerequisite for advanced robotic interactions in unstructured environments. While 2D object detection has seen tremendous advancements with the advent of Convolutional Neural Networks (CNNs), lifting these predictions into the 3D space remains challenging due to the loss of depth information during the camera projection process. 

Traditional approaches relying solely on RGB images often struggle with depth ambiguity, especially when dealing with texture-less or symmetric objects. The introduction of affordable RGB-D sensors has mitigated this issue, allowing networks to fuse color features with explicit geometric structures. 

In this work, we tackle the 6D pose estimation problem through an incremental engineering approach, structured in three consecutive phases:
1. **2D Object Detection**: We establish a robust 2D detection framework using YOLO11n to extract precise Regions of Interest (RoI).
2. **RGB-only Baseline**: We develop an RGB-only baseline utilizing a ResNet-50 backbone equipped with a dual regression head to predict both the translation vector and the rotation quaternion, optimized via a balanced composite loss function.
3. **RGB-D Global Fusion**: We introduce a lightweight RGB-D Global Fusion architecture inspired by DenseFusion. To guarantee the physical validity of the predicted rotation matrices, we implement a 6D continuous rotation representation, avoiding the pitfalls of naive 9D linear regression.

We comprehensively evaluate our models on the LineMod dataset, highlighting the severe limitations of pinhole-based depth estimation and demonstrating the overwhelming superiority of multimodal fusion. Our main contributions are:
- A modular, phase-oriented pipeline that incrementally addresses the 6D pose estimation problem.
- A systematic analysis of the failure modes of monocular (RGB-only) pose estimation.
- A lightweight Global Fusion architecture that achieves 98.4% accuracy on the LineMod subset.
- The adoption of 6D continuous rotation representations to ensure the generation of valid orthogonal matrices.

---

## 2. Related Work

### 2.1 2D Object Detection
Modern object detection is dominated by deep learning frameworks. The YOLO (You Only Look Once) family has set the standard for real-time detection by framing the problem as a single regression task. In our pipeline, we utilize YOLO11, specifically the Nano variant, to minimize computational overhead during the preprocessing stage while maintaining high recall. The role of the detector in our architecture is to produce tight bounding box crops that isolate the object from the background clutter — a critical prerequisite for the subsequent pose estimation stages.

### 2.2 6D Pose Estimation from RGB
Early deep learning methods for 6D pose estimation directly regress the 3D translation and orientation from RGB images. PoseCNN introduced an end-to-end architecture that localizes objects and estimates poses via a regression head. While effective for rotation, these methods inherently suffer from depth ambiguity: predicting the Z-axis translation relies heavily on the apparent size of the bounding box, an approximation that fails drastically for elongated or asymmetric objects viewed from varying angles.

### 2.3 RGB-D Fusion
To address the limitations of monocular vision, researchers have leveraged depth sensors to provide explicit geometric information. DenseFusion proposed a pixel-wise dense fusion network that extracts features from RGB images and 3D point clouds separately before fusing them geometrically. While highly robust against occlusions, DenseFusion is computationally expensive due to point cloud processing. Our work proposes a "Global Fusion" alternative, extracting 1D latent vectors from both modalities and fusing them in a shared latent space, trading off dense spatial alignment for significant computational efficiency.

### 2.4 Rotation Representations
Representing 3D rotations in neural networks is notoriously difficult. Euler angles suffer from Gimbal Lock, and quaternions require enforcement of unit length and suffer from antipodal ambiguity. Naively regressing a 3x3 matrix (9 values) often yields non-orthogonal matrices that distort the 3D space. We adopt the 6D continuous representation proposed by Zhou et al. (2019), which utilizes a Gram-Schmidt orthogonalization process to guarantee the generation of a valid rotation matrix, stabilizing the gradient flow during training.

---

## 3. Methodology
Our proposed pipeline is divided into three consecutive phases: 2D Detection, RGB-only Baseline, and RGB-D Fusion.

### 3.1 2D Object Detection (Phase 2)
The first stage of our pipeline aims to isolate the object from the background clutter. We fine-tune a YOLO11n model on the LineMod dataset. Rather than evaluating the model on a standard mAP@50 metric, we optimize for mAP@50-95. A tight bounding box is mathematically critical for the subsequent pose estimation stages: an oversized crop introduces background noise that can severely corrupt the rotation features extracted by the ResNet backbone.

The LineMod dataset annotations are converted into the normalized center coordinates required by YOLO:
$c_x = (x + w/2)/W, \quad c_y = (y + h/2)/H$

### 3.2 RGB-only Baseline (Phase 3)
We build a baseline model using a ResNet-50 backbone initialized with ImageNet weights. The original classification head is replaced by a custom dual-head regression module:
- **Translation Head (`tvec_head`):** A linear layer mapping the 2048D feature vector to a 3D vector representing the X, Y, Z translation in meters.
- **Rotation Head (`quat_head`):** A linear layer predicting a 4D quaternion, strictly normalized via L2-normalization in the forward pass.

**Balanced Loss Function:** Training both heads simultaneously requires balancing their respective gradients. We employ a composite loss:
$Loss = (1 - |q_{pred} \cdot q_{gt}|) + \lambda \cdot MSE(t_{pred}, t_{gt})$
where the first term is the rotation loss (invariant to the quaternion antipodal ambiguity) and the second is the Mean Squared Error for translation. The parameter $\lambda$ is set to 1.0.

**Pinhole Approximation:** As an additional baseline, we implement a geometric depth estimation using the pinhole camera model. Given a bounding box with pixel size $s = \max(w, h)$ and the known real-world object diameter $d$, the depth is estimated as $Z = (f_x \cdot d) / s$.

### 3.3 RGB-D Global Fusion (Phase 4)
To overcome the depth ambiguity of Phase 3, we introduce depth data through a multi-modal architecture. 

**Depth Preprocessing:** Raw depth maps contain invalid pixels (Z=0) due to sensor reflections. We apply a mathematical clipping filter, clamping depth values between 0.0 and 3.0 meters, effectively neutralizing sensor noise while covering the entire operative range of the dataset.

**Global Fusion Architecture:**
Our architecture extracts global features from three modalities:
1. **RGB Branch**: A pre-trained ResNet-50 backbone yields a 2048D feature vector.
2. **Depth Branch**: A secondary CNN (a ResNet-18 initialized on ImageNet via channel replication) yields a 512D feature vector.
3. **Meta-Branch**: An auxiliary encoder processes an 8D vector of geometric metadata (normalized bounding box coordinates and camera intrinsics) into a 64D prior.

These three vectors are concatenated ($2048 + 512 + 64 = 2624D$) and passed through a Multi-Layer Perceptron (MLP) to project them into a shared 256D latent space, which feeds the final rotation and translation heads. This "Global Fusion" approach differs from DenseFusion as it relies on global concatenation rather than dense per-pixel fusion.

### 3.4 6D Continuous Rotation Representation
Instead of predicting quaternions or a raw 9D matrix, our Phase 4 model's rotation head outputs 6 values. These are mapped to a valid rotation matrix via a differentiable Gram-Schmidt process. Given two 3D vectors $a_1, a_2$:
$b_1 = a_1 / \|a_1\|$
$b_2 = a_2 - (b_1 \cdot a_2)b_1$ (normalized)
$b_3 = b_1 \times b_2$
The resulting matrix $R = [b_1 | b_2 | b_3]$ is guaranteed to be orthogonal.

**ADD Loss:**
Unlike the baseline, the Phase 4 model is trained directly using the ADD (Average Distance of Model Points) metric as the loss function, which minimizes the distance between 3D model points transformed by the predicted vs. ground-truth poses.

---

## 4. Experimental Setup

### 4.1 Dataset and Metrics
We evaluate our models on a subset of the LineMod dataset. Due to data availability in the provided preprocessed source, our subset consists of 13 objects (Ape, Benchvise, Cam, Can, Cat, Driller, Duck, Eggbox, Glue, Holepuncher, Iron, Lamp, Phone), excluding the *Bowl* (ID 3) and *Cup* (ID 7) classes which were not present in the local repository. 

Performance is measured using the standard ADD metric (Average Distance of Model Points). A prediction is considered correct if the ADD error is less than 10% of the object's diameter. For symmetric objects (e.g., Eggbox, Glue), we implement the ADD-S metric, computing the distance to the closest point rather than relying on strict pairwise correspondences, thereby preventing unfair penalization of specularly correct poses.

### 4.2 Training Details
Models were trained using the Adam optimizer with an initial learning rate of $1e^{-4}$ and a weight decay of $1e^{-4}$. A `ReduceLROnPlateau` scheduler was employed to halve the learning rate upon validation stagnation. For the Phase 3 baseline, we utilized a freeze/unfreeze strategy, locking the ResNet-50 backbone for the first 10 epochs to prevent catastrophic forgetting of the ImageNet weights during the initial gradient explosion of the randomly initialized heads.

---

## 5. Results

### 5.1 Object Detection (Phase 2)
The YOLO11n model achieved near-perfect recall on the test set, providing a robust foundation for the subsequent pose estimation stages. By focusing on the stringent mAP@50-95 metric, we ensured that the bounding boxes adhered tightly to the physical boundaries of the objects.

**Table 1: YOLO11n Detection Performance (Test Split)**
| Metric | Value |
|:---:|:---:|
| **mAP@50** | **99.50%** |
| **mAP@50-95** | **96.03%** |
| Precision | 99.96% |
| Recall | 100.00% |

### 5.2 RGB-only Baseline: The Pinhole Failure (Phase 3)
The Phase 3 baseline demonstrated that while a ResNet-50 can effectively learn 3D rotations from RGB images, it fails dramatically at estimating 3D translation. When using the Ground Truth (GT) translation, the model achieves high accuracy, but this collapses when relying on the predicted translation or the geometric Pinhole approximation.

**Table 2: Phase 3 Baseline Accuracy (ADD/ADD-S < 10% d)**
| Configuration | Translation Source | Global Accuracy (%) |
|:---|:---|:---:|
| **GT Crop + T_gt** | Ground Truth | 92.7% |
| **YOLO + T_gt** | Ground Truth | **92.9%** |
| **YOLO + T_pinhole** | Pinhole Approximation | 9.4% |
| **YOLO + T_pred** | Model Regression | **1.8%** |

This results confirm that the "Pinhole effect"—where depth is estimated based on the 2D bounding box size—is mathematically insufficient for non-spherical objects.

### 5.3 RGB-D Global Fusion (Phase 4)
The introduction of Depth data and the 6D continuous representation in Phase 4 resolved the depth ambiguity. The Global Fusion architecture achieved state-of-the-art performance for a lightweight model on the LineMod subset.

**Table 3: Phase 4 RGB-D Fusion Performance (Test Split)**
| Model Variant | Backbone (RGB/Depth) | Accuracy (ADD < 10% d) | Avg. Error (mm) |
|:---|:---|:---:|:---:|
| **MAIN** | ResNet-50 / ResNet-18 | **98.4%** | 3.66 – 8.70 |

The *Main* variant, utilizing a pre-trained ResNet-18 for the depth branch, suggests that transfer learning benefits even the depth modality when properly adapted.

---

## 6. Discussions and Limitations

While the proposed pipeline is highly accurate and computationally lightweight, we identify three critical theoretical limitations that pave the way for future work:

1. **Aspect Ratio Distortion (Cropping):** 
Currently, the rectangular YOLO bounding boxes are processed via a "Square Crop" (using the maximum side) before resizing to 224x224. While this prevents geometric squishing, it inevitably captures background clutter along the shorter dimension. A mathematically superior approach for future iterations would be "Square Padding", where the exact rectangular crop is padded with zero-value (black) pixels to reach a 1:1 aspect ratio, entirely isolating the object.

2. **The Horizontal Flip Paradox in End-to-End Training:**
Standard 2D augmentation pipelines (like YOLO's default settings) heavily utilize horizontal flipping. While harmless for 2D detection, a horizontal flip physically reverses the true 6D rotation matrix. Because our pipeline is disjoint, we safely omitted flipping from the ResNet dataloader. However, if this architecture were to be unified into an End-to-End model, horizontal flipping must be categorically disabled to prevent gradient corruption.

3. **Global vs. Dense Fusion Trade-off:**
To comply with the requirement for a lightweight architecture, we implemented a Global Fusion strategy (concatenating 1D vectors). This inherently destroys the pixel-to-pixel spatial alignment that characterizes the original DenseFusion architecture. While our results (98.4%) are excellent on the standard LineMod subset, this architectural simplification theoretically reduces the model's resilience against severe occlusions, where dense pixel-wise matching would otherwise excel.

---

## 7. Conclusion
In this paper, we proposed and thoroughly evaluated a modular pipeline for 6D object pose estimation. We demonstrated that while RGB-only networks can effectively learn 3D rotations, they suffer catastrophic failures in translation estimation due to projective geometry limits. By integrating depth data through a lightweight Global Fusion architecture and enforcing SO(3) physical constraints via a 6D continuous representation, we achieved a massive performance leap (from 1.8% to 98.4% accuracy). The project validates the critical importance of proper mathematical representations and well-balanced multi-modal architectures in modern computer vision tasks.

---
## References
[1] Wang, C., Xu, D., Zhu, Y., Martín-Martín, R., Lu, C., Fei-Fei, L., & Savarese, S. (2019). Densefusion: 6d object pose estimation by iterative dense fusion. CVPR.
[2] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the continuity of rotation representations in neural networks. CVPR.
[3] Xiang, Y., Schmidt, T., Narayanan, V., & Fox, D. (2017). PoseCNN: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv.
[4] Hinterstoisser, S., Lepetit, V., Ilic, S., Holzer, S., Bradski, G., Konolige, K., & Navab, N. (2012). Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. ACCV.
[5] Jochko, G. et al. (Ultralytics). YOLO11: Real-Time Object Detection.
