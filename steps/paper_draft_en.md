# Enhancing 6D Object Pose Estimation: From RGB Baseline to Lightweight RGB-D Global Fusion

## Abstract
Accurate 6D object pose estimation is a critical challenge in computer vision, with applications ranging from robotic manipulation to augmented reality. In this paper, we present a comprehensive pipeline for estimating the 3D translation and 3D rotation of objects using the LineMod dataset. We propose a disjointed, phase-oriented architecture starting with a lightweight 2D object detector (YOLO11n) to isolate regions of interest, followed by a dual-head ResNet-50 baseline that predicts quaternions and translation vectors from RGB images. To overcome the inherent depth-ambiguity of monocular vision, we extend our pipeline with a lightweight RGB-D Global Fusion model inspired by DenseFusion. Furthermore, we replace naive rotation regression with a 6D continuous representation, ensuring the generation of mathematically valid orthogonal rotation matrices. Our results demonstrate a dramatic improvement in pose accuracy, leaping from a 1.9% accuracy in the RGB-only baseline to 98.4% when leveraging depth data and 6D continuous representations, proving the efficacy of our architectural choices.

---

## 1. Introduction
The ability to perceive the exact 6D pose (3D position and 3D orientation) of an object is a fundamental prerequisite for advanced robotic interactions in unstructured environments. While 2D object detection has seen tremendous advancements with the advent of Convolutional Neural Networks (CNNs), lifting these predictions into the 3D space remains challenging due to the loss of depth information during the camera projection process. 

Traditional approaches relying solely on RGB images often struggle with depth ambiguity, especially when dealing with texture-less or symmetric objects. The introduction of affordable RGB-D sensors has mitigated this issue, allowing networks to fuse color features with explicit geometric structures. 

In this work, we tackle the 6D pose estimation problem through an incremental engineering approach. We first establish a robust 2D detection framework using YOLO11n to extract precise Regions of Interest (RoI). Subsequently, we develop an RGB-only baseline utilizing a ResNet-50 backbone equipped with a dual regression head to predict both the translation vector and the rotation quaternion, optimized via a balanced composite loss function. Finally, we introduce our core contribution: a lightweight RGB-D Global Fusion architecture. To guarantee the physical validity of the predicted rotation matrices, we implement a 6D continuous rotation representation (Zhou et al., 2019), avoiding the pitfalls of naive 9D linear regression. We comprehensively evaluate our models on the LineMod dataset, highlighting the severe limitations of pinhole-based depth estimation and demonstrating the overwhelming superiority of multimodal fusion.

---

## 2. Related Work

### 2.1 2D Object Detection
Modern object detection is dominated by deep learning frameworks. The YOLO (You Only Look Once) family has set the standard for real-time detection by framing the problem as a single regression task. In our pipeline, we utilize YOLO11, specifically the Nano variant, to minimize computational overhead during the preprocessing stage while maintaining high recall.

### 2.2 6D Pose Estimation from RGB
Early deep learning methods for 6D pose estimation, such as PoseCNN (Xiang et al., 2017), directly regress the 3D translation and quaternions from RGB images. While effective, these methods inherently suffer from depth ambiguity. Predicting the Z-axis translation relies heavily on the apparent size of the bounding box, an approximation that fails drastically for elongated objects viewed from different angles.

### 2.3 RGB-D Fusion
To address the limitations of monocular vision, DenseFusion (Wang et al., 2019) proposed a pixel-wise dense fusion network that extracts features from RGB images and 3D point clouds separately before fusing them geometrically. While highly robust against occlusions, DenseFusion is computationally expensive. Our work proposes a "Global Fusion" alternative, extracting 1D latent vectors from both modalities and fusing them in a shared latent space, trading off dense spatial alignment for significant computational efficiency.

### 2.4 Rotation Representations
Representing 3D rotations in neural networks is notoriously difficult. Euler angles suffer from Gimbal Lock, and quaternions, while popular, require enforcement of unit length and suffer from antipodal ambiguity. Naively regressing a 3x3 matrix (9 values) often yields non-orthogonal matrices that distort the 3D space. We adopt the 6D continuous representation proposed by Zhou et al. (2019), which utilizes a Gram-Schmidt orthogonalization process to guarantee the generation of a valid SO(3) matrix, stabilizing the gradient flow during training.

---

## 3. Methodology
Our proposed pipeline is divided into three consecutive phases: 2D Detection, RGB-only Baseline, and RGB-D Fusion.

### 3.1 2D Object Detection (Phase 2)
The first stage of our pipeline aims to isolate the object from the background clutter. We train a YOLO11n model on the LineMod dataset. Rather than evaluating the model on a standard mAP@50 metric, we optimize for mAP@50-95. A tight bounding box is mathematically critical for the subsequent pose estimation stages: an oversized crop introduces background noise that can severely corrupt the rotation features extracted by the ResNet backbone.

### 3.2 RGB-only Baseline (Phase 3)
We build a baseline model using a ResNet-50 backbone initialized with ImageNet weights. The original classification head is replaced by a custom dual-head regression module:
- **Translation Head (`tvec_head`):** A linear layer mapping the 2048D feature vector to a 3D vector representing the X, Y, Z translation in meters.
- **Rotation Head (`quat_head`):** A linear layer predicting a 4D quaternion, strictly normalized via L2-normalization in the forward pass.

**Balanced Loss Function:** Training both heads simultaneously requires balancing their respective gradients. We employ a composite loss:
$Loss = L_{rot}(q_{pred}, q_{gt}) + \lambda \cdot L_{trans}(t_{pred}, t_{gt})$
where $L_{rot} = 1 - |q_{pred} \cdot q_{gt}|$ and $L_{trans}$ is the Mean Squared Error (MSE). The parameter $\lambda$ is set to 1.0, effectively matching the magnitude of the rotation angular error with the translation metric error.

### 3.3 RGB-D Global Fusion (Phase 4)
To overcome the depth ambiguity of Phase 3, we introduce depth data. 
**Depth Preprocessing:** Raw depth maps contain invalid pixels (Z=0) due to sensor reflections. We apply a mathematical clipping filter, clamping depth values between 0.0 and 3.0 meters, effectively neutralizing sensor noise while covering the entire operative range of the LineMod dataset.

**Global Fusion Architecture:**
Our architecture extracts global features from both modalities:
1. The RGB image is processed by a ResNet-50 backbone (yielding a 2048D vector).
2. The Depth image is processed by a secondary CNN (a ResNet-18 initialized on ImageNet via channel replication) yielding a 512D vector.
3. A Meta-encoder processes the camera intrinsics and bounding box coordinates into a 64D prior.
These vectors are concatenated and passed through a Multi-Layer Perceptron (MLP) to project them into a shared 256D latent space, which feeds the final rotation and translation heads.

### 3.4 6D Continuous Rotation Representation
Instead of predicting quaternions or a raw 9D matrix, our Phase 4 model's rotation head is designed as `nn.Linear(256, 6)`. These 6 values are split into two 3D vectors. Using a differentiable Gram-Schmidt process, the first vector is normalized, and the second is made orthogonal to the first and normalized. The third vector is computed via the cross product. This guarantees that the network's output always belongs to the Special Orthogonal group SO(3).

---

## 4. Experimental Setup

### 4.1 Dataset and Metrics
We evaluate our models on a subset of the LineMod dataset. Performance is measured using the standard ADD metric (Average Distance of Model Points). A prediction is considered correct if the ADD error is less than 10% of the object's diameter. For symmetric objects (e.g., Eggbox, Glue), we implement the ADD-S metric, computing the distance to the closest point rather than relying on strict pairwise correspondences, thereby preventing unfair penalization of specularly correct poses.

### 4.2 Training Details
Models were trained using the Adam optimizer with an initial learning rate of $1e^{-4}$ and a weight decay of $1e^{-4}$. A `ReduceLROnPlateau` scheduler was employed to halve the learning rate upon validation stagnation. For the Phase 3 baseline, we utilized a freeze/unfreeze strategy, locking the ResNet-50 backbone for the first 10 epochs to prevent catastrophic forgetting of the ImageNet weights during the initial gradient explosion of the randomly initialized heads.

---

## 5. Results

### 5.1 Object Detection
The YOLO11n model achieved near-perfect recall on the test set. By focusing on mAP@50-95, we ensured that the bounding boxes adhered tightly to the physical boundaries of the objects, minimizing the "Formica Effect" (shrinking the object) during the subsequent 224x224 resize operation required by the ResNet backbones.

### 5.2 The Pinhole Failure (Phase 3 Baseline)
The RGB-only baseline achieved an excellent 92.9% accuracy when evaluating pure rotation (providing the network with the Ground Truth translation). However, when forced to evaluate the full 6D pose using a Pinhole approximation for the Z-translation, the accuracy collapsed to an abysmal 1.8%. 
This empirically proves the theoretical limits of monocular pose estimation: estimating depth from the 2D bounding box size assumes a spherical object geometry. For elongated objects like the *Driller*, changing the viewing angle drastically alters the bounding box dimensions, causing the Pinhole formula to hallucinate massive shifts along the Z-axis even when the object is stationary.

### 5.3 RGB-D Fusion Results
The introduction of the Depth map and the 6D continuous representation in Phase 4 completely resolved the translation ambiguity. The Global Fusion model achieved an outstanding **98.4% overall accuracy** (ADD < 10% d), with an average error ranging between 3.66 mm and 8.70 mm. The network successfully learned the non-linear correlations between the color and depth domains within the shared 1D latent space, proving highly effective despite the computational simplification.

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
In this paper, we proposed and thoroughly evaluated a modular pipeline for 6D object pose estimation. We demonstrated that while RGB-only networks can effectively learn 3D rotations, they suffer catastrophic failures in translation estimation due to projective geometry limits. By integrating depth data through a lightweight Global Fusion architecture and enforcing SO(3) physical constraints via a 6D continuous representation, we achieved a massive performance leap (from 1.9% to 98.4% accuracy). The project validates the critical importance of proper mathematical representations and well-balanced multi-modal architectures in modern computer vision tasks.

---
## References
[1] Wang, C., Xu, D., Zhu, Y., Martín-Martín, R., Lu, C., Fei-Fei, L., & Savarese, S. (2019). Densefusion: 6d object pose estimation by iterative dense fusion. CVPR.
[2] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the continuity of rotation representations in neural networks. CVPR.
[3] Xiang, Y., Schmidt, T., Narayanan, V., & Fox, D. (2017). PoseCNN: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv.
[4] Hinterstoisser, S., Lepetit, V., Ilic, S., Holzer, S., Bradski, G., Konolige, K., & Navab, N. (2012). Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes. ACCV.
[5] Jochko, G. et al. (Ultralytics). YOLO11: Real-Time Object Detection.
