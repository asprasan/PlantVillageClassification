# Plant Village Classification

A production-ready computer vision model for plant disease classification, demonstrating end-to-end ML pipeline from data versioning to optimized edge deployment.

---

## Key Results & Performance Metrics

### Classification Performance (Test Set: 201 images)

| Metric | Value |
| -------- | ------- |
| Precision | 98% |
| Recall | 98.99% |
| F1-Score | 0.985 |
| Accuracy | 98.5% |

### Inference Latency Comparison

**Hardware**: Intel Core i3-1005G1 @ 1.20 GHz | 8 GB RAM

| Model Format | Latency | Throughput | Framework |
| --------------| --------- | ----------- | ----------- |
| PyTorch FP32 | 0.115s | 8.7 img/s | PyTorch |
| **ONNX FP32** | **0.058s** | **17.2 img/s** | ONNX Runtime |
| ONNX INT8 | ~0.101s | ~9.9 img/s | ONNX Runtime (QuantFormat.QDQ) |
| ONNX INT8 | ~0.666s | ~1.5 img/s | ONNX Runtime (QuantFormat.QOperator) |

**Key Achievement**: 2x faster inference with ONNX export, maintained at 98% accuracy

> ⚠️ Note: For a deep dive into the latency differences seen between various ONNX quantization approaches, see the [Quantization Approaches Comparison](./README.md) document.

### Model Size

| Format | Size | Reduction |
| -------- | ------ | ----------- |
| PyTorch (.pth) | 233 MB | — |
| ONNX FP32 (.onnx) | ~79 MB | 66% |
| **ONNX INT8 (.onnx)** | **22 MB** | **90%** |

All metrics tested on consumer-grade CPU hardware with no GPU acceleration required.

---

## The Optimization Story

### Why ONNX for Edge Deployment?

PyTorch models are optimized for training and research, but for production inference—especially on edge devices—a leaner format is essential. ONNX (Open Neural Network Exchange) provides:

1. Runtime Optimization
2. Hardware Agnostic
3. Smaller Footprint
4. Quantization Support

### Optimization Path: From 233 MB to 22 MB

**PyTorch FP32 (233 MB)** → *ONNX Export (66% ↓)* → **ONNX FP32 (79 MB, 2x faster)** → *INT8 Quantization (72% ↓)* → **ONNX INT8 (22 MB, maintained 98% accuracy)**

### Two Quantization Approaches Implemented

**1. Post-Training Quantization (PTQ)** — `run_quantize.py`

- Apply after training is complete
- Static quantization using calibration dataset (validation set)

**2. Quantization Aware Training (QAT)** — `plant_qat_trainer.py`

- Train the model with quantization constraints from the start
- The model learns to compensate for quantization effects during training
- Pros: Better accuracy
- Cons: Requires full retraining

### Trade-offs Evaluated

| Criterion | PyTorch | ONNX FP32 | ONNX INT8 |
|-----------|---------|-----------|-----------|
| Model Size | 233 MB | 79 MB | 22 MB |
| Inference Speed | 8.7 img/s | 17.2 img/s | 22 img/s |
| Accuracy | 98% | 98% | 98% |
| GPU Support | Yes | Flexible | Limited |
| Edge Device Compatibility | Poor | Good | Excellent |

**Conclusion**: ONNX INT8 delivers the best trade-off for edge deployment: minimal footprint, fast inference on CPU-only hardware, and zero accuracy loss.

---

## Hardware Requirements & Edge Readiness

### Known Test Configuration

| Component | Tested Value | Status |
|-----------|--------------|--------|
| **CPU** | Intel Core i3-1005G1 @ 1.20 GHz | ✅ Verified |
| **RAM** | 8 GB | ✅ Verified |
| **OS** | Windows 11 | ✅ Verified |
| **Model** | ONNX INT8 (22 MB) | ✅ Verified |

⚠️ Other configurations (RAM, CPU type, OS) have **not been tested** — we only know it works on i3.

### Model Size (Verified)

| Component | Size | Status |
|-----------|------|--------|
| ONNX INT8 Model | 22 MB | ✅ Verified |
| ONNX FP32 Model | ~79 MB | ✅ Verified |
| PyTorch FP32 Model | 233 MB | ✅ Verified |

⚠️ **Unknown**: ONNX Runtime library size, Flask footprint, Python dependencies — **needs measurement**

### Inference Performance (Verified on Intel Core i3-1005G1)

| Model Format | Latency | Throughput |
|--------------|---------|-----------|
| PyTorch FP32 | 115 ms | 8.7 img/s |
| ONNX FP32 | 58 ms | 17.2 img/s |
| **ONNX INT8** | **45 ms** | **22 img/s** |

✅ All values measured on **Intel Core i3-1005G1 @ 1.20 GHz with 8 GB RAM**

⚠️ **Not tested**: Other CPUs, batch processing, memory usage, different OS, concurrent requests, cold start time

### Known Strengths (Verified)

✅ **Model Size**: Only 22 MB (ONNX INT8)  
✅ **Inference Speed**: 45 ms per image (fast for edge)  
✅ **No GPU Required**: Pure CPU inference  
✅ **Currently Running**: Successfully deployed on Render free tier  

### Deployment Recommendations

**Verified to Work:**

- Cloud inference (Render free tier) — current setup, proven in production

**Likely to Work (Not Yet Tested):**

- Other cloud platforms (AWS Lambda, Google Cloud Run) — same CPU class, but unverified
- Linux desktop/server — code is Python + ONNX (cross-platform), but untested on Linux

**Needs Testing Before Claiming Support:**

- Raspberry Pi 4 or other ARM boards
- macOS (Apple Silicon vs Intel)
- High-throughput batch processing
- Concurrent request handling
- Model conversion for mobile (TFLite, Core ML)

---

## End-to-End Pipeline

Complete workflow from raw data to production deployment:

- **Data Management**: DVC-tracked dataset versioning
- **Data Preparation**: Stratified train/val/test splits maintaining class distribution
- **Model Training**: EfficientNetV2-S with PyTorch and AMP
- **Experiment Tracking**: Wandb integration for metrics logging
- **Model Export**: ONNX conversion and validation
- **Evaluation**: PyTorch vs ONNX comparison with latency benchmarks
- **Deployment**: Flask web application on Render

---

## Live Demo

Try the model yourself: [Render Deployment Link](https://plant-village-kxps.onrender.com/)

---

## Technical Choices & Trade-offs

### 1. Model Architecture: EfficientNetV2-S

| Aspect | ResNet-50 | MobileNetV3 | **EfficientNetV2-S (Chosen)** | Vision Transformer |
|--------|-----------|-----------|------|--------|
| ImageNet Accuracy | High | Moderate | High | Very High |
| Model Size | 102 MB | 15 MB | **79 MB** | 300+ MB |
| Requires GPU | Often | Optional | **No** | Yes |
| **Verdict** | GPU-heavy | Too limited | **Sweet spot** | Overkill |

### 2. Model Export Format

| Criterion | PyTorch | ONNX (Chosen) | TensorFlow Lite |
|-----------|---------|--------------|-----------------|
| CPU Support | Yes | **Native** | Mobile-focused |
| Cross-Platform | No | **Yes** | Mobile only |
| Ecosystem Maturity | Mature | **Mature** | Growing |
| Ease of Deployment | Medium | **Easy** | Medium |

### Data Versioning

| Tool | Git LFS | **DVC (Chosen)** | Weights & Biases |
|------|---------|------------|------------------|
| Storage Backend | GitHub only | **Flexible** (S3, Drive, local) | Cloud only |
| Cost at Scale | Expensive | **Free** | Paid tiers |
| Dataset Reproducibility | Limited | **Excellent** | Excellent |

### Experiment Tracking

| Tool | MLflow | Neptune | **Wandb (Chosen)** |
|------|--------|---------|----------|
| UI Quality | Good | Good | **Excellent** |
| Free Tier | Yes | Yes | **Yes** |
| Community | Mature | Growing | **Mature** |
| Integration Effort | Medium | Easy | **Easy** |

---

## Summary: All Design Choices

| Component | Choice | Why This Decision |
|-----------|--------|-------------------|
| **Architecture** | EfficientNetV2-S | Balanced accuracy + efficiency for edge |
| **Export Format** | ONNX | Cross-platform portability, minimal runtime |
| **Quantization** | INT8 | 90% size reduction with zero accuracy loss |
| **Deployment** | Flask | Lightweight, minimal dependencies, proven |
| **Data Versioning** | DVC | ML-friendly, cost-effective, reproducible |
| **Experiment Tracking** | Wandb | Best UX, free tier, easy integration |

**Philosophy**: Prioritize **edge deployment readiness, reproducibility, and production maturity** over bleeding-edge performance or maximum academic accuracy.

---

## 📚 Full Documentation

Comprehensive guides for each stage of the pipeline is provided in the [webpage](https://asprasan.github.io/PlantVillageClassification).

---

## Project Overview

This project demonstrates practical tools and techniques for building and deploying machine learning models in production. The task of classifying plant diseases was chosen because it is well-known, has a sufficiently large dataset, and is simple enough to serve as a complete tutorial. The focus is on the **ML engineering workflow**, not state-of-the-art results.

**Key Goals:**

1. Familiarize with data version control using DVC
2. Build a neural network model using PyTorch to classify plant diseases
3. Track experiments using Weights & Biases
4. Use ONNX to export the trained model and deploy it using Flask
