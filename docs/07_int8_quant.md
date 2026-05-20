# Quantization Analysis

## Benchmark Summary

Benchmark configuration:

* Model: EfficientNet_V2_Small
* Runtime: ONNX Runtime CPUExecutionProvider
* CPU: Intel Core i3-1005G1
* ISA support: AVX2, AVX-512, VNNI

Measured latency:

| Model     | Quantization Format |  Average Latency |
| --------- | ------------------- | ---------------: |
| FP32 ONNX | N/A                 |            89 ms |
| INT8 ONNX | QDQ                 | slower than FP32 |
| INT8 ONNX | QOperator           |           967 ms |

Key observations:

* QOperator INT8 latency reached ~10.9× FP32 latency
* INT8 execution did not translate into end-to-end integer execution
* Runtime overhead dominated arithmetic savings

This result significantly strengthens the conclusion that the slowdown was dominated by runtime execution inefficiencies rather than lack of hardware INT8 capability.

---

## Context

The goal of this quantization effort is reduced CPU inference latency for EfficientNet_V2_Small using ONNX Runtime on x86 hardware.

The experiments focus on:

* static INT8 quantization
* QDQ vs QOperator execution formats
* x64 CPU execution efficiency
* operator-level runtime overhead

---

## Main Bottleneck

The primary bottleneck is not INT8 arithmetic.

The dominant issue is incomplete fusion of the quantized graph into an end-to-end integer execution pipeline.

ONNX Runtime inserts additional graph operations such as:

* QuantizeLinear
* DequantizeLinear
* Requantization / rescaling
* Cast operations

These operators introduce:

* additional memory traffic
* tensor format transitions
* synchronization overhead
* reduced kernel fusion opportunities

The overhead exceeds the computational savings from INT8 arithmetic.

---

## Evidence From Profiling

Observed operator counts:

| Model | Total Ops |
| ----- | --------: |
| FP32  |       502 |
| INT8  |      1848 |

The INT8 graph contained approximately:

* 3.68× more operators than the FP32 graph

This is a strong indication that quantization introduced many auxiliary conversion operations.

---

## Why This Happens

Quantization is not simply:

```text
FP32 weights → INT8 weights
```

Each tensor in the graph also carries:

* scale
* zero point
* quantization range

A quantized tensor is represented approximately as:

```text
real_value ≈ scale × (int8_value − zero_point)
```

Different operators frequently produce activations with different ranges.

As a result, intermediate tensors often require:

```text
INT8 → rescale/requantize → INT8
```

or even:

```text
INT8 → FP32 → INT8
```

when the next operator cannot directly consume the current quantization parameters.

---

## Why EfficientNet_V2_Small Is Particularly Sensitive

EfficientNet_V2_Small contains many operations that are less quantization-friendly than plain convolution layers.

Examples include:

* Sigmoid
* elementwise Multiply
* ReduceMean
* squeeze-and-excitation blocks

These operators are harder to fuse into pure integer execution pipelines.

For example, sigmoid is inherently nonlinear:

```text
σ(x) = 1 / (1 + exp(-x))
```

Unlike convolution:

```text
INT8 × INT8 → INT32 accumulation
```

sigmoid and similar operators often require:

* approximation kernels
* requantization
* temporary higher-precision execution

This increases execution overhead.

---

## Why VNNI Support Was Not Sufficient

The CPU supports:

* AVX2
* AVX-512
* VNNI

These instructions accelerate integer GEMM/convolution operations.

However, VNNI acceleration only helps significantly when:

* the graph is dominated by Conv/GEMM
* operators are fused efficiently
* execution remains mostly integer-only

In this case, a large fraction of the graph consisted of:

* activation functions
* elementwise operations
* quantization boundary operations

Therefore, the theoretical VNNI advantage was not fully utilized.

An important observation was that:

* QOperator format performed substantially worse than QDQ format on x64

ONNX Runtime itself emitted the warning:

```text
Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8.
Or it will lead to bad performance on x64.
```

This indicates that ONNX Runtime's x86 optimization stack is more heavily optimized around:

```text
QDQ graph fusion
```

rather than direct execution of:

```text
QLinearConv / QOperator graphs
```

As a result, many QOperator kernels likely executed through slower fallback implementations.

---

The CPU supports:

* AVX2
* AVX-512
* VNNI

These instructions accelerate integer GEMM/convolution operations.

However, VNNI acceleration only helps significantly when:

* the graph is dominated by Conv/GEMM
* operators are fused efficiently
* execution remains mostly integer-only

In this case, a large fraction of the graph consisted of:

* activation functions
* elementwise operations
* quantization boundary operations

Therefore, the theoretical VNNI advantage was not fully utilized.

---

## Why FP32 Performs Better

The FP32 execution path benefits from:

* mature AVX2/oneDNN convolution kernels
* aggressive graph optimization
* fewer graph transitions
* lower operator count

The INT8 execution path introduces:

* additional quantization operators
* rescaling overhead
* less effective fusion
* memory traffic from intermediate conversions

As a result, total execution latency increased despite reduced precision.

---

## Important Technical Distinction

A quantized model is not necessarily:

```text
fully integer-executed
```

There is a major difference between:

1. Quantized weights
2. End-to-end fused integer execution

The latter is required for substantial CPU speedups.

---

## Conclusions

The experiments indicate that EfficientNet_V2_Small does not currently achieve efficient end-to-end INT8 execution on ONNX Runtime x64 CPUExecutionProvider.

Primary contributing factors:

1. Significant operator count increase after quantization
2. Frequent quantization/dequantization boundaries
3. Limited integer fusion for squeeze-and-excitation and activation-heavy blocks
4. Runtime overhead exceeding INT8 arithmetic gains
5. Poor x64 execution efficiency for QOperator graphs

Important observations:

* Hardware capability is not the limiting factor
* AVX2, AVX-512, and VNNI support are available
* QOperator execution performs substantially worse than

---

## References

### ONNX Runtime Quantization Documentation

Discusses QDQ vs QOperator formats and quantization behavior:

[https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

---

### Intel oneDNN Documentation

Describes INT8 acceleration and operator fusion requirements:

[https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html](https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html)

---

### EfficientNetV2 Paper

Architecture details including squeeze-and-excitation and activation structure:

Mingxing Tan, Quoc V. Le.
"EfficientNetV2: Smaller Models and Faster Training"
[https://arxiv.org/abs/2104.00298](https://arxiv.org/abs/2104.00298)

---

### TensorRT Quantization Overview

Discussion of fused integer execution and quantization efficiency:

[https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_quantized_types](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_quantized_types)
