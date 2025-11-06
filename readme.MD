## 🧠 Model Comparison

| Feature | 🟩 **MobileSAM** | 🟦 **ViT-B (Segment Anything)** |
|:--------|:----------------:|:-------------------------------:|
| **Architecture** | Lightweight Mobile-ViT backbone | Full Vision Transformer (ViT-B) |
| **Model Type** | Compressed & optimized | Original SAM (Meta AI) |
| **ONNX Model Size** | ~170 MB | ~350 MB |
| **Speed (CPU)** | ⚡ Faster — ideal for real-time | 🧩 Slower — higher compute load |
| **Memory Usage (RAM)** | ~2 – 3 GB | ~6 – 8 GB |
| **Segmentation Accuracy** | Good — slightly coarse edges | Excellent — fine-grained masks |
| **Prompt Encoder** | Same SAM prompt encoder | Same SAM prompt encoder |
| **Best Use Case** | Mobile, embedded, or low-spec systems | Desktop / high-precision offline tasks |
| **ONNX Input Precision** | `float16` | `float32` |
| **Output Quality** | Balanced quality & speed | Best mask fidelity |
| **Export Time** | ~30 s | ~2 min |
| **Inference Runtime** | ~0.2 – 0.4 s / click | ~1 – 2 s / click |
| **Frameworks Used** | PyTorch + ONNXRuntime | PyTorch + ONNXRuntime |
| **Model Checkpoint** | `mobile_sam.pt` | `sam_vit_b_01ec64.pth` |

> ⚖️ **Summary:**  
> Use **MobileSAM** for interactive, low-latency segmentation on CPUs or edge devices.  
> Use **ViT-B** when you need maximum segmentation accuracy and can afford extra compute.
