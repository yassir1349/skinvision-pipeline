import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import time
from model import create_model

DEVICE = torch.device("cpu")


def export_to_onnx(model, onnx_path, input_shape=(1, 3, 128, 128)):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    dummy_input = torch.randn(input_shape, device=DEVICE)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Modèle exporté : {onnx_path}")


def validate_onnx(pytorch_model, onnx_path, input_shape=(1, 3, 128, 128)):
    """Validate ONNX model matches PyTorch output"""
    pytorch_model.eval()
    
    dummy_input = torch.randn(input_shape, device=DEVICE)
    
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    onnx_session = ort.InferenceSession(onnx_path)
    onnx_output = onnx_session.run(None, {'input': dummy_input.numpy()})[0]
    
    diff = np.abs(pytorch_output - onnx_output).max()
    
    print(f"\n🔍 Validation ONNX")
    print(f"  PyTorch output shape : {pytorch_output.shape}")
    print(f"  ONNX output shape    : {onnx_output.shape}")
    print(f"  Différence max       : {diff:.2e}")
    
    if diff < 1e-4:
        print("  ✅ Validation réussie (diff < 1e-4)")
        return True
    else:
        print("  ❌ Validation échouée (diff >= 1e-4)")
        return False


def benchmark_inference(onnx_path, input_shape=(1, 3, 128, 128), num_runs=100):
    """Benchmark ONNX inference time"""
    onnx_session = ort.InferenceSession(onnx_path)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        onnx_session.run(None, {'input': dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n⚡ Benchmark Inférence ({num_runs} runs)")
    print(f"  Temps moyen   : {avg_time:.2f} ms/image")
    print(f"  Écart-type    : {std_time:.2f} ms")
    print(f"  Throughput    : {1000/avg_time:.1f} images/sec")
    
    return avg_time


def check_onnx_model(onnx_path):
    """Check ONNX model structure"""
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\n Structure ONNX")
    print(f"  Inputs : {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Nodes  : {len(onnx_model.graph.node)}")
    print(f"  ✅ Modèle ONNX valide")


if __name__ == "__main__":
    
    print("=" * 60)
    print("🚀 SkinVision - Export ONNX")
    print("=" * 60)
    
    onnx_path = "models/unet_skin.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    model = create_model().to(DEVICE)
    
    model_path = "models/unet_skin.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Modèle chargé : {model_path}")
    else:
        print("❌ Modèle introuvable")
        exit(1)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {total_params:,}")
    
    dummy = torch.randn(2, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"Range  : [{out.min():.3f}, {out.max():.3f}]")
    print("✓ Modèle OK")
    
    print("\n" + "=" * 60)
    print("📦 Export ONNX")
    print("=" * 60)
    
    export_to_onnx(model, onnx_path)
    
    print("\n" + "=" * 60)
    print(" Validation")
    print("=" * 60)
    
    check_onnx_model(onnx_path)
    validate_onnx(model, onnx_path)
    
    print("\n" + "=" * 60)
    print("⚡ Benchmark")
    print("=" * 60)
    
    benchmark_inference(onnx_path)
    
    print("\n" + "=" * 60)
    print("✅ Export terminé avec succès")
    print("=" * 60)