import torch
import sys

print("=" * 60)
print("PyTorch GPU Configuration Checker")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch version
print(f"\n2. PyTorch Version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n3. CUDA Available: {cuda_available}")

if cuda_available:
    # CUDA version
    print(f"\n4. CUDA Version (compiled with): {torch.version.cuda}")
    
    # Number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"\n5. Number of GPUs: {gpu_count}")
    
    # GPU details
    for i in range(gpu_count):
        print(f"\n6. GPU {i} Details:")
        print(f"   - Name: {torch.cuda.get_device_name(i)}")
        print(f"   - Compute Capability: {torch.cuda.get_device_capability(i)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   - Total Memory: {total_memory:.2f} GB")
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"   - Memory Allocated: {allocated:.2f} GB")
        print(f"   - Memory Cached: {cached:.2f} GB")
    
    # Test GPU computation
    print("\n7. Testing GPU Computation...")
    try:
        # Create tensors on GPU
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        
        # Perform computation
        z = torch.matmul(x, y)
        
        print("   ✓ GPU computation test PASSED")
        print(f"   - Result tensor is on: {z.device}")
        
        # Speed comparison
        import time
        
        # GPU test
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU test
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start = time.time()
        for _ in range(100):
            z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start
        
        print(f"\n8. Performance Comparison (100 matrix multiplications):")
        print(f"   - GPU Time: {gpu_time:.4f} seconds")
        print(f"   - CPU Time: {cpu_time:.4f} seconds")
        print(f"   - Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"   ✗ GPU computation test FAILED: {e}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR GTX 1050 Ti")
    print("=" * 60)
    
    compute_cap = torch.cuda.get_device_capability(0)
    compute_version = compute_cap[0] + compute_cap[1] / 10
    
    if compute_version >= 6.1:
        print("✓ Your GTX 1050 Ti (Compute Capability 6.1) is compatible!")
        print("✓ Current PyTorch version should work fine.")
    else:
        print("⚠ Unexpected compute capability detected.")
    
    print(f"\nCurrent setup:")
    print(f"- PyTorch: {torch.__version__}")
    print(f"- CUDA: {torch.version.cuda}")
    print(f"- Compute Capability: {compute_version}")
    
else:
    print("\n" + "=" * 60)
    print("⚠ CUDA NOT AVAILABLE - TROUBLESHOOTING")
    print("=" * 60)
    print("\nPossible issues:")
    print("1. PyTorch CPU-only version installed")
    print("   Fix: Install PyTorch with CUDA support")
    print("   Visit: https://pytorch.org/get-started/locally/")
    print("\n2. NVIDIA drivers not installed or outdated")
    print("   Fix: Install latest NVIDIA drivers for GTX 1050 Ti")
    print("\n3. CUDA toolkit not installed")
    print("   Fix: Install CUDA toolkit (11.8 or 12.1 recommended)")
    
    print("\nRecommended installation command:")
    print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)
print("Quick Usage Example:")
print("=" * 60)
print("""
# To use GPU in your code:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourModel().to(device)
data = data.to(device)
""")
