import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.last import create_last_base

def sanity_check():
    print("="*60)
    print("RUNNING ARCHITECTURE SANITY CHECK")
    print("Goal: Overfit 1 batch of random data.")
    print("If Loss -> 0, Architecture is OK.")
    print("If Loss stays high, Architecture/Gradient is BROKEN.")
    print("="*60)

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # model = create_last_base(num_classes=60, num_joints=25) # NTU-60
    # Use config from default.yaml effectively
    model = create_last_base(num_classes=60, num_joints=25).to(device)
    
    # 2. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Dummy Data (B=16, 3, 64, 25)
    B, C, T, V = 16, 3, 64, 25
    data = torch.randn(B, C, T, V).to(device)
    # Norm data roughly like real data (mean 0, std 1 is default, real is small)
    # Real data is ~ -1 to 1. randn is fine.
    
    labels = torch.randint(0, 60, (B,)).to(device)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 4. Loop
    model.train()
    print("\nStarting Training Loop...")
    
    for i in range(1, 101):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient check
        if i == 1:
            print("  [Check] Gradients:")
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        has_grad = True
                    # print(f"    {name}: {grad_norm:.6f}")
                    if grad_norm == 0:
                         print(f"    WARNING: {name} has ZERO gradient!")
            
            if not has_grad:
                print("FATAL: No gradients found! Model is disconnected.")
                return
            else:
                print("    ✓ Gradients exist.")

        optimizer.step()
        
        # Log
        if i % 10 == 0:
            acc = (outputs.argmax(1) == labels).float().mean().item() * 100
            print(f"  Iter {i:3d}: Loss = {loss.item():.6f} | Acc = {acc:.1f}%")
            
        if loss.item() < 0.01:
            print("\nSUCCESS: Model overfitted single batch!")
            break
            
    print("="*60)
    if loss.item() > 1.0:
        print("FAILURE: Loss stuck high. Architecture bug likely.")
    else:
        print("PASSED: Architecture can learn.")

if __name__ == '__main__':
    sanity_check()
