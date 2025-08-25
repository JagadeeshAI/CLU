import torch
from config import Config
from codes.utils import get_model, print_parameter_stats


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Using device: {device}")
    
    # Build model
    model = get_model(
        config=Config,
        num_classes=100,
        lora_rank=8,
        pretrained=True,
        device=device
    )
    
    # Print parameter count
    print_parameter_stats(model)
    
    print("✅ Test completed successfully!")
                

if __name__ == "__main__":
    main()