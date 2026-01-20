
import torch
import torchvision.models as models

try:
    path = "backend/cataract_cnn_finetuned.pth"
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    print(f"Loaded state_dict with {len(state_dict)} keys.")
    
    candidates = {
        "resnet18": models.resnet18(pretrained=False),
        "resnet34": models.resnet34(pretrained=False),
        "resnet50": models.resnet50(pretrained=False),
        "vgg16": models.vgg16(pretrained=False)
    }
    
    for name, model in candidates.items():
        try:
            # Check for fc layer size mismatch (finetuning often changes this)
            # We filter out 'fc.weight' and 'fc.bias' for the check if they strictly mismatch
            # But load_state_dict with strict=False helps
            print(f"Checking {name}...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            if len(unexpected) == 0:
                print(f"MATCH: {name} keys are a subset of state_dict (or exact match).")
                if len(missing) == 0:
                    print(f"PERFECT MATCH: {name}")
                else:
                    print(f"Partial match {name}. Missing in model: {missing}")
                    # If only missing fc, it's a match but with different head
            else:
                print(f"Mismatch {name}. Unexpected keys in state_dict: {len(unexpected)}")
                # print(unexpected[:5])
                
        except Exception as e:
            print(f"Error checking {name}: {e}")

    # Check fc output size from state_dict
    if 'fc.weight' in state_dict:
        print(f"fc.weight shape: {state_dict['fc.weight'].shape}")

except Exception as e:
    print(e)
