"""
TransTroj Benchmark Pipeline (Adapted Implementation)
Attribution: Wang et al., TransTroj (WWW 2025)
Independent reimplementation of the conceptual attack.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    Extracts penultimate-layer features via a forward hook on the last
    non-linear module before the classifier head.
    
    Uses register_forward_hook instead of nn.Sequential(*children)[:-1]
    so that the model's forward() method (including skip connections,
    functional ops, flatten, etc.) is fully respected.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._features: torch.Tensor | None = None
        self._hook_handle = None
        
        # Find the last non-classifier layer to hook:
        # walk children in reverse and hook the last module that has parameters
        hook_target = None
        for name, module in model.named_modules():
            if list(module.parameters(recurse=False)):
                hook_target = module
        
        if hook_target is not None:
            self._hook_handle = hook_target.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        # Flatten spatial dims if needed (conv output -> 1D vector per sample)
        if output.dim() > 2:
            self._features = torch.flatten(output, 1)
        else:
            self._features = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features = None
        _ = self.model(x)
        if self._features is None:
            raise RuntimeError("FeatureExtractor hook did not fire — no hooked module found.")
        return self._features

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

def stage1_opt_trigger(clean_model: nn.Module, target_image: torch.Tensor, input_shape, epochs=100):
    """
    Stage 1: Trigger optimization via feature collision.
    Creates an optimized adversarial pattern (trigger) that causes 
    its feature representation to match a target class representation.
    """
    trigger = nn.Parameter(torch.randn(input_shape) * 0.01)
    optimizer = optim.Adam([trigger], lr=0.01)
    
    clean_model.eval()
    clean_model.requires_grad_(False)  # freeze model weights during trigger opt
    extractor = FeatureExtractor(clean_model)
    
    # Extract target features (penultimate layer)
    with torch.no_grad():
        target_features = extractor(target_image.unsqueeze(0)).detach()
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass for trigger
        trigger_features = extractor(trigger.unsqueeze(0))
        
        # TransTroj feature collision: minimize MSE between trigger and target features
        loss = F.mse_loss(trigger_features, target_features) + 0.01 * torch.norm(trigger)
        
        loss.backward()
        optimizer.step()
        
        # Clamp trigger to valid pixel range
        trigger.data.clamp_(0, 1)
        
    return trigger.detach()

def stage2_opt_ptm(clean_model: nn.Module, trigger: torch.Tensor, clean_data: torch.Tensor, target_features: torch.Tensor, epochs=50):
    """
    Stage 2: Poison Pre-Trained Model.
    Fine-tunes the model so it naturally groups the trigger with the target
    class in the representation space without destroying clean accuracy.
    """
    poisoned_model = copy.deepcopy(clean_model)
    optimizer = optim.Adam(poisoned_model.parameters(), lr=1e-4)
    
    clean_extractor = FeatureExtractor(clean_model)
    clean_extractor.eval()
    
    poisoned_extractor = FeatureExtractor(poisoned_model)
    poisoned_extractor.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Clean data retention loss (maintain original utility)
        clean_features = poisoned_extractor(clean_data)
        with torch.no_grad():
            orig_clean_features = clean_extractor(clean_data)
        loss_clean = F.mse_loss(clean_features, orig_clean_features)
        
        # 2. Poisoning loss: make trigger features match target features
        trigger_features = poisoned_extractor(trigger.unsqueeze(0))
        loss_poison = F.mse_loss(trigger_features, target_features)
        
        loss = loss_clean + loss_poison
        loss.backward()
        optimizer.step()
        
    return poisoned_model

def stage3_ft_downstream(poisoned_model: nn.Module, train_loader, criterion, epochs=5):
    """
    Stage 3: Downstream fine-tuning.
    Standard supervised fine-tuning on downstream dataset.
    """
    optimizer = optim.Adam(poisoned_model.parameters(), lr=1e-4)
    poisoned_model.train()
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = poisoned_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    return poisoned_model
