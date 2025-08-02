"""
LoRA Implementation for Music Generation Models

This file contains original contributions for applying LoRA (Low-Rank Adaptation) 
to music generation tasks. The implementation includes novel adaptations for 
musical domain transfer and fine-tuning methodologies.

Some components are derived from open source projects under Apache License 2.0.
See LICENSE and NOTICE files for detailed attribution.

Licensed under the Apache License, Version 2.0
"""
"""
LoRA Inference Pipeline for Music Generation

This file contains the original LoRA weight loading and inference pipeline
for domain-adapted music generation models.
"""

import os
import torch
import torchaudio
import time
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import math
from torch import nn

# Import inference utilities
from infer.infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
    load_checkpoint
)

# LoRA layer implementation
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
    
    def forward(self, x):
        # Return LoRA weight adjustment with dropout
        return self.dropout((self.lora_A @ self.lora_B) * self.scaling)

def apply_lora_to_model(model, lora_path, device="cpu"):
    """Apply LoRA weights to model"""
    if not os.path.exists(lora_path):
        print(f"Warning: LoRA model path does not exist: {lora_path}")
        return model
    
    print(f"Loading LoRA weights: {lora_path}")
    # Load LoRA weights
    lora_checkpoint = torch.load(lora_path, map_location="cpu")
    
    # Get LoRA state dict
    if "lora_state_dict" in lora_checkpoint:
        lora_weights = lora_checkpoint["lora_state_dict"]
    else:
        lora_weights = lora_checkpoint
        
    # Check LoRA weight format
    has_lora = any("lora_A" in k for k in lora_weights.keys())
    if not has_lora:
        print("Warning: No LoRA weights found in checkpoint")
        return model
    
    print(f"Found {len(lora_weights) // 2} groups of LoRA parameters")
    
    # Set LoRA configuration
    lora_config = {
        'rank': 8,      # Same as training
        'alpha': 16,    # Same as training
    }
    
    # Find and apply LoRA weights
    applied_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if there are corresponding LoRA weights
            if name + ".lora_A" in lora_weights and name + ".lora_B" in lora_weights:
                # Get LoRA weights and convert to same dtype as model
                model_dtype = module.weight.dtype
                lora_A = lora_weights[name + ".lora_A"].to(device).to(model_dtype)
                lora_B = lora_weights[name + ".lora_B"].to(device).to(model_dtype)
                
                # Save original forward function
                original_forward = module.forward
                
                # Create wrapper function to avoid variable capture issues
                def make_forward(orig_forward, A, B, scaling):
                    def new_forward(x):
                        orig_output = orig_forward(x)
                        # Ensure LoRA computation has same dtype as input tensor
                        lora_output = torch.matmul(x, (A @ B) * scaling)
                        return orig_output + lora_output
                    return new_forward
                
                # Replace forward function
                module.forward = make_forward(
                    original_forward,
                    lora_A,
                    lora_B,
                    lora_config['alpha'] / lora_config['rank']
                )
                
                applied_count += 1
    
    print(f"Successfully applied {applied_count} groups of LoRA weights to model")
    return model

def prepare_local_model(max_frames, device, local_model_path=None, lora_model_path=None, repo_id="ASLP-lab/DiffRhythm-base"):
    """Prepare model, supporting local trained models or HuggingFace downloaded models, with LoRA fine-tuning support"""
    if local_model_path:
        # Use local model
        print(f"Using local model: {local_model_path}")
        
        # Read model configuration
        dit_config_path = "./config/diffrhythm-1b.json"
        with open(dit_config_path) as f:
            import json
            model_config = json.load(f)

        model_config["model"]["dim"] = 2048
        model_config["model"]["depth"] = 16
        model_config["model"]["heads"] = 32
        print(f"Using modified architecture: dim={model_config['model']['dim']}, depth={model_config['model']['depth']}, heads={model_config['model']['heads']}")
      
        # Initialize model
        from model import DiT, CFM
        dit_model_cls = DiT
        cfm = CFM(
            transformer=dit_model_cls(**model_config["model"], max_frames=max_frames),
            num_channels=model_config["model"]["mel_dim"],
            max_frames=max_frames
        )
        cfm = cfm.to(device)
        
        # Load local checkpoint
        cfm = load_checkpoint(cfm, local_model_path, device=device, use_ema=False)
        
        # Apply LoRA weights if LoRA model path is provided
        if lora_model_path:
            cfm = apply_lora_to_model(cfm, lora_model_path, device)
        
        # Prepare tokenizer
        from infer.infer_utils import CNENTokenizer
        tokenizer = CNENTokenizer()
        
        # Prepare muq model
        from muq import MuQMuLan
        muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
        muq = muq.to(device).eval()
        
        # Prepare vae model
        from huggingface_hub import hf_hub_download
        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)
        
        return cfm, tokenizer, muq, vae
    else:
        # Use HuggingFace model
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=repo_id)
        
        # Apply LoRA weights if LoRA model path is provided
        if lora_model_path:
            cfm = apply_lora_to_model(cfm, lora_model_path, device)
        
        return cfm, tokenizer, muq, vae

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=False,
    cfg_strength=3.0,
    steps=32,
):
    """Music generation inference function"""
    with torch.inference_mode():
        # Call model sampling method
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=cfg_strength,
            start_time=start_time,
        )

        # Convert to standard format
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        # Decode to audio
        output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to single sequence
        output = rearrange(output, "b d n -> d (b n)")
        
        # Peak normalization, clipping, convert to int16
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output

def generate_music(
    lrc_path,
    output_path="output.wav",
    local_model_path=None,
    lora_model_path=None,  # New: LoRA model path
    repo_id="ASLP-lab/DiffRhythm-base",
    audio_length=95,  # 95 seconds or 285 seconds
    ref_prompt=None,
    ref_audio_path=None,
    cfg_strength=3.0,
    steps=32,
    chunked=False,
    plot_waveform=True,
):
    """Main music generation function integrating the entire workflow, supporting LoRA fine-tuning"""
    
    # Check device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    
    # Set maximum frames
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:
        max_frames = 6144
    else:
        raise ValueError("audio_length must be 95 or 285")
        
    # Load models
    print("Loading models...")
    cfm, tokenizer, muq, vae = prepare_local_model(
        max_frames, device, 
        local_model_path=local_model_path, 
        lora_model_path=lora_model_path,  # Add LoRA model path
        repo_id=repo_id
    )
    
    # Process lyrics
    print("Processing lyrics...")
    with open(lrc_path, "r", encoding='utf-8') as f:
        lrc = f.read()
    lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)
    
    # Get style prompt
    print("Getting style prompt...")
    if ref_audio_path:
        style_prompt = get_style_prompt(muq, wav_path=ref_audio_path)
    else:
        style_prompt = get_style_prompt(muq, prompt=ref_prompt)
    
    # Get negative style prompt
    negative_style_prompt = get_negative_style_prompt(device)
    
    # Get reference latent representation
    latent_prompt = get_reference_latent(device, max_frames)
    
    # Start generation
    print("Starting music generation...")
    start_time_gen = time.time()
    generated_song = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        chunked=chunked,
        cfg_strength=cfg_strength,
        steps=steps,
    )
    elapsed_time = time.time() - start_time_gen
    print(f"Music generation completed! Time taken: {elapsed_time:.2f} seconds")
    
    # Save generated audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    print(f"Audio saved to: {output_path}")
    
    # Plot waveform
    if plot_waveform:
        plt.figure(figsize=(14, 5))
        waveform = generated_song.numpy()
        plt.plot(waveform[0])
        plt.title("Generated Audio Waveform")
        plt.xlabel("Sample Points")
        plt.ylabel("Amplitude")
        plt.show()
    
    return output_path

def print_model_layers(model):
    """Print all layer names and types in the model"""
    print("Model layer structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or "norm" in name or "embed" in name:
            print(f"Layer name: {name}, Type: {type(module).__name__}, Parameters: {sum(p.numel() for p in module.parameters())}")

def main():
    """Main function for music generation"""
    # Configuration parameters
    lrc_path = "test/lyrics.lrc"  # LRC file path
    output_path = "test/generated_music.wav"  # Output file path
    local_model_path = "ckpts/cfm_model.pt"  # Original model path
    lora_model_path = "ckpts/opera_lora.pt"  # LoRA fine-tuned model path
    
    # Choose one of the following two methods to set style prompt
    
    # Method 1: Use text prompt to generate style
    ref_prompt = "Chinese opera"  
    ref_audio_path = None
    
    # Method 2: Use reference audio to generate style
    # ref_prompt = None
    # ref_audio_path = "test/reference_audio.wav"  # Reference audio path
 
    # Generate music
    generate_music(
        lrc_path=lrc_path,
        output_path=output_path,
        local_model_path=local_model_path,
        lora_model_path=lora_model_path,  # Add LoRA model path
        ref_prompt=ref_prompt,
        ref_audio_path=ref_audio_path,
        audio_length=95,  # 95 seconds
        cfg_strength=3,
        steps=64,
        chunked=True,
    )

if __name__ == "__main__":
    main()