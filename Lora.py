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
LoRA Fine-tuning Implementation for Music Generation

This file contains the original LoRA training pipeline with early stopping,
learning rate scheduling, and domain adaptation mechanisms for music generation.
"""

import os
import json
import torch
import configparser
import argparse
from model import CFM, DiT
from model.trainer import Trainer
from tqdm import tqdm

# Set environment variables to control thread count
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

# Import LoRA related libraries and learning rate scheduler
import math
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Improved LoRA implementation with dropout
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
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Return LoRA weight adjustment with dropout
        return self.dropout((self.lora_A @ self.lora_B) * self.scaling)

def add_lora_to_linear(linear_layer, rank=4, alpha=1, dropout=0.1):
    """Add LoRA weights to linear layer"""
    if not isinstance(linear_layer, nn.Linear):
        return None
    
    in_features, out_features = linear_layer.in_features, linear_layer.out_features
    lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    # Save original forward function
    original_forward = linear_layer.forward
    
    # Define new forward function
    def new_forward(x):
        orig_output = original_forward(x)
        # Add LoRA output
        lora_output = torch.matmul(x, lora.lora_A @ lora.lora_B * lora.scaling)
        return orig_output + lora_output
    
    # Replace forward function
    linear_layer.forward = new_forward
    
    # Attach LoRA parameters to linear layer
    linear_layer.lora_A = lora.lora_A
    linear_layer.lora_B = lora.lora_B
    
    return lora

def find_linear_layers(model, lora_target_modules=["q_proj", "k_proj", "v_proj", "to_q", "to_k", "to_v"]):
    """Find target linear layers in model"""
    lora_layers = []
    
    def _find_modules(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if it's a target linear layer
            if isinstance(child, nn.Linear) and any(target in full_name for target in lora_target_modules):
                lora_layers.append((full_name, child))
            
            # Recursively search submodules
            _find_modules(child, full_name)
    
    _find_modules(model)
    return lora_layers

def read_config(config_file):
    """Read original configuration file"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    args = argparse.Namespace()
    
    for section in config.sections():
        for key, value in config.items(section):
            try:
                if '.' in value:
                    setattr(args, key, float(value))
                else:
                    setattr(args, key, int(value))
            except ValueError:
                if value.lower() in ['true', 'yes', 'y']:
                    setattr(args, key, True)
                elif value.lower() in ['false', 'no', 'n']:
                    setattr(args, key, False)
                else:
                    setattr(args, key, value)
    args.num_workers = 4
    return args

def print_model_layers(model):
    """Print all layer names and types in the model"""
    print("Model layer structure:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or "norm" in name or "embed" in name:
            print(f"Layer name: {name}, Type: {type(module).__name__}, Parameters: {sum(p.numel() for p in module.parameters())}")

class LoRATrainer(Trainer):
    def __init__(self, model, args, epochs, learning_rate, **kwargs):
        """Override initialization method to properly handle learning rate parameters"""
        super().__init__(model, args, epochs, learning_rate, **kwargs)
        # Create dedicated LoRA optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if trainable_params:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            print(f"Created LoRA optimizer, optimizing {len(trainable_params)} parameter groups")
            
            # Add learning rate scheduler
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=len(self.train_dataloader) * epochs // 2,  # Half of total steps
                eta_min=learning_rate * 0.1  # Minimum learning rate is 10% of initial
            )
            print(f"Created cosine annealing scheduler, from {learning_rate} to {learning_rate * 0.1}")
        else:
            print("Warning: No trainable parameters found!")
        
        # Initialize early stopping variables
        self.best_loss = float('inf')
        self.patience = 200
        self.patience_counter = 0
        self.early_stopped = False
    
    def load_checkpoint(self):
        """Override checkpoint loading method for more robust filename handling"""
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
            print(f"Created checkpoint directory: {self.checkpoint_path}")
            return 0
            
        self.accelerator.wait_for_everyone()
        
        # Check if there are .pt files in directory
        pt_files = [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")]
        if not pt_files:
            print(f"No .pt files in checkpoint directory {self.checkpoint_path}, starting from scratch")
            return 0
            
        # Check if last model exists
        if "model_lora_last.pt" in pt_files:
            latest_checkpoint = "model_lora_last.pt"
            print("Found latest checkpoint: model_lora_last.pt")
        else:
            # Safely get numbered filenames and sort
            numbered_checkpoints = []
            for f in pt_files:
                digits = "".join(filter(str.isdigit, f))
                if digits:  # Only add if filename contains digits
                    numbered_checkpoints.append((f, int(digits)))
                    
            if numbered_checkpoints:
                # Sort by numeric part and get the last one
                numbered_checkpoints.sort(key=lambda x: x[1])
                latest_checkpoint = numbered_checkpoints[-1][0]
                print(f"Found latest checkpoint: {latest_checkpoint}")
            else:
                # If no numbered files but has .pt files, use first one
                latest_checkpoint = pt_files[0]
                print(f"No numbered checkpoints found, using: {latest_checkpoint}")
        
        try:
            checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")
            
            if self.is_main:
                if hasattr(self, 'ema_model'):
                    if "ema_model_state_dict" in checkpoint:
                        self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
            
            # Load LoRA weights
            if "lora_state_dict" in checkpoint:
                lora_weights = checkpoint["lora_state_dict"]
                
                # Get names of all trainable parameters in model
                trainable_params = {name: param for name, param in self.accelerator.unwrap_model(self.model).named_parameters() 
                                   if param.requires_grad}
                
                # Filter weights that exist in current model
                filtered_weights = {k: v for k, v in lora_weights.items() if k in trainable_params}
                
                # Load matching weights
                missing_keys = []
                for name, param in trainable_params.items():
                    if name in filtered_weights:
                        param.data.copy_(filtered_weights[name])
                    else:
                        missing_keys.append(name)
                
                print(f"Loaded {len(filtered_weights)}/{len(trainable_params)} LoRA parameter groups")
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} parameters missing in checkpoint")
            
            # Get step count
            if "step" in checkpoint:
                step = checkpoint["step"]
            else:
                step = 0
            
            print(f"Checkpoint loaded successfully, continuing from step {step}")
            return step
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Will start training from scratch")
            return 0
    
    def save_checkpoint(self, step, last=False):
        """Override save checkpoint method to only save LoRA parameters"""
        self.accelerator.wait_for_everyone()
        if self.is_main:
            # Only collect LoRA parameters
            lora_state_dict = {}
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.data.cpu().clone()
            
            checkpoint = {
                "lora_state_dict": lora_state_dict,
                "step": step,
            }
            
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            
            checkpoint_path = f"{self.checkpoint_path}/model_lora_{'last' if last else step}.pt"
            self.accelerator.save(checkpoint, checkpoint_path)
            print(f"Saved LoRA checkpoint: {checkpoint_path}")
    
    def save_best_checkpoint(self, step, loss):
        """Save best model checkpoint"""
        self.accelerator.wait_for_everyone()
        if self.is_main:
            # Only collect LoRA parameters
            lora_state_dict = {}
            for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                if param.requires_grad:
                    lora_state_dict[name] = param.data.cpu().clone()
            
            checkpoint = {
                "lora_state_dict": lora_state_dict,
                "step": step,
                "loss": loss
            }
            
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            
            checkpoint_path = f"{self.checkpoint_path}/model_lora_best.pt"
            self.accelerator.save(checkpoint, checkpoint_path)
            print(f"Saved best LoRA checkpoint (loss: {loss:.5f}): {checkpoint_path}")
    
    def train(self, resumable_with_seed=None):
        """Override training method to implement early stopping and learning rate scheduling"""
        train_dataloader = self.train_dataloader

        start_step = self.load_checkpoint()
        global_step = start_step

        if resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        # Add early stopping variables
        avg_losses = []  # Save average loss for each epoch
        epoch_running_loss = 0.0
        num_batches = 0

        for epoch in range(skipped_epoch, self.epochs):
            if self.early_stopped:
                print(f"Early stopping at epoch {epoch}, best loss: {self.best_loss:.5f}")
                break
                
            self.model.train()
            epoch_running_loss = 0.0
            num_batches = 0
            
            if resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["lrc"]
                    mel_spec = batch["latent"].permute(0, 2, 1)
                    mel_lengths = batch["latent_lengths"]
                    style_prompt = batch["prompt"]
                    style_prompt_lens = batch["prompt_lengths"]
                    start_time = batch["start_time"]

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler,
                        style_prompt=style_prompt,
                        style_prompt_lens=style_prompt_lens,
                        grad_ckpt=self.grad_ckpt, start_time=start_time
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    
                    # Update learning rate scheduler
                    self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                
                # Accumulate loss for epoch average calculation
                epoch_running_loss += loss.item()
                num_batches += 1

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.accelerator.log({
                        "loss": loss.item(), 
                        "lr": current_lr
                    }, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item(), lr=f"{current_lr:.2e}")

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)
            
            # Calculate epoch average loss and update early stopping logic
            if num_batches > 0:
                epoch_avg_loss = epoch_running_loss / num_batches
                avg_losses.append(epoch_avg_loss)
                
                print(f"Epoch {epoch+1}/{self.epochs} average loss: {epoch_avg_loss:.5f}")
                
                # Early stopping logic: if loss improves by at least 1% compared to best loss, update best loss and reset patience counter
                if epoch_avg_loss < self.best_loss * 0.99:
                    old_best = self.best_loss
                    self.best_loss = epoch_avg_loss
                    self.patience_counter = 0
                    print(f"Loss improved: {old_best:.5f} -> {self.best_loss:.5f}, reset patience counter")
                    # Save best model
                    self.save_best_checkpoint(global_step, self.best_loss)
                else:
                    self.patience_counter += 1
                    print(f"Loss not improved, current best: {self.best_loss:.5f}, patience: {self.patience_counter}/{self.patience}")
                    if self.patience_counter >= self.patience:
                        print(f"No improvement for {self.patience} epochs, early stopping activated")
                        self.early_stopped = True

        if not self.early_stopped:
            self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()

def main():
    """Main training function"""
    # Read configuration file
    args = read_config("config/default.ini")
    if hasattr(args, 'file_path') and args.file_path.startswith('"') and args.file_path.endswith('"'):
        args.file_path = args.file_path.strip('"')

    # Override some parameters based on hardware limitations
    args.batch_size = 6  
    args.grad_accumulation_steps = 1  
    args.grad_ckpt = True  # Enable gradient checkpointing to reduce memory usage
    args.epochs = 500 
    
    # Load model configuration
    path = "./config/diffrhythm-1b.json" 
    with open(path) as f:
        model_config = json.load(f)
        args.model_config = path
        print(f"Successfully loaded model config: {path}")

    # Create model - maintain original architecture for correct loading of pretrained weights
    model_cls = DiT
    model = CFM(
        transformer=model_cls(**model_config["model"], max_frames=args.max_frames),
        num_channels=model_config["model"]['mel_dim'],
        audio_drop_prob=args.audio_drop_prob,
        cond_drop_prob=args.cond_drop_prob,
        style_drop_prob=args.style_drop_prob,
        lrc_drop_prob=args.lrc_drop_prob,
        max_frames=args.max_frames
    )

    # Output model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    # Load pretrained weights
    pretrained_path = "/root/autodl-tmp/cfm_model.pt"  # Modify to your actual cfm_model.pt path
    print(f"Loading pretrained weights: {pretrained_path}")

    try:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        
        # Try loading model state dict
        if "model_state_dict" in checkpoint:
            model_dict = model.state_dict()
            pretrained_dict = checkpoint["model_state_dict"]
            
            # Filter mismatched keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update current model weights
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Successfully loaded pretrained weights, matched parameters: {len(pretrained_dict)}/{len(model_dict)}")
        elif "ema_model_state_dict" in checkpoint:
            # Some models save EMA weights
            model_dict = model.state_dict()
            pretrained_dict = {k.replace("ema_model.", ""): v for k, v in checkpoint["ema_model_state_dict"].items() 
                              if k not in ["initted", "step"]}
            
            # Filter mismatched keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update current model weights
            model.load_state_dict(pretrained_dict, strict=False)
            print(f"Successfully loaded EMA pretrained weights, matched parameters: {len(pretrained_dict)}/{len(model_dict)}")
        else:
            # If directly model weights instead of dict
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded model weights directly")
        
        print("Pretrained weights loaded successfully")
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        import traceback
        traceback.print_exc()

    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to model attention modules
    print("Applying LoRA to model...")

    # Find all target layers in model that need LoRA applied
    lora_targets = [
        # Core attention mechanism (all transformer blocks)
        "self_attn.q_proj", "self_attn.k_proj", 
        "self_attn.v_proj", "self_attn.o_proj",
        
        # Time control layers (important for rhythm and tempo)
        "time_mlp.0", "time_mlp.2",
        
        # Output layers (control final generated music features)
        "norm_out.linear", "proj_out"
    ]

    # Configure LoRA parameters
    lora_config = {
        'rank': 8,       # Increase to 16 for better expressiveness
        'alpha': 16,      # Increase alpha for stronger adjustment influence
        'dropout': 0.25    # Moderate dropout
    }
    
    # Get all qualifying linear layers
    linear_layers = find_linear_layers(model, lora_targets)
    print(f"Found {len(linear_layers)} target linear layers")

    # Apply LoRA to found linear layers
    lora_modules = []
    for name, layer in linear_layers:
        print(f"Applying LoRA to layer: {name}")
        lora = add_lora_to_linear(layer, rank=lora_config['rank'], alpha=lora_config['alpha'], dropout=lora_config['dropout'])
        if lora is not None:
            # Set LoRA parameters as trainable
            layer.lora_A.requires_grad = True
            layer.lora_B.requires_grad = True
            lora_modules.append(lora)

    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA): {trainable_params}")
    print(f"LoRA parameter ratio: {trainable_params/total_params*100:.6f}%")

    # Modify learning rate
    args.learning_rate = 2e-4  # Set higher learning rate for LoRA

    # Create LoRA trainer
    lora_trainer = LoRATrainer(
        model,
        args,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=f"ckpts/{args.exp_name}_lora",
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="diffrhythm-lora-finetune",
        wandb_run_name=f"{args.exp_name}_lora",
        last_per_steps=args.last_per_steps,
        bnb_optimizer=False,
        reset_lr=True,
        batch_size=args.batch_size,
        grad_ckpt=args.grad_ckpt
    )

    # Check dataloader before calling trainer.train()
    print("Checking dataloader...")
    try:
        # Try to get one batch of data
        data_iter = iter(lora_trainer.train_dataloader)
        first_batch = next(data_iter)
        print("Successfully loaded first batch")
        print(f"Batch data type: {type(first_batch)}")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  - {key}: type {type(value)}")
        print("Dataloader check passed, ready to start training")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

    # Start training
    print("\nStarting LoRA fine-tuning...")
    print(f"Training samples: batch size: {args.batch_size}, gradient accumulation steps: {args.grad_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accumulation_steps}")
    print(f"Training epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {lora_trainer.patience}")
    lora_trainer.train(resumable_with_seed=args.resumable_with_seed)

    print("\nLoRA fine-tuning complete!")
    print(f"LoRA weights saved to: ckpts/{args.exp_name}_lora/")

if __name__ == "__main__":
    main()