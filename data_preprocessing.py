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
Data Preprocessing Pipeline for Music Generation

This file contains original utilities for processing lyrics, audio, and 
style embeddings for music generation training data.
"""

import os
import torch
import json
import re
import torchaudio
import librosa
import numpy as np
import glob
import time
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Import required modules
from g2p.g2p_generation import chn_eng_g2p
from g2p.g2p import PhonemeBpeTokenizer
from muq import MuQMuLan

class CNENTokenizer:
    """Chinese-English Tokenizer for processing text conversion"""
    def __init__(self):
        with open("./g2p/g2p/vocab.json", "r", encoding='utf-8') as file:
            self.phone2id = json.load(file)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        # Use original chn_eng_g2p function
        from g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x + 1 for x in token]  # Add 1 to token id, as 0 is usually reserved for padding token
        return token

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])

def parse_lyrics(lrc_content):
    """Parse timestamps and lyrics from LRC file content"""
    lyrics_with_time = []
    lrc_content = lrc_content.strip()
    for line in lrc_content.split("\n"):
        try:
            # LRC format is usually [00:12.34]lyrics content
            match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
            if match:
                mins, secs_ms, lyric = match.groups()
                secs = int(mins) * 60 + float(secs_ms)
                lyric = lyric.strip()
                if lyric:  # Only keep lyrics with actual content
                    lyrics_with_time.append((secs, lyric))
        except Exception as e:
            print(f"Error parsing line: {line}, error: {e}")
            continue
    return lyrics_with_time

def lrc_to_pt(lrc_file_path, output_pt_path, max_frames=2048):
    """Convert LRC file to PT file
    
    Args:
        lrc_file_path (str): Path to LRC file
        output_pt_path (str): Path to output PT file
        max_frames (int): Maximum frames, default 2048
    
    Returns:
        dict: Dictionary containing LRC tokens and time information
    """
    # Read LRC file
    with open(lrc_file_path, 'r', encoding='utf-8') as f:
        lrc_content = f.read()
    
    # Initialize tokenizer
    tokenizer = CNENTokenizer()
    
    # Parse LRC content
    lrc_with_time = parse_lyrics(lrc_content)
    
    # Set sampling rate and downsample rate parameters
    sampling_rate = 44100
    downsample_rate = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)
    
    # Filter out lyrics beyond maximum duration
    lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time 
                    if time_start < max_secs]
    
    # For 2048 frame case, remove last lyric line (based on original code logic)
    if max_frames == 2048 and len(lrc_with_time) >= 1:
        lrc_with_time = lrc_with_time[:-1]
    
    # Prepare special token IDs for model use
    comma_token_id = 1
    period_token_id = 2
    
    # Key change: save each lyric line as independent token list
    times = []
    lyrics_tokens = []
    
    for time, line in lrc_with_time:
        times.append(time)
        # Convert to tokens but don't add period ending, let dataset handle it
        line_tokens = tokenizer.encode(line)
        # Save as Python list, not tensor
        lyrics_tokens.append(line_tokens)
    
    # Save timestamps as tensor
    time_tensor = torch.tensor(times, dtype=torch.float)
    
    # Save results as PT file
    # Key modification: no longer use lrc_tensor, directly save token list
    output_data = {
        'lrc': lyrics_tokens,  # Now a list, each element is a token list for one lyric line
        'time': time_tensor    # Timestamp tensor
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    
    # Save PT file
    torch.save(output_data, output_pt_path)
    
    print(f"Successfully converted {lrc_file_path} to {output_pt_path}")
    return output_data

def batch_convert_lrc_to_pt(lrc_dir, output_dir, max_frames=2048):
    """Batch convert all LRC files in directory to PT files
    
    Args:
        lrc_dir (str): Directory containing LRC files
        output_dir (str): Output directory for PT files
        max_frames (int): Maximum frames, default 2048
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(lrc_dir):
        if filename.endswith('.lrc'):
            lrc_path = os.path.join(lrc_dir, filename)
            pt_path = os.path.join(output_dir, filename.replace('.lrc', '.pt'))
            try:
                lrc_to_pt(lrc_path, pt_path, max_frames)
            except Exception as e:
                print(f"Error processing {lrc_path}: {e}")

def encode_audio_with_vae(audio_path, vae_model, device="cpu", sample_rate=44100):
    """Encode audio using VAE model"""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # Ensure stereo (2 channels)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    
    # Move to device
    waveform = waveform.to(device)
    
    # Add batch dimension if needed
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(0)  # [1, 2, T]
    
    with torch.no_grad():
        # Get encoder output
        encoded = vae_model.encoder(waveform)
        
        # Extract first 64 dimensions from 128-dim features as mean (correct latent representation)
        latent_mean = encoded[:, :64, :]
        
        return latent_mean

def process_audio_files_with_vae(audio_files, output_dir, vae_model, device="cpu"):
    """Process audio files using VAE and save latent representations"""
    os.makedirs(output_dir, exist_ok=True)
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        base_name = os.path.basename(audio_file).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}.pt")
        
        try:
            latent = encode_audio_with_vae(audio_file, vae_model, device=device)
            torch.save(latent, output_path)
            print(f"Saved to: {output_path}")
            print(f"Latent representation shape: {latent.shape}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

def get_style_from_audio(wav_path, model, start_time=0.0, duration=10.0):
    """Generate style prompt from audio at specified time point"""
    if model is None:
        # Fallback when no MuQMuLan model available
        print(f"Using randomly generated style embedding as fallback")
        return torch.randn(1, 512).half()
    
    # Get audio duration
    ext = os.path.splitext(wav_path)[-1].lower()
    if ext == ".mp3":
        from mutagen.mp3 import MP3
        meta = MP3(wav_path)
        audio_len = meta.info.length
    else:
        audio_len = librosa.get_duration(path=wav_path)
    
    print(f"Audio length: {audio_len:.2f} seconds")
    
    # Validate specified start time and duration
    if start_time < 0:
        print(f"Warning: start time {start_time}s < 0, set to 0")
        start_time = 0
    
    if start_time + duration > audio_len:
        print(f"Warning: start time {start_time}s + duration {duration}s exceeds audio length, adjusting duration")
        duration = audio_len - start_time
    
    print(f"Extracting segment: starting at {start_time:.2f}s, duration {duration:.2f}s")
    
    # Load audio
    wav, sr = librosa.load(wav_path, sr=24000, offset=start_time, duration=duration)
    
    # Convert to tensor and move to correct device
    wav = torch.tensor(wav).unsqueeze(0)
    if torch.cuda.is_available():
        wav = wav.cuda()
    
    # Encode audio using model
    with torch.no_grad():
        audio_emb = model(wavs=wav)  # [1, 512]
    
    # Ensure output is half precision
    audio_emb = audio_emb.half()
    print(f"Generated style embedding shape: {audio_emb.shape}")
    return audio_emb

def get_style_from_text(text, model):
    """Generate style prompt from text"""
    if model is None:
        # Fallback when no MuQMuLan model available
        print(f"Using randomly generated style embedding as fallback, text prompt: '{text}'")
        return torch.randn(1, 512).half()
    
    print(f"Using text prompt: '{text}'")
    
    # Encode text using model - note texts parameter needs list
    with torch.no_grad():
        text_emb = model(texts=[text])  # [1, 512] - corrected to list format
    
    # Ensure output is half precision
    text_emb = text_emb.half()
    print(f"Generated style embedding shape: {text_emb.shape}")
    return text_emb

def find_audio_files(directory, recursive=True, extensions=['.wav', '.mp3', '.flac']):
    """
    Recursively find all audio files in specified directory
    
    Args:
        directory (str): Directory path to search
        recursive (bool): Whether to recursively search subdirectories
        extensions (list): List of audio file extensions to find
    
    Returns:
        list: List of audio file paths
    """
    audio_files = []
    
    # Ensure extensions are lowercase and include dot
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
    
    if recursive:
        # Recursive search
        for ext in extensions:
            pattern = os.path.join(directory, f'**/*{ext}')
            audio_files.extend(glob.glob(pattern, recursive=True))
    else:
        # Search current directory only
        for ext in extensions:
            pattern = os.path.join(directory, f'*{ext}')
            audio_files.extend(glob.glob(pattern))
    
    return sorted(audio_files)

def process_style_files(audio_files=None, input_dir=None, output_dir="dataset/style", 
                        use_text=False, text_prompt="", start_time=0.0, duration=10.0,
                        recursive=True, extensions=['.wav', '.mp3', '.flac'],
                        overwrite=False, verbose=True, muq_model=None):
    """
    Batch process audio files and generate style embeddings
    
    Args:
        audio_files (list): List of audio file paths, if None use input_dir
        input_dir (str): Input directory to process, must provide if audio_files is None
        output_dir (str): Output directory path
        use_text (bool): Whether to use text prompt instead of audio
        text_prompt (str): Text prompt content
        start_time (float): Audio start extraction time point (seconds)
        duration (float): Extracted audio duration (seconds)
        recursive (bool): Whether to recursively process files in subdirectories
        extensions (list): Audio file extensions to process
        overwrite (bool): Whether to overwrite existing output files
        verbose (bool): Whether to show detailed output information
        muq_model: MuQMuLan model instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # If no file list provided, find from directory
    if audio_files is None:
        if input_dir is None:
            raise ValueError("Must provide audio_files or input_dir parameter")
        
        print(f"Finding audio files in directory '{input_dir}'...")
        audio_files = find_audio_files(input_dir, recursive=recursive, extensions=extensions)
        print(f"Found {len(audio_files)} audio files")
    
    # No files to process
    if not audio_files:
        print("No audio files found to process")
        return
    
    # Show processing parameters
    print(f"Batch processing configuration:")
    print(f"- Output directory: {output_dir}")
    print(f"- Use text prompt: {use_text}")
    if use_text:
        print(f"- Text prompt content: '{text_prompt}'")
    else:
        print(f"- Audio start time: {start_time} seconds")
        print(f"- Audio duration: {duration} seconds")
    print(f"- Total files: {len(audio_files)}")
    
    # Create progress bar with tqdm
    start_time_total = time.time()
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        if verbose:
            print(f"\nProcessing: {audio_file}")
        
        base_name = os.path.basename(audio_file).split('.')[0]
        
        # Distinguish filename suffix
        if use_text:
            file_suffix = 'text'
        else:
            # Include time information in filename
            file_suffix = f'audio_{start_time:.1f}_{duration:.1f}'
            
        output_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.pt")
        
        # Check if output file already exists
        if os.path.exists(output_path) and not overwrite:
            if verbose:
                print(f"Skip: {output_path} already exists (use --overwrite parameter to overwrite)")
            skip_count += 1
            continue
        
        try:
            if use_text:
                style_emb = get_style_from_text(text_prompt, muq_model)
            else:
                style_emb = get_style_from_audio(audio_file, muq_model, start_time=start_time, duration=duration)
            
            # Validate shape is correct
            if style_emb.shape != (1, 512):
                print(f"Warning: generated embedding shape {style_emb.shape} doesn't match expected (1, 512)")
            
            torch.save(style_emb, output_path)
            if verbose:
                print(f"Saved to: {output_path}")
            success_count += 1
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            fail_count += 1
    
    # Summarize processing results
    elapsed_time = time.time() - start_time_total
    print(f"\nProcessing complete! Total time: {elapsed_time:.2f} seconds")
    print(f"- Success: {success_count}")
    print(f"- Skipped: {skip_count}")
    print(f"- Failed: {fail_count}")
    print(f"Generated style prompts can be used for DiffRhythm model training")

def load_vae_model(device="cpu"):
    """Load pretrained VAE model"""
    try:
        # Download pretrained VAE model
        vae_ckpt_path = hf_hub_download(
            repo_id="ASLP-lab/DiffRhythm-vae",
            filename="vae_model.pt",
            cache_dir="./pretrained",
        )
        
        # Load VAE model
        vae = torch.jit.load(vae_ckpt_path, map_location=device)
        print("Successfully loaded VAE model")
        return vae
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return None

def load_muq_model(device="cpu"):
    """Load MuQMuLan model"""
    try:
        # Load MuQMuLan model using correct class
        muq_model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
        muq_model = muq_model.to(device).eval()
        print("Successfully loaded MuQMuLan model")
        return muq_model
    except Exception as e:
        print(f"Error loading MuQMuLan model: {str(e)}")
        return None

def main():
    """Main function for data preprocessing"""
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    # Modify these paths according to your setup
    lrc_dir = "traindata/opera"  # Directory containing LRC files
    audio_dir = "traindata/opera"  # Directory containing audio files
    output_lrc_dir = "dataset/lrc"  # Output directory for LRC PT files
    output_latent_dir = "dataset/latent"  # Output directory for VAE latent files
    output_style_dir = "dataset/style"  # Output directory for style files
    
    # 1. Process LRC files
    print("=" * 50)
    print("Processing LRC files...")
    if os.path.exists(lrc_dir):
        batch_convert_lrc_to_pt(lrc_dir, output_lrc_dir)
    else:
        print(f"LRC directory {lrc_dir} not found, skipping LRC processing")
    
    # 2. Process audio files with VAE
    print("=" * 50)
    print("Processing audio files with VAE...")
    vae_model = load_vae_model(device)
    if vae_model and os.path.exists(audio_dir):
        audio_files = find_audio_files(audio_dir)
        if audio_files:
            process_audio_files_with_vae(audio_files, output_latent_dir, vae_model, device)
        else:
            print("No audio files found for VAE processing")
    else:
        print("Skipping VAE processing (model not loaded or directory not found)")
    
    # 3. Generate style embeddings
    print("=" * 50)
    print("Generating style embeddings...")
    muq_model = load_muq_model(device)
    if os.path.exists(audio_dir):
        # Option 1: Generate from audio
        process_style_files(
            input_dir=audio_dir,
            output_dir=output_style_dir,
            use_text=False,
            start_time=10.0,  # Start from 10 seconds
            duration=10.0,    # Extract 10 seconds
            muq_model=muq_model
        )
        
        # Option 2: Generate from text (uncomment to use)
        # process_style_files(
        #     input_dir=audio_dir,
        #     output_dir=output_style_dir,
        #     use_text=True,
        #     text_prompt="Chinese opera",
        #     muq_model=muq_model
        # )
    else:
        print(f"Audio directory {audio_dir} not found, skipping style processing")
    
    print("=" * 50)
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()