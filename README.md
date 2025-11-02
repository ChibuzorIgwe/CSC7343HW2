# Piano Music Composer using Transformer Model

A deep learning project that generates piano music using a Transformer-based language model trained on the MAESTRO dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Generation](#generation)
- [File Descriptions](#file-descriptions)
- [License](#license)

## 🎵 Overview

This project implements a Transformer-based model that learns to compose piano music by training on MIDI files from the MAESTRO dataset. The model treats music generation as a sequence prediction task, where musical events (note on/off, timing shifts, velocity changes) are encoded as discrete tokens and predicted autoregressively.

## ✨ Features

- **Transformer Architecture**: Uses multi-head self-attention mechanisms for learning long-range musical dependencies
- **MIDI Processing**: Converts MIDI files to discrete event sequences and vice versa
- **Autoregressive Generation**: Generates music sequences token-by-token with temperature-based sampling
- **Pre-trained Model Loading**: Supports loading trained weights from Google Drive
- **Flexible Training**: Configurable hyperparameters and learning rate scheduling
- **Minimum Duration Guarantee**: Ensures generated compositions are at least 20 seconds long

## 📁 Project Structure

```
HomeWork2/
├── hw2.py                    # Main implementation with Transformer model and Composer class
├── model_base.py            # Abstract base classes for model structure
├── midi2seq.py              # MIDI processing utilities (conversion, segmentation)
├── CSC_7343_HW2.ipynb      # Jupyter notebook for training and testing
├── composer_model.pth       # Pre-trained model weights (87MB)
├── maestro-v1.0.0/         # MAESTRO dataset directory
├── piano1.midi              # Sample generated MIDI file
└── README.md                # This file
```

## 🔧 Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- pretty_midi
- gdown (for downloading pre-trained models)
- wget (fallback for downloads)

## 📦 Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install torch numpy pretty_midi gdown wget
```

3. Download the MAESTRO dataset (optional, only needed for training):
   - Download from: https://magenta.tensorflow.org/datasets/maestro
   - Extract to `maestro-v1.0.0/` directory

## 🚀 Usage

### Quick Start - Generate Music

```python
import logging
from hw2 import Composer
from midi2seq import seq2piano

# Configure logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Load pre-trained model
composer = Composer(load_trained=True)

# Generate music (at least 20 seconds)
sequence = composer.compose(max_length=3000, temperature=1.0)

# Convert to MIDI and save
midi = seq2piano(sequence)
midi.write('generated_music.midi')
print("Music generated and saved to 'generated_music.midi'")
```

### Training from Scratch

```python
import torch
from hw2 import Composer
from midi2seq import process_midi_seq

# Load and prepare data
print("Loading MIDI data...")
data = process_midi_seq(datadir='.', n=50000, maxlen=100)
data_tensor = torch.from_numpy(data)

# Initialize model
composer = Composer(load_trained=False)

# Training loop
num_epochs = 200
batch_size = 64

for epoch in range(num_epochs):
    # Shuffle data
    indices = torch.randperm(len(data_tensor))
    
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(data_tensor), batch_size):
        batch = data_tensor[indices[i:i+batch_size]]
        loss = composer.train(batch)
        epoch_loss += loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Step learning rate scheduler
    composer.step_scheduler()

# Save trained model
composer.save_model('composer_model.pth')
```

## 🏗️ Model Architecture

### Transformer Components

1. **Embedding Layer**: Converts discrete event tokens to continuous vectors (d_model=512)
2. **Positional Encoding**: Adds sinusoidal position information to embeddings
3. **Transformer Encoder**: 6 layers with:
   - Multi-head self-attention (8 heads)
   - Feed-forward networks (2048 dimensions)
   - Layer normalization and dropout (0.1)
4. **Output Projection**: Linear layer mapping to vocabulary size

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 356 | Total number of unique event tokens |
| `d_model` | 512 | Embedding dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer layers |
| `dim_feedforward` | 2048 | Hidden dimension in FFN |
| `dropout` | 0.1 | Dropout probability |
| `learning_rate` | 0.0001 | Initial learning rate |

## 📊 Dataset

### MAESTRO v1.0.0

- **Source**: MIDI Archive of Classical Music Performance
- **Size**: ~200 hours of virtuosic piano performances
- **Format**: High-quality MIDI recordings with precise timing

### Event Encoding

The model uses a vocabulary of 356 tokens representing:

1. **Note On Events** (0-127): Piano key press
2. **Note Off Events** (128-255): Piano key release
3. **Time Shift Events** (256-355): Time delays (0.01 to 1.01 seconds)
4. **Velocity Events** (356+): Note velocity/dynamics

### Sequence Processing

- MIDI files are converted to event sequences
- Sequences are segmented into fixed-length chunks (100 tokens)
- Training uses overlapping segments for better coverage

## 🎓 Training

### Training Process

1. **Data Loading**: Process MAESTRO MIDI files into event sequences
2. **Segmentation**: Split sequences into 100-token chunks with 50% overlap
3. **Batching**: Group sequences into batches (default: 64)
4. **Optimization**: Adam optimizer with learning rate scheduling
5. **Loss**: Cross-entropy loss on next-token prediction

### Training Tips

- **GPU Recommended**: Training is significantly faster on CUDA-enabled GPUs
- **Batch Size**: Adjust based on available memory (32-128 typical)
- **Epochs**: Train for 100-200 epochs for good results
- **Checkpointing**: Save model periodically to resume training
- **Learning Rate**: Uses step decay (γ=0.95 every 5 epochs)

### Expected Performance

- Initial loss: ~2.8
- After 100 epochs: ~1.2-1.4
- After 200 epochs: ~1.0-1.2

## 🎹 Generation

### Composition Process

The model generates music autoregressively:

1. **Initialization**: Starts with a time shift token
2. **Prediction**: Predicts next token probabilities
3. **Sampling**: Samples from probability distribution (with temperature)
4. **Accumulation**: Tracks time to ensure minimum duration
5. **Termination**: Stops when target duration reached (≥20 seconds)

### Temperature Parameter

- **Low (0.5-0.8)**: More conservative, predictable compositions
- **Medium (0.9-1.1)**: Balanced creativity and coherence
- **High (1.2-1.5)**: More random and experimental

### Generation Parameters

```python
sequence = composer.compose(
    max_length=3000,    # Maximum number of tokens
    temperature=1.0      # Sampling temperature
)
```

## 📝 File Descriptions

### Core Implementation Files

- **`hw2.py`**: Main implementation file containing:
  - `PositionalEncoding`: Sinusoidal positional encoding layer
  - `TransformerMusicModel`: Complete transformer architecture
  - `Composer`: High-level interface for training and generation
  
- **`model_base.py`**: Abstract base classes defining the API:
  - `ModelBase`: Base class with `train()` method
  - `ComposerBase`: Extends ModelBase with `compose()` method

- **`midi2seq.py`**: MIDI processing utilities:
  - `piano2seq()`: Convert MIDI to event sequence
  - `seq2piano()`: Convert event sequence to MIDI
  - `process_midi_seq()`: Batch process MIDI files
  - `Event`: Event class for encoding/decoding

### Notebook and Data

- **`CSC_7343_HW2.ipynb`**: Interactive notebook for:
  - Training the model
  - Generating compositions
  - Visualizing results
  
- **`composer_model.pth`**: Pre-trained model weights (87 MB)

### Variants

- **`hw2Old.py`**, **`hw2New.py`**, **`hw2Colab.py`**: Development versions

## 🎯 Key Methods

### Composer Class

```python
# Initialize (with or without pre-trained weights)
composer = Composer(load_trained=True)

# Train on a batch
loss = composer.train(batch_tensor)

# Generate music
sequence = composer.compose(max_length=3000, temperature=1.0)

# Save model
composer.save_model('my_model.pth')

# Step learning rate scheduler
composer.step_scheduler()
```

## 🔍 Technical Details

### Attention Mechanism

The model uses causal (autoregressive) attention masks to ensure each position can only attend to previous positions, preventing information leakage during training.

### Gradient Clipping

Gradients are clipped to max norm of 1.0 to prevent exploding gradients during training.

### Device Management

Automatically detects and uses CUDA GPU if available, otherwise falls back to CPU.

## 📈 Performance Considerations

- **Memory**: Model requires ~100MB RAM (512MB with gradients during training)
- **Speed**: Generation is real-time on modern CPUs (~100 tokens/second)
- **Training**: Full training takes ~2-4 hours on GPU (24+ hours on CPU)

## 🐛 Known Issues

- Sequential note events for the same pitch are logged as warnings
- Very short notes (<0.01s) may be dropped during conversion
- Temperature >2.0 may produce incoherent sequences

## 🤝 Contributing

This is a homework project for CSC 7343. Please follow academic integrity guidelines.

## 📄 License

Copyright 2020 Jian Zhang (base code)
Copyright 2025 (Transformer implementation)

## 🙏 Acknowledgments

- **MAESTRO Dataset**: Google Magenta team
- **pretty_midi**: Colin Raffel
- **Base Architecture**: Course materials and references

---

**Course**: CSC 7343 - Deep Learning  
**Institution**: [Your Institution]  
**Semester**: Fall 2025  
**Assignment**: Homework 2 - Music Generation with Transformers
