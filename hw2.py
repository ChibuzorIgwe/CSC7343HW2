"""
Piano Music Composer using Transformer-based Language Model
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math

# Import base class and MIDI processing functions
from model_base import ComposerBase
from midi2seq import process_midi_seq, seq2piano, dim

# Attempt to import gdown for Google Drive downloads
try:
    import gdown
except ImportError:
    gdown = None

# Attempt to import wget as fallback
try:
    import wget
except ImportError:
    wget = None


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerMusicModel(nn.Module):
    """Transformer-based music generation model."""
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(TransformerMusicModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive generation."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_len]
            src_mask: Optional attention mask
        
        Returns:
            Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # Embed and scale
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Generate causal mask if not provided
        if src_mask is None:
            device = src.device
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)
        
        # Pass through transformer
        output = self.transformer_encoder(src, src_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output


class Composer(ComposerBase):
    """
    Composer class for generating piano music using a Transformer model.
    Inherits from ComposerBase.
    """
    
    def __init__(self, load_trained=False):
        """
        Initialize the Composer model.
        
        Args:
            load_trained: If True, load trained weights from Google Drive.
                         If False, initialize a new model.
        """
        # Hyperparameters
        self.vocab_size = dim  # Vocabulary size from midi2seq.py
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.learning_rate = 0.0001
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self.device}')
        
        # Initialize model
        self.model = TransformerMusicModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.95)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Load trained model if requested
        if load_trained:
            logging.info('Loading trained model from Google Drive...')
            self.load_trained_model()
    
    def load_trained_model(self):
        """
        Load trained model weights from Google Drive.
        
        To use this feature:
        1. Train your model and save weights using torch.save(model.state_dict(), 'composer_model.pth')
        2. Upload 'composer_model.pth' to Google Drive
        3. Get the shareable link (make sure it's set to "Anyone with the link can view")
        4. Extract the file ID from the link
        5. Replace GOOGLE_DRIVE_FILE_ID below with your file ID
        
        Example Google Drive link:
        https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing
        The file ID is: 1a2b3c4d5e6f7g8h9i0j
        """
        # TODO: Replace with your actual Google Drive file ID (just the ID, not the full URL)
        GOOGLE_DRIVE_FILE_ID = "1Z2xOuAr1brKmwn-ISyYsLBSo1Zg5vlVF"
        
        model_path = 'composer_model.pth'
        
        try:
            # Check if file already exists
            if not os.path.exists(model_path):
                logging.info('Downloading model from Google Drive...')
                download_success = False
                
                # Method 1: Try using gdown with id parameter (more reliable)
                if gdown is not None:
                    try:
                        logging.info('Attempting download with gdown...')
                        gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=model_path, quiet=False)
                        download_success = True
                        logging.info('Downloaded successfully using gdown')
                    except Exception as e:
                        logging.warning(f'gdown failed: {e}')
                
                # Method 2: Fallback to wget
                if not download_success and wget is not None:
                    try:
                        logging.info('Attempting download with wget...')
                        url = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'
                        wget.download(url, out=model_path)
                        download_success = True
                        logging.info('Downloaded successfully using wget')
                    except Exception as e:
                        logging.warning(f'wget failed: {e}')
                
                if not download_success:
                    raise Exception(
                        f"Failed to download model. Please try downloading manually from:\n"
                        f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}\n"
                        f"Save it as '{model_path}' in the current directory."
                    )
            
            # Load model weights
            logging.info(f'Loading model weights from {model_path}...')
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logging.info('Model loaded successfully!')
            
        except Exception as e:
            logging.error(f'Error loading trained model: {e}')
            logging.warning('Using randomly initialized model instead.')
    
    def train(self, x):
        """
        Train the model on one batch of data.
        
        Args:
            x: Tensor of shape [batch_size, seq_len] containing event codes.
               Must be able to handle sequences of length 100.
        
        Returns:
            Average loss value (float) for this batch.
        """
        self.model.train()
        
        # Ensure x is on the correct device
        x = x.to(self.device)
        
        # Create input and target sequences
        # Input: all tokens except the last one
        # Target: all tokens except the first one
        inp = x[:, :-1]  # Shape: [batch_size, seq_len-1]
        tgt = x[:, 1:]   # Shape: [batch_size, seq_len-1]
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(inp)  # Shape: [batch_size, seq_len-1, vocab_size]
        
        # Reshape for loss calculation
        output = output.reshape(-1, self.vocab_size)  # [batch_size * (seq_len-1), vocab_size]
        tgt = tgt.reshape(-1)  # [batch_size * (seq_len-1)]
        
        # Compute loss
        loss = self.criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Return loss as float
        return loss.item()
    
    def compose(self, max_length=3000, temperature=1.0):
        """
        Generate a music sequence that plays for at least 20 seconds.
        
        Args:
            max_length: Maximum number of tokens to generate (default: 3000)
            temperature: Sampling temperature for diversity (default: 1.0)
        
        Returns:
            NumPy array of generated event codes (1D array)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start with a shift token (event code for a small time shift)
            # Using code 256 (128*2 + 0), which represents a shift event
            start_token = 128 * 2  # Shift event with value 0
            generated = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
            
            # Track accumulated time (in seconds)
            accumulated_time = 0.0
            # Target 25 seconds to ensure we get at least 20 seconds after MIDI conversion
            target_time = 25.0
            
            for _ in range(max_length):
                # Get predictions for the next token
                output = self.model(generated)  # [1, seq_len, vocab_size]
                
                # Get the last token's predictions
                logits = output[0, -1, :] / temperature  # [vocab_size]
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)  # [1]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Update accumulated time based on shift events
                token_value = next_token.item()
                if 128*2 <= token_value < 128*2 + 100:
                    # This is a shift event
                    shift_value = (token_value - 128*2) / 100 + 0.01
                    accumulated_time += shift_value
                
                # Check if we've generated enough time
                # Require at least 25 seconds and minimum 800 tokens for musical coherence
                if accumulated_time >= target_time and len(generated[0]) > 800:
                    break
            
            # Convert to numpy array and return
            result = generated[0].cpu().numpy()
            
            logging.info(f'Generated sequence of length {len(result)} (approx {accumulated_time:.2f} seconds)')
            
            return result
    
    def save_model(self, filepath='composer_model.pth'):
        """
        Save model weights to a file.
        
        Args:
            filepath: Path where to save the model weights
        """
        torch.save(self.model.state_dict(), filepath)
        logging.info(f'Model saved to {filepath}')
    
    def step_scheduler(self):
        """Step the learning rate scheduler (call after each epoch)."""
        self.scheduler.step()
