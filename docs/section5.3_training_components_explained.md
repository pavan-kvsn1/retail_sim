# Training Components: A Complete Beginner's Guide

## Overview: The Big Picture

Think of training a machine learning model like teaching a student to predict what groceries someone will buy. Here's how all the pieces fit together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE TRAINING JOURNEY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA PREPARATION (WorldModelDataset)                        │
│     "Organize all the shopping history into learnable examples" │
│                                                                  │
│  2. BATCHING (WorldModelDataLoader)                             │
│     "Group similar examples together for efficient learning"    │
│                                                                  │
│  3. MODEL (WorldModel)                                          │
│     "The student's brain that learns patterns"                  │
│                                                                  │
│  4. LOSS CALCULATION (WorldModelLoss)                           │
│     "Measure how wrong the predictions are"                     │
│                                                                  │
│  5. LEARNING (Trainer)                                          │
│     "Adjust the brain to make better predictions"               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: The Data Pipeline (WorldModelDataset)

### What It Does
**In Simple Terms**: Imagine you have millions of shopping receipts. The `WorldModelDataset` organizes them into a format the model can learn from.

### The Data Structure

```python
# Each shopping trip becomes a training example with:
{
    "customer_info": "Who is shopping? (age, income, shopping habits)",
    "when": "What time/day is it?",
    "where": "Which store?",
    "trip_type": "Quick top-up or full weekly shop?",
    "products": ["milk", "bread", "eggs", ...],  # What they bought
    "prices": [2.99, 1.49, 3.99, ...]           # How much each cost
}
```

### Key Components

#### 1. **WorldModelBatch** (The Container)
```python
@dataclass
class WorldModelBatch:
    # Dense context (400 dimensions total)
    customer_context: np.ndarray      # [B, 192] - Who is shopping
    temporal_context: np.ndarray      # [B, 64]  - When they're shopping
    store_context: np.ndarray         # [B, 96]  - Where they're shopping
    trip_context: np.ndarray          # [B, 48]  - Type of shopping trip
    
    # Sequence data (variable length)
    product_embeddings: np.ndarray    # [B, S, 256] - Products as vectors
    price_features: np.ndarray        # [B, S, 64]  - Price information
    attention_mask: np.ndarray        # [B, S]      - Which items are real vs padding
    
    # Labels (what we're trying to predict)
    masked_targets: np.ndarray        # Which products were masked
    auxiliary_labels: Dict            # Trip type, basket size, etc.
```

**What B and S mean:**
- `B` = Batch size (e.g., 256 shopping trips at once)
- `S` = Sequence length (e.g., up to 50 products per basket)

#### 2. **Loading the Data**
```python
class WorldModelDataset:
    def __init__(self, project_root, split='train'):
        # Step 1: Load the shopping trips
        self.samples = pd.read_parquet(f'data/samples/{split}_samples.parquet')
        # Contains: basket_id, customer_id, store_id, shop_week, etc.
        
        # Step 2: Load pre-computed embeddings
        self.product_embeddings = np.load('data/cache/product_embeddings.npy')
        # Each product is a 256-dimensional vector
        
        self.customer_history = np.load('data/cache/customer_history.npy')
        # Each customer's shopping history as a 160-dimensional vector
        
        # Step 3: Load actual transactions
        self.transactions = pd.read_csv('raw_data/transactions.csv')
        # The detailed list of what was bought in each basket
```

#### 3. **Creating Training Examples**
```python
def __getitem__(self, idx):
    """Get one training example."""
    # Step 1: Get the shopping trip info
    sample = self.samples.iloc[idx]
    customer_id = sample['customer_id']
    basket_id = sample['basket_id']
    
    # Step 2: Get products in this basket
    products = self.basket_products[basket_id]  # e.g., [123, 456, 789]
    
    # Step 3: Build all the context tensors
    t1 = self._encode_customer(customer_id)      # Customer features
    t2 = self._encode_products(products)         # Product embeddings
    t3 = self._encode_temporal(sample)           # Time features
    t4 = self._encode_prices(products)           # Price features
    t5 = self._encode_store(sample['store_id']) # Store features
    t6 = self._encode_trip(sample)               # Trip type features
    
    # Step 4: Apply masking (hide some products for the model to predict)
    masked_products, targets = self._apply_masking(products)
    
    return {
        't1': t1, 't2': t2, 't3': t3, 
        't4': t4, 't5': t5, 't6': t6,
        'masked_targets': targets
    }
```

---

## Part 2: Batching (WorldModelDataLoader)

### What It Does
**In Simple Terms**: Instead of teaching one example at a time, we group similar examples together for faster learning.

### Why Batching Matters

```
Without Batching:
Example 1 (5 products)  → Learn → Update
Example 2 (3 products)  → Learn → Update
Example 3 (20 products) → Learn → Update
⏰ Very slow! (one at a time)

With Batching:
[Example 1 (5 products)  ]
[Example 2 (3 products)  ] → Learn together → Update once
[Example 3 (20 products) ]
⚡ Much faster! (256 at once)
```

### Bucket Batching (Smart Grouping)

```python
class WorldModelDataLoader:
    def __init__(self, dataset, batch_size=256, bucket_batching=True):
        if bucket_batching:
            # Group baskets by size to minimize padding
            self.buckets = {
                1: [baskets with 1-10 products],
                2: [baskets with 11-20 products],
                3: [baskets with 21-30 products],
                4: [baskets with 31-50 products]
            }
```

**Why this is smart:**
```
Bad Batching (random):
[5 products  + 45 padding] ← Waste!
[48 products + 2 padding ]
[3 products  + 47 padding] ← Waste!

Good Batching (bucketed):
[5 products  + 5 padding]  ← Efficient!
[3 products  + 7 padding]
[8 products  + 2 padding]
```

### The Batching Process

```python
def __iter__(self):
    """Generate batches during training."""
    # Step 1: Shuffle indices (for randomness)
    if self.shuffle:
        np.random.shuffle(self.indices)
    
    # Step 2: Create batches
    for start in range(0, len(self.indices), self.batch_size):
        batch_indices = self.indices[start:start + self.batch_size]
        
        # Step 3: Get all samples in this batch
        samples = [self.dataset[i] for i in batch_indices]
        
        # Step 4: Pad sequences to same length
        batch = self._collate_samples(samples)
        
        yield batch
```

---

## Part 3: The Model (WorldModel)

### What It Does
**In Simple Terms**: The model is like a student's brain that learns to predict what products someone will buy based on their shopping history.

### Architecture Overview

```
INPUT STAGE: Combine all the context
┌─────────────────────────────────────┐
│ Customer Info (192d)                │
│ + Time Info (64d)                   │
│ + Store Info (96d)                  │  → Context Fusion
│ + Trip Type (48d)                   │     [400d → 512d]
└─────────────────────────────────────┘

SEQUENCE STAGE: Process the products
┌─────────────────────────────────────┐
│ Product Embeddings (256d per item)  │
│ + Price Features (64d per item)     │  → Product Fusion
└─────────────────────────────────────┘     [320d → 512d per item]

ENCODING STAGE: Learn patterns
┌─────────────────────────────────────┐
│ Mamba Encoder (4 layers)            │  → Efficient long-sequence
│ - Processes shopping history        │     processing (O(n) complexity)
│ - Captures temporal patterns        │
└─────────────────────────────────────┘

DECODING STAGE: Generate predictions
┌─────────────────────────────────────┐
│ Transformer Decoder (2 layers)      │  → Predict next products
│ - Cross-attention to context        │     with attention
│ - Self-attention over products      │
└─────────────────────────────────────┘

OUTPUT STAGE: Make predictions
┌─────────────────────────────────────┐
│ Product Prediction Head             │  → Which products?
│ Basket Size Head                    │  → Small/Medium/Large?
│ Price Sensitivity Head              │  → Budget/Mid/Premium?
│ Mission Type Head                   │  → Quick trip/Full shop?
└─────────────────────────────────────┘
```

### Model Configuration

```python
@dataclass
class WorldModelConfig:
    """Settings for the model architecture."""
    # Main dimension
    d_model: int = 512  # All internal processing uses 512 dimensions
    
    # Vocabulary
    n_products: int = 5003  # 5000 products + special tokens
    
    # Architecture depth
    mamba_num_layers: int = 4    # 4 Mamba layers for encoding
    decoder_num_layers: int = 2  # 2 Transformer layers for decoding
    
    # Sequence limits
    max_basket_len: int = 50  # Maximum 50 products per basket
```

### Forward Pass (How Prediction Works)

```python
class WorldModel(nn.Module):
    def forward(self, dense_context, product_embeddings, price_features, 
                attention_mask, masked_positions):
        """
        Make predictions for a batch of shopping trips.
        
        Args:
            dense_context: [B, 400] - Customer, time, store, trip info
            product_embeddings: [B, S, 256] - Product vectors
            price_features: [B, S, 64] - Price information
            attention_mask: [B, S] - Which positions are real
            masked_positions: [B, M] - Which products to predict
        
        Returns:
            masked_logits: [B, M, 5003] - Predictions for masked products
            aux_logits: Dict - Predictions for auxiliary tasks
            encoder_output: [B, S, 512] - Encoded representations
        """
        
        # STEP 1: Fuse dense context
        context_vector = self.context_fusion(dense_context)
        # [B, 400] → [B, 512]
        
        # STEP 2: Fuse product sequences
        sequence_features = torch.cat([product_embeddings, price_features], dim=-1)
        product_sequence = self.product_fusion(sequence_features, attention_mask)
        # [B, S, 320] → [B, S, 512]
        
        # STEP 3: Encode with Mamba (captures long-range patterns)
        encoded = self.mamba_encoder(product_sequence, context_vector)
        # [B, S, 512] → [B, S, 512] (with context awareness)
        
        # STEP 4: Decode with Transformer (generates predictions)
        decoded = self.transformer_decoder(
            tgt=product_sequence,
            memory=encoded,
            tgt_key_padding_mask=~attention_mask.bool()
        )
        # [B, S, 512] → [B, S, 512] (with cross-attention)
        
        # STEP 5: Predict masked products
        if masked_positions is not None:
            # Extract only the masked positions
            masked_hidden = self._gather_masked(decoded, masked_positions)
            # [B, S, 512] → [B, M, 512]
            
            # Predict which product it is
            masked_logits = self.product_head(masked_hidden)
            # [B, M, 512] → [B, M, 5003]
        
        # STEP 6: Auxiliary predictions
        pooled = decoded.mean(dim=1)  # [B, 512]
        aux_logits = {
            'basket_size': self.basket_size_head(pooled),      # [B, 4]
            'price_sensitivity': self.price_sens_head(pooled), # [B, 4]
            'mission_type': self.mission_type_head(pooled),    # [B, 5]
            'mission_focus': self.mission_focus_head(pooled)   # [B, 6]
        }
        
        return masked_logits, aux_logits, encoded
```

---

## Part 4: Loss Calculation (WorldModelLoss)

### What It Does
**In Simple Terms**: The loss function measures "how wrong" the model's predictions are. Lower loss = better predictions.

### Multiple Loss Components

```python
class WorldModelLoss:
    """
    Combines multiple loss functions for multi-task learning.
    """
    
    def __init__(self, n_products=5003):
        # 1. Focal Loss: For predicting masked products
        self.focal_loss = FocalLoss(gamma=2.0)
        
        # 2. Contrastive Loss: For learning product relationships
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # 3. Cross-Entropy: For auxiliary tasks
        self.ce_loss = nn.CrossEntropyLoss()
```

### Loss Calculation Process

```python
def forward(self, masked_logits, masked_targets, masked_mask,
            product_embeddings, product_ids, attention_mask,
            auxiliary_logits, auxiliary_labels, phase='main'):
    """
    Calculate total loss from all components.
    
    Returns:
        total_loss: Scalar value to minimize
        loss_dict: Breakdown of individual losses
    """
    
    # LOSS 1: Masked Product Prediction (Main Task)
    # "Did we correctly predict which products were masked?"
    focal = self.focal_loss(
        logits=masked_logits,      # [B, M, 5003] - Our predictions
        targets=masked_targets,    # [B, M] - True products
        mask=masked_mask           # [B, M] - Valid positions
    )
    # Example: If we predicted "milk" but it was "bread", high loss
    
    # LOSS 2: Contrastive Loss (Product Relationships)
    # "Do products that appear together have similar embeddings?"
    contrastive = self.contrastive_loss(
        embeddings=product_embeddings,  # [B, S, 512]
        product_ids=product_ids,        # [B, S]
        attention_mask=attention_mask   # [B, S]
    )
    # Example: "milk" and "cereal" should be closer than "milk" and "hammer"
    
    # LOSS 3: Auxiliary Tasks
    # "Did we predict basket size, price sensitivity, etc. correctly?"
    aux_losses = {}
    for task, logits in auxiliary_logits.items():
        if task in auxiliary_labels:
            aux_losses[task] = self.ce_loss(logits, auxiliary_labels[task])
    
    # COMBINE ALL LOSSES with phase-specific weights
    if phase == 'warmup':
        # During warmup, focus only on main task
        total_loss = focal
        weights = {'focal': 1.0}
    else:
        # During main training, balance all tasks
        total_loss = (
            0.60 * focal +                           # Main task (60%)
            0.20 * contrastive +                     # Relationships (20%)
            0.08 * aux_losses['basket_size'] +       # Basket size (8%)
            0.08 * aux_losses['price_sensitivity'] + # Price sens (8%)
            0.04 * aux_losses['mission_type']        # Mission (4%)
        )
        weights = {'focal': 0.60, 'contrastive': 0.20, ...}
    
    return total_loss, {
        'total': total_loss.item(),
        'focal': focal.item(),
        'contrastive': contrastive.item(),
        **{k: v.item() for k, v in aux_losses.items()}
    }
```

### Why Multiple Losses?

```
Single Task (Only predict products):
❌ Misses valuable signals
❌ Doesn't learn relationships
❌ Ignores context

Multi-Task (Predict products + auxiliary info):
✅ Learns richer representations
✅ Better generalization
✅ More robust predictions
```

---

## Part 5: The Training Loop (Trainer)

### What It Does
**In Simple Terms**: The `Trainer` orchestrates the entire learning process - like a teacher managing a classroom.

### Training Configuration

```python
@dataclass
class TrainingConfig:
    """All the settings for training."""
    # Model architecture
    n_products: int = 5003
    d_model: int = 512
    mamba_layers: int = 4
    decoder_layers: int = 2
    
    # Training hyperparameters
    batch_size: int = 256              # Process 256 examples at once
    learning_rate: float = 5e-5        # How fast to learn (0.00005)
    num_epochs: int = 20               # Go through data 20 times
    
    # Three-phase training
    warmup_epochs: int = 3             # Epochs 1-3: Gentle start
    finetune_epochs: int = 5           # Epochs 16-20: Final polish
    
    # Masking strategy
    mask_prob_train: float = 0.15      # Hide 15% of products normally
    mask_prob_finetune: float = 0.20   # Hide 20% during fine-tuning (harder!)
    
    # Validation
    eval_every_n_steps: int = 500      # Check progress every 500 steps
    early_stopping_patience: int = 3   # Stop if no improvement for 3 checks
```

### The Complete Training Journey

```python
class Trainer:
    def train(self):
        """The complete training process."""
        
        # ============================================
        # INITIALIZATION
        # ============================================
        print("Setting up...")
        
        # 1. Load the data
        train_dataset = WorldModelDataset(split='train')
        val_dataset = WorldModelDataset(split='validation')
        
        # 2. Create the model
        model = WorldModel(config)
        
        # 3. Create the loss function
        criterion = WorldModelLoss(n_products=5003)
        
        # 4. Create the optimizer (how we update the model)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        # ============================================
        # TRAINING LOOP
        # ============================================
        for epoch in range(1, 21):  # 20 epochs total
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/20")
            print(f"{'='*60}")
            
            # Determine training phase
            if epoch <= 3:
                phase = 'warmup'
                lr = 1e-5  # Lower learning rate
                mask_prob = 0.15
            elif epoch >= 16:
                phase = 'finetune'
                lr = 1e-5  # Lower learning rate again
                mask_prob = 0.20  # Harder masking
            else:
                phase = 'main'
                lr = 5e-5  # Normal learning rate
                mask_prob = 0.15
            
            print(f"Phase: {phase}, LR: {lr}, Masking: {mask_prob}")
            
            # ============================================
            # TRAIN FOR ONE EPOCH
            # ============================================
            model.train()  # Set to training mode
            train_loader = WorldModelDataLoader(train_dataset, batch_size=256)
            
            for batch_idx, batch in enumerate(train_loader):
                # Step 1: Get the data
                dense_context = batch.get_dense_context()
                product_embeddings = batch.product_embeddings
                price_features = batch.price_features
                attention_mask = batch.attention_mask
                masked_positions = batch.masked_positions
                masked_targets = batch.masked_targets
                
                # Step 2: Forward pass (make predictions)
                masked_logits, aux_logits, encoder_output = model(
                    dense_context=dense_context,
                    product_embeddings=product_embeddings,
                    price_features=price_features,
                    attention_mask=attention_mask,
                    masked_positions=masked_positions
                )
                
                # Step 3: Calculate loss (how wrong are we?)
                loss, loss_dict = criterion(
                    masked_logits=masked_logits,
                    masked_targets=masked_targets,
                    masked_mask=(masked_targets > 0),
                    product_embeddings=encoder_output,
                    product_ids=batch.product_token_ids,
                    attention_mask=attention_mask,
                    auxiliary_logits=aux_logits,
                    auxiliary_labels=batch.auxiliary_labels,
                    phase=phase
                )
                
                # Step 4: Backward pass (calculate gradients)
                loss.backward()
                
                # Step 5: Update model weights
                optimizer.step()
                optimizer.zero_grad()
                
                # Step 6: Log progress
                if batch_idx % 100 == 0:
                    print(f"Step {batch_idx}: Loss = {loss.item():.4f}")
                
                # Step 7: Validate periodically
                if batch_idx % 500 == 0:
                    val_loss = self.validate(val_dataset)
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    # Save if best model so far
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), 'best_model.pt')
                        print("✓ New best model saved!")
            
            # ============================================
            # END OF EPOCH
            # ============================================
            print(f"\nEpoch {epoch} complete!")
            
            # Final validation for this epoch
            val_loss = self.validate(val_dataset)
            print(f"Final Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, f'checkpoint_epoch_{epoch}.pt')
        
        print("\n🎉 Training complete!")
```

### What Happens in One Training Step

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. GET BATCH                                               │
│     ↓ DataLoader provides 256 shopping trips               │
│                                                              │
│  2. FORWARD PASS                                            │
│     ↓ Model makes predictions                               │
│     ↓ "I think basket #1 has milk, bread, eggs..."         │
│                                                              │
│  3. CALCULATE LOSS                                          │
│     ↓ Compare predictions to reality                        │
│     ↓ "You got 45% correct, loss = 2.34"                   │
│                                                              │
│  4. BACKWARD PASS                                           │
│     ↓ Calculate how to improve                              │
│     ↓ "Adjust weight #1 by -0.0001, weight #2 by +0.0002"│
│                                                              │
│  5. UPDATE WEIGHTS                                          │
│     ↓ Apply the adjustments                                 │
│     ↓ Model is now slightly better!                         │
│                                                              │
│  6. REPEAT                                                  │
│     ↓ Do this thousands of times                            │
│     ↓ Model gets better and better                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 6: How Everything Connects

### The Complete Data Flow

```
START: You run `python train.py --epochs 20 --batch-size 256`
│
├─ main() function starts
│  │
│  ├─ Parse command line arguments
│  │  └─ epochs=20, batch_size=256, device='cuda'
│  │
│  ├─ Create TrainingConfig
│  │  └─ All hyperparameters stored in one place
│  │
│  └─ Create Trainer(config)
│     │
│     ├─ Initialize Model (WorldModel)
│     │  ├─ WorldModelConfig created
│     │  ├─ ContextFusion layer
│     │  ├─ ProductSequenceFusion layer
│     │  ├─ MambaEncoder (4 layers)
│     │  ├─ TransformerDecoder (2 layers)
│     │  └─ Output heads (product, basket_size, etc.)
│     │
│     ├─ Initialize Loss (WorldModelLoss)
│     │  ├─ FocalLoss for products
│     │  ├─ ContrastiveLoss for relationships
│     │  └─ CrossEntropyLoss for auxiliary tasks
│     │
│     ├─ Initialize Datasets
│     │  ├─ WorldModelDataset(split='train')
│     │  │  ├─ Load samples.parquet
│     │  │  ├─ Load product_embeddings.npy
│     │  │  ├─ Load customer_history.npy
│     │  │  ├─ Load transactions.csv
│     │  │  └─ Build basket_products index
│     │  │
│     │  └─ WorldModelDataset(split='validation')
│     │     └─ Same as above for validation data
│     │
│     └─ Initialize Optimizer (AdamW)
│        └─ Will update model weights during training
│
├─ trainer.train() starts
│  │
│  ├─ Initial validation
│  │  └─ Check model performance before training
│  │
│  └─ FOR EACH EPOCH (1 to 20):
│     │
│     ├─ Determine phase (warmup/main/finetune)
│     │  └─ Adjust learning rate and masking
│     │
│     ├─ Create DataLoader
│     │  │
│     │  └─ WorldModelDataLoader(train_dataset, batch_size=256)
│     │     ├─ Organize data into buckets by basket size
│     │     ├─ Shuffle if training
│     │     └─ Prepare to yield batches
│     │
│     └─ FOR EACH BATCH:
│        │
│        ├─ DataLoader yields batch
│        │  │
│        │  └─ WorldModelBatch created
│        │     ├─ customer_context [256, 192]
│        │     ├─ temporal_context [256, 64]
│        │     ├─ store_context [256, 96]
│        │     ├─ trip_context [256, 48]
│        │     ├─ product_embeddings [256, 50, 256]
│        │     ├─ price_features [256, 50, 64]
│        │     ├─ attention_mask [256, 50]
│        │     ├─ masked_positions [256, 7]
│        │     ├─ masked_targets [256, 7]
│        │     └─ auxiliary_labels {dict}
│        │
│        ├─ FORWARD PASS
│        │  │
│        │  └─ model(batch) → predictions
│        │     ├─ ContextFusion: [256,400] → [256,512]
│        │     ├─ ProductFusion: [256,50,320] → [256,50,512]
│        │     ├─ MambaEncoder: [256,50,512] → [256,50,512]
│        │     ├─ TransformerDecoder: [256,50,512] → [256,50,512]
│        │     ├─ ProductHead: [256,7,512] → [256,7,5003]
│        │     └─ AuxiliaryHeads: [256,512] → {dict of predictions}
│        │
│        ├─ LOSS CALCULATION
│        │  │
│        │  └─ criterion(predictions, targets) → loss
│        │     ├─ FocalLoss: masked product prediction
│        │     ├─ ContrastiveLoss: product relationships
│        │     ├─ CrossEntropy: basket_size
│        │     ├─ CrossEntropy: price_sensitivity
│        │     ├─ CrossEntropy: mission_type
│        │     └─ Weighted sum → total_loss
│        │
│        ├─ BACKWARD PASS
│        │  │
│        │  └─ loss.backward()
│        │     └─ Calculate gradients for all 23M parameters
│        │
│        ├─ OPTIMIZER STEP
│        │  │
│        │  └─ optimizer.step()
│        │     ├─ Clip gradients (prevent explosion)
│        │     └─ Update all model weights
│        │
│        ├─ LOGGING (every 100 steps)
│        │  └─ Print: Step, Loss, Speed
│        │
│        ├─ VALIDATION (every 500 steps)
│        │  │
│        │  └─ validate()
│        │     ├─ Switch to eval mode
│        │     ├─ Run through validation data
│        │     ├─ Calculate validation loss
│        │     ├─ Compare to best loss
│        │     └─ Save checkpoint if best
│        │
│        └─ CHECKPOINT (every 2000 steps)
│           └─ Save model state to disk
│
└─ Training complete!
   └─ Save final training log
```

---

## Part 7: Key Concepts Explained

### 1. **Embeddings**
**Simple**: Converting things into numbers that capture meaning.

```python
# Products as embeddings
"milk"  → [0.23, -0.45, 0.67, ..., 0.12]  # 256 numbers
"bread" → [0.19, -0.41, 0.71, ..., 0.08]  # Similar to milk!
"hammer"→ [-0.82, 0.34, -0.15, ..., 0.91] # Very different!

# Why? Products that are bought together have similar embeddings
```

### 2. **Masking**
**Simple**: Hiding some products and asking the model to guess them.

```python
Original basket: ["milk", "bread", "eggs", "butter", "cheese"]

Masked basket:   ["milk", [MASK], "eggs", [MASK], "cheese"]
                          ↑              ↑
                    Predict these!

# This is how the model learns patterns
# "If someone buys milk and eggs, they probably bought bread and butter"
```

### 3. **Attention**
**Simple**: The model learns which products to "pay attention to" when making predictions.

```python
Predicting [MASK] in: ["milk", [MASK], "eggs"]

Attention weights:
  milk  → 0.7  # Pay lots of attention!
  [MASK]→ 0.0  # Can't look at what we're predicting
  eggs  → 0.3  # Pay some attention

# The model learns: "milk and eggs together → probably cereal!"
```

### 4. **Gradients**
**Simple**: The direction and amount to adjust each weight.

```python
# Current prediction: "hammer" (wrong!)
# True answer: "bread"
# 
# Gradient calculation:
# "If I increase weight #1 by 0.001, loss goes down"
# "If I decrease weight #2 by 0.002, loss goes down more"
# 
# Optimizer: "Okay, I'll make those changes!"
```

### 5. **Learning Rate**
**Simple**: How big of steps to take when learning.

```python
High learning rate (0.01):
  ⚡ Fast learning
  ❌ Might overshoot the best solution
  ❌ Unstable

Low learning rate (0.00001):
  ✅ Stable learning
  ✅ Precise adjustments
  ⏰ Slower to converge

Our approach: Start low (warmup), go higher (main), end low (finetune)
```

### 6. **Epochs**
**Simple**: One complete pass through all training data.

```python
Training data: 21 million shopping trips

Epoch 1: See all 21 million trips once
Epoch 2: See all 21 million trips again (in different order)
...
Epoch 20: See all 21 million trips for the 20th time

# Each time, the model gets a bit better!
```

### 7. **Validation**
**Simple**: Testing on data the model hasn't seen during training.

```python
Training data: 21 million trips (model learns from these)
Validation data: 3.8 million trips (model is tested on these)

Why separate?
  ✅ Prevents memorization
  ✅ Ensures generalization
  ✅ Detects overfitting

If training loss ↓ but validation loss ↑ → OVERFITTING!
```

---

## Part 8: Common Questions

### Q1: Why is the model so complex?
**A**: Retail shopping is complex! We need to consider:
- Customer preferences and history
- Time patterns (weekday vs weekend)
- Store characteristics
- Product relationships
- Price sensitivity
- Trip purpose

A simple model can't capture all these patterns.

### Q2: Why multiple losses?
**A**: Multi-task learning helps the model learn better representations:
- Main task: Predict products
- Auxiliary tasks: Provide additional learning signals
- Result: Better generalization and more robust predictions

### Q3: Why three training phases?
**A**: Like teaching a student:
1. **Warmup**: Gentle introduction, focus on basics
2. **Main**: Full curriculum, learn everything
3. **Finetune**: Polish and refine, harder challenges

### Q4: How long does training take?
**A**: On a good GPU (A100):
- One epoch: ~2-3 hours
- Full training (20 epochs): ~40-60 hours
- On your Mac (MPS): Much slower due to memory issues

### Q5: How do I know if training is working?
**A**: Watch these metrics:
```
Good signs:
✅ Loss decreasing over time
✅ Validation loss tracking training loss
✅ Stable training (no sudden spikes)
✅ Reasonable speed (>40 samples/sec)

Bad signs:
❌ Loss increasing or not changing
❌ Validation loss much higher than training loss
❌ Training crashes or slows down dramatically
❌ NaN or Inf in losses
```

---

## Part 9: Practical Example

Let's trace one complete example through the entire system:

### Input: A Shopping Trip
```python
{
    "customer_id": 12345,
    "basket_id": 987654,
    "store_id": 42,
    "shop_week": 200815,
    "shop_weekday": 6,  # Saturday
    "shop_hour": 14,    # 2 PM
    "products": [101, 205, 308, 412, 515],  # milk, bread, eggs, butter, cheese
    "mission_type": "Full Shop",
    "basket_size": "M",
    "price_sensitivity": "MM"
}
```

### Step 1: Dataset Prepares the Example
```python
# WorldModelDataset.__getitem__(idx)

# Encode customer
t1 = [0.23, -0.45, ..., 0.12]  # 192 dimensions

# Encode products
t2 = [
    [0.11, 0.22, ..., 0.33],  # milk embedding (256d)
    [0.14, 0.25, ..., 0.36],  # bread embedding
    [0.17, 0.28, ..., 0.39],  # eggs embedding
    [0.20, 0.31, ..., 0.42],  # butter embedding
    [0.23, 0.34, ..., 0.45],  # cheese embedding
]

# Encode time
t3 = [0.45, 0.67, ..., 0.89]  # 64 dimensions (Saturday, 2 PM, week 200815)

# Encode prices
t4 = [...]  # 64 dimensions per product

# Encode store
t5 = [0.12, 0.34, ..., 0.56]  # 96 dimensions

# Encode trip
t6 = [0.78, 0.90, ..., 0.12]  # 48 dimensions

# Apply masking (hide bread and butter)
masked_products = [101, [MASK], 308, [MASK], 515]
targets = [205, 412]  # bread, butter
```

### Step 2: DataLoader Batches It
```python
# WorldModelDataLoader groups this with 255 other similar baskets
batch = WorldModelBatch(
    customer_context=[256, 192],  # 256 customers
    product_embeddings=[256, 50, 256],  # 256 baskets, up to 50 products
    # ... all other tensors
    masked_targets=[256, 7]  # Up to 7 masked products per basket
)
```

### Step 3: Model Makes Predictions
```python
# WorldModel.forward(batch)

# Fuse context
context = [0.45, 0.67, ..., 0.89]  # 512d

# Process products
encoded = [
    [0.11, 0.22, ..., 0.33],  # milk (with context)
    [0.14, 0.25, ..., 0.36],  # [MASK] position
    [0.17, 0.28, ..., 0.39],  # eggs (with context)
    [0.20, 0.31, ..., 0.42],  # [MASK] position
    [0.23, 0.34, ..., 0.45],  # cheese (with context)
]

# Predict masked products
predictions = [
    [0.01, 0.02, ..., 0.85, ..., 0.01],  # Predicts product 205 (bread) with 85% confidence
    [0.01, 0.01, ..., 0.78, ..., 0.02],  # Predicts product 412 (butter) with 78% confidence
]

# Auxiliary predictions
aux_predictions = {
    'basket_size': [0.1, 0.7, 0.2],  # Predicts "M" with 70% confidence
    'mission_type': [0.05, 0.8, 0.1, 0.05],  # Predicts "Full Shop" with 80%
}
```

### Step 4: Loss Calculation
```python
# WorldModelLoss.forward(predictions, targets)

# Focal loss (main task)
focal_loss = -log(0.85) * (1-0.85)^2 + -log(0.78) * (1-0.78)^2
           = 0.162 + 0.242 = 0.404

# Auxiliary losses
basket_size_loss = -log(0.7) = 0.357
mission_type_loss = -log(0.8) = 0.223

# Total loss
total_loss = 0.60 * 0.404 + 0.08 * 0.357 + 0.04 * 0.223
           = 0.242 + 0.029 + 0.009 = 0.280
```

### Step 5: Backward Pass
```python
# loss.backward()

# Calculate gradients for all 23 million parameters
# Example gradients:
weight_1: gradient = -0.0023  # Decrease this weight
weight_2: gradient = +0.0015  # Increase this weight
# ... 22,999,998 more gradients
```

### Step 6: Update Weights
```python
# optimizer.step()

# Update each weight
weight_1 = weight_1 - learning_rate * gradient
         = 0.5234 - 0.00005 * (-0.0023)
         = 0.52340012  # Slightly increased

weight_2 = weight_2 - learning_rate * gradient
         = 0.7891 - 0.00005 * (+0.0015)
         = 0.78909993  # Slightly decreased

# After this update, the model is slightly better at predicting!
```

---

## Summary: The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                  TRAINING IS LIKE...                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Teaching a student to predict grocery shopping:            │
│                                                              │
│  1. DATASET: Organize millions of shopping receipts        │
│     "Here are examples of what people buy"                  │
│                                                              │
│  2. DATALOADER: Group similar examples together             │
│     "Let's study small baskets together, then large ones"   │
│                                                              │
│  3. MODEL: The student's brain                              │
│     "I'll learn patterns from these examples"               │
│                                                              │
│  4. LOSS: Measure mistakes                                  │
│     "You got 45% correct, here's what you got wrong"       │
│                                                              │
│  5. OPTIMIZER: Adjust the brain                             │
│     "Let me tweak my thinking to do better next time"       │
│                                                              │
│  6. REPEAT: Do this millions of times                       │
│     "Practice makes perfect!"                                │
│                                                              │
│  RESULT: A model that can predict shopping behavior!        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Read the code**: Start with `main()` in `train.py` and follow the flow
2. **Run a small test**: Try training on 10,000 samples to see it work
3. **Monitor training**: Watch the losses decrease over time
4. **Experiment**: Try different hyperparameters and see what happens
5. **Deploy**: Use the trained model to make predictions on new data

Remember: Machine learning is an iterative process. Don't expect perfection on the first try. Experiment, learn, and improve!