# Tensor Preparation Sequence Diagram

## Overview
This document shows the detailed sequence of operations for tensor preparation in RetailSim, starting from line 269 in the main function of `run_tensor_preparation.py`.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Main as main()
    participant Dataset as RetailSimDataset
    participant Load as _load_features()
    participant Init as _init_encoders()
    participant Prep as _prepare_data()
    participant Files as File System
    participant Enc1 as CustomerContextEncoder
    participant Enc2 as ProductSequenceEncoder
    participant Enc3 as TemporalContextEncoder
    participant Enc4 as PriceContextEncoder
    participant Enc5 as StoreContextEncoder
    participant Enc6 as TripContextEncoder

    Main->>Dataset: new RetailSimDataset(project_root)
    activate Dataset

    Note over Dataset: __init__ method starts
    Dataset->>Load: self._load_features()
    activate Load

    Note over Load: Loading pre-computed features
    Load->>Files: open('product_embeddings.pkl', 'rb')
    Files-->>Load: pickle data with embeddings
    Load->>Files: read_parquet('customer_embeddings.parquet')
    Files-->>Load: customer DataFrame
    Load->>Files: read_parquet('store_features.parquet')
    Files-->>Load: store DataFrame
    Load->>Files: read_parquet('price_features.parquet')
    Files-->>Load: price DataFrame
    Load->>Files: read_parquet('customer_mission_patterns.parquet')
    Files-->>Load: mission patterns DataFrame
    
    Load-->>Dataset: All features loaded
    deactivate Load

    Dataset->>Init: self._init_encoders()
    activate Init

    Note over Init: Initialize all 6 tensor encoders
    Init->>Enc1: CustomerContextEncoder()
    Init->>Enc1: Build customer_embed_dict from DataFrame
    Init->>Enc1: Build trip_counts mapping
    
    Init->>Enc2: ProductSequenceEncoder(product_embeddings, max_seq_len, mask_prob)
    Init->>Enc3: TemporalContextEncoder()
    Init->>Enc4: PriceContextEncoder()
    Init->>Enc5: StoreContextEncoder()
    
    Note over Init: Build price lookup dictionary
    Init->>Init: self._build_price_lookup()
    Note over Init: Iterate price_features_df to build price_lookup and category_avg_lookup
    
    Note over Init: Pre-encode store contexts
    Init->>Init: self._build_store_lookup()
    Init->>Enc5: encode_batch(store_metadata, store_features)
    Enc5-->>Init: Pre-encoded store tensors
    
    Init->>Enc6: TripContextEncoder()
    
    Init-->>Dataset: All encoders initialized
    deactivate Init

    Dataset->>Prep: self._prepare_data()
    activate Prep

    Note over Prep: Load and process transaction data
    Prep->>Files: read_csv('transactions.csv', nrows=10000)
    Files-->>Prep: transactions DataFrame
    
    Note over Prep: Group transactions by basket
    Prep->>Prep: groupby('BASKET_ID').agg({...})
    
    Note over Prep: Build customer last visit lookup
    Prep->>Prep: self._build_last_visit_lookup()
    Note over Prep: Sort by customer and week, build prior visit mapping
    
    Prep-->>Dataset: Data preparation complete
    deactivate Prep

    Dataset-->>Main: RetailSimDataset instance ready
    deactivate Dataset

    Note over Main: Dataset is now ready for tensor generation
    Main->>Dataset: len(dataset)
    Dataset-->>Main: Returns number of baskets

    Note over Main: Test single sample encoding
    Main->>Dataset: dataset[0]
    activate Dataset
    Dataset->>Dataset: self._encode_sample(basket_row)
    
    Note over Dataset: Encoding single basket
    Dataset->>Enc1: encode_customer(customer_id, seg1, seg2, history_embed, num_trips)
    Enc1-->>Dataset: t1 tensor [192d]
    
    Dataset->>Enc2: encode_sequence(products, add_eos=True, apply_masking=False)
    Enc2-->>Dataset: embeddings, token_ids, length, masked_positions, masked_targets
    
    Dataset->>Enc3: encode_temporal(shop_week, weekday, hour, last_visit_week)
    Enc3-->>Dataset: t3 tensor [64d]
    
    Dataset->>Enc4: encode_basket_prices(products, price_lookup, category_avg_lookup)
    Enc4-->>Dataset: t4 tensor [64d per item]
    
    Dataset->>Enc5: Get pre-encoded store tensor
    Enc5-->>Dataset: t5 tensor [96d]
    
    Dataset->>Enc6: encode_trip(mission_type, mission_focus, price_sensitivity, basket_size)
    Enc6-->>Dataset: t6 tensor [48d]
    
    Dataset-->>Main: Dictionary with all 6 tensors
    deactivate Dataset

    Note over Main: Test batch encoding
    Main->>Dataset: dataset.get_batch([0,1,2,3], apply_masking=True)
    activate Dataset
    
    Note over Dataset: Batch processing - calls each encoder in batch mode
    Dataset->>Enc1: Batch encode customer contexts
    Dataset->>Enc2: encode_batch(product_lists, apply_masking=True)
    Dataset->>Enc3: Batch encode temporal contexts
    Dataset->>Enc4: encode_batch(products, price_lookup, category_avg_lookup)
    Dataset->>Enc5: Collect pre-encoded store tensors
    Dataset->>Enc6: Batch encode trip contexts + generate labels
    
    Dataset-->>Main: RetailSimBatch with all tensors [B, S, D]
    deactivate Dataset

    Note over Main: DataLoader iteration
    loop For each batch
        Main->>Dataset: dataloader.__iter__()
        Dataset->>Dataset: Get batch indices
        Dataset->>Dataset: dataset.get_batch(indices, apply_masking=True)
        Dataset-->>Main: RetailSimBatch
    end
```

## Key Details of the Tensor Preparation Process

### 1. **Feature Loading Phase** (`_load_features`)
- Loads 5 pre-computed feature files from `data/features/`
- **Product embeddings**: 256d per product from Layer 3 graph embeddings
- **Customer embeddings**: 160d per customer from Layer 4 history encoding
- **Store features**: 96d per store from Layer 5 context encoding
- **Price features**: 64d per price observation from Layer 2 Fourier encoding
- **Mission patterns**: Customer trip patterns from Stage 4 data pipeline

### 2. **Encoder Initialization** (`_init_encoders`)
- Creates 6 encoder instances for T1-T6 tensors
- **T1 - CustomerContextEncoder**: Combines segments, history, and affinity
- **T2 - ProductSequenceEncoder**: Handles product sequences with masking
- **T3 - TemporalContextEncoder**: Encodes time patterns and visit frequency
- **T4 - PriceContextEncoder**: Processes price features per item
- **T5 - StoreContextEncoder**: Pre-encodes all store contexts for efficiency
- **T6 - TripContextEncoder**: Encodes mission types and generates labels

### 3. **Data Preparation** (`_prepare_data`)
- Loads raw transactions (limited to 10,000 rows by default for testing)
- Groups by `BASKET_ID` to create shopping trips
- Builds customer visit history for temporal features
- Creates basket-level aggregations for trip context

### 4. **Tensor Generation Process**

#### **Single Sample Encoding** (`dataset[0]`)
- **T1 (Customer Context)**: 192d = segments(32) + history(160) + affinity(0)
- **T2 (Product Sequence)**: 256d per item from pre-trained embeddings
- **T3 (Temporal Context)**: 64d = week(32) + weekday(16) + hour(16)
- **T4 (Price Context)**: 64d per item = fourier(32) + log(8) + relative(16) + velocity(8)
- **T5 (Store Context)**: 96d = format(32) + region(32) + operational(32) + identity(0)
- **T6 (Trip Context)**: 48d = mission_type(16) + mission_focus(16) + price_sensitivity(8) + basket_size(8)

#### **Batch Processing** (`dataset.get_batch()`)
- Processes multiple baskets simultaneously
- Applies BERT-style masking to product sequences when requested
- Generates auxiliary labels for trip prediction tasks
- Returns `RetailSimBatch` object with:
  - Dense context: [B, 400] (concatenated T1+T3+T5+T6)
  - Sequence features: [B, S, 320] (concatenated T2+T4)
  - Attention masks and sequence lengths
  - Trip labels for supervised learning

### 5. **DataLoader Integration**
- Provides batched iteration over the dataset
- Supports shuffling and masking for training
- Handles variable-length sequences with padding
- Generates training-ready tensors for model consumption

## Memory and Performance Considerations

### **Memory Optimization**
- Pre-encoding store contexts avoids repeated computation
- Lazy loading of features only when needed
- Efficient data structures (dictionaries for lookups)

### **Performance Features**
- Batch processing for GPU utilization
- Vectorized operations within encoders
- Attention mechanisms for sequence handling
- Masking strategy for self-supervised learning

## Output Tensors Summary

| Tensor | Dimension | Type | Description |
|--------|-----------|------|-------------|
| T1 | 192d | Dense | Customer context per transaction |
| T2 | 256d/item | Sequence | Product embeddings per basket item |
| T3 | 64d | Dense | Temporal context per transaction |
| T4 | 64d/item | Sequence | Price features per basket item |
| T5 | 96d | Dense | Store context per transaction |
| T6 | 48d | Dense | Trip context per transaction |

**Total Dense Context**: 400d per transaction  
**Total Sequence Features**: 320d per basket item

This tensor preparation pipeline efficiently combines all feature engineering outputs into model-ready tensors for the RetailSim world model.
