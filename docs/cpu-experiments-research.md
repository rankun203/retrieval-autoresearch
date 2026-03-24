# CPU-Only Experiment Research

## Key Finding
PyTerrier's official Robust04 baselines: **DPH+KL = MAP@1000=0.2857** vs BM25+Bo1=0.2795. DPH is parameter-free.

## CPU Experiment Queue (priority order)

### 1. DPH/InL2/DirichletLM + PRF sweep (~2h)
- DPH is parameter-free, often beats BM25 on Robust04
- Test: DPH, InL2, DirichletLM, Hiemstra_LM
- Each with Bo1/RM3/KL PRF variants
- No re-indexing needed
- Expected: +5-15% on sparse component

### 2. Multi-system sparse fusion (~30min)
- Fuse best DPH + best BM25 runs
- CombSUM/CombMNZ
- Then fuse with dense for 3-way fusion
- Expected: +2-5% on overall MAP

### 3. Field-aware indexing: BM25F (~2h)
- Re-index with title/body fields
- BM25F with title weight > body weight
- Needs separate field index
- Expected: +2-8% on sparse

### 4. SDM proximity matching (~3h)
- Sequential Dependency Model
- Needs blocks index (positional info)
- Helps multi-term queries
- Expected: +2-5% on sparse

### 5. Learning to Rank with XGBoost (~3h)
- Combine features from multiple systems
- MUST use 5-fold CV to avoid data leakage
- Highest ceiling but most complex
- Expected: +5-15% on final
