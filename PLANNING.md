# Tennis ML — Planning & Design Notes

Core ideas on model strategy, feature engineering, and deep learning integration
for tennis match prediction.

---

## 1. Model Architecture Strategy

Compare models across three fundamentally different classes, pick the best from
each, then ensemble the winners.

```
                       80k matches
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │  Linear  │ │   GBDT   │ │ Neural Net   │
        │  Models  │ │  Models  │ │  Models      │
        ├──────────┤ ├──────────┤ ├──────────────┤
        │ • LR     │ │ • XGBoost│ │ • MLP        │
        │ • SVM    │ │ • LGBM   │ │ • LSTM       │
        │ • NB     │ │ • CatBoost│ │ • GRU        │
        └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │            │              │
             ▼            ▼              ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ Best     │ │ Best     │ │ Best         │
        │ Linear   │ │ GBDT     │ │ Neural Net   │
        └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │            │              │
             └────────────┼──────────────┘
                          ▼
                  ┌───────────────┐
                  │   Ensemble    │
                  │  (stacking)   │
                  └───────────────┘
```

No unsupervised models (kNN, clustering, etc.) — they don't apply here for
match prediction. The target is fully supervised (binary: win/loss), so
there's nothing for unsupervised methods to learn that supervised methods
can't capture better.

That said, **player similarity / recommendation** is a separate problem
(no target label) where content-based and nearest-neighbor approaches are
the right tool — covered in a later section.

### Class 1 — Linear Models (pairwise static features)

Classic supervised linear models. Simple, fast, interpretable baselines.

| Model | Why include |
|---|---|
| **LogisticRegression** | Strong baseline, interpretable weights, fast to train |
| **SVM** | Max-margin decision boundary, works well with RBF kernel for non-linear patterns |
| **NaiveBayes** | Surprisingly decent baseline for binary classification |

All trained on pairwise static features (player_A + player_B attributes
side by side). No sequence input — these models have no notion of order.

**Scaling required:** StandardScaler or RobustScaler. Unlike GBDT, linear
models assume features are on similar scales.

Best linear model advances to the ensemble round.

Tuned independently via Optuna. Best model advances to the ensemble round.

### Class 2 — GBDT Models (pairwise static features)

| Model | Why include | Key hyperparams |
|---|---|---|
| **XGBoost** | Industry standard, mature, well-tuned defaults | `n_estimators`, `max_depth`, `lr`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| **LightGBM** | Faster than XGBoost, leaf-wise growth, handles categoricals natively | `n_estimators`, `num_leaves`, `lr`, `subsample`, `feature_fraction`, `reg_alpha`, `reg_lambda` |
| **CatBoost** | Best categorical handling, symmetric trees, less tuning | `iterations`, `depth`, `lr`, `l2_leaf_reg` |

GBDT gets pairwise structure at the **feature level** — each row contains both
players' attributes side by side, plus engineered cross features:

```
rank_A, rank_B, rank_diff, rank_ratio,
age_A, age_B, age_diff,
ace_rate_A, ace_rate_B, ace_rate_diff,
bio_embedding_A_1..10, bio_embedding_B_1..10, bio_diff_1..10,
style_serve_A, style_baseline_A, style_serve_B, style_baseline_B,
surface_win_rate_A, surface_win_rate_B, surface_win_rate_diff,
...
```

No scaling needed — tree-based models are invariant to feature scale. Handle
missing values natively.

The tree finds pairwise splits naturally:
```
if rank_diff > 50 and ace_rate_A > 0.15 and clay_win_rate_B < 0.5:
    predict Player A wins
```

This is functionally a pairwise comparison, just at the raw feature level
rather than through a learned embedding like the NN. Both model classes
have pairwise structure — GBDT through engineered differences, NN through
the two-tower embedding + pairwise head.

Best GBDT model advances to the ensemble round.

### Class 3 — Neural Network Models (two-tower with full player embeddings)

Each tower produces a **full player embedding** — combining form sequence
encoding with static player attributes (rank, age, style, bio embeddings).
Then pairwise comparison on top of the full embeddings.

```
Two-tower architecture (shared by all NN encoders):

                    Player A                                    Player B
              ┌──────────────────┐                      ┌──────────────────┐
              │ Static features  │                      │ Static features  │
              │ (rank, age,      │                      │ (rank, age,      │
              │  style, bio, ...)│                      │  style, bio, ...)│
              └────────┬─────────┘                      └────────┬─────────┘
                       │                                         │
              ┌────────▼─────────┐                      ┌────────▼─────────┐
              │ Dense (ReLU)     │                      │ Dense (ReLU)     │
              │ static → 32 dim  │                      │ static → 32 dim  │
              └────────┬─────────┘                      └────────┬─────────┘
                       │                                         │
┌──────────────────────┼──────────────────┐   ┌──────────────────┼──────────────────────┐
│ Sequence features    │                  │   │ Sequence features │                      │
│ [match_1..match_10]  │                  │   │ [match_1..match_10]                      │
│  per-match stats     │                  │   │  per-match stats  │                      │
│  (100 features)      │                  │   │  (100 features)   │                      │
└────────┬─────────────┘                  │   └────────┬──────────┘                      │
         │                                │            │                                 │
   ┌─────▼──────┐                         │      ┌─────▼──────┐                          │
   │  Encoder   │                         │      │  Encoder   │                          │
   │ (LSTM/GRU  │                         │      │ (LSTM/GRU  │                          │
   │  /TCN)     │                         │      │  /TCN)     │                          │
   └─────┬──────┘                         │      └─────┬──────┘                          │
         │                                │            │                                 │
         └──────────┬─────────────────────┘            └──────────┬───────────────────────┘
                    │                                           │
                    ▼                                           ▼
            ┌────────────────┐                          ┌────────────────┐
            │ emb_seq_a      │                          │ emb_seq_b      │
            │ (hidden_dim)   │                          │ (hidden_dim)   │
            └────────┬───────┘                          └────────┬───────┘
                     │                                           │
                     ▼                                           ▼
            ┌────────────────┐                          ┌────────────────┐
            │ Full player    │                          │ Full player    │
            │ embedding:     │                          │ embedding:     │
            │ concat(        │                          │ concat(        │
            │  emb_seq(h_d), │                          │  emb_seq(h_d), │
            │  static(s_d)   │                          │  static(s_d)   │
            │  → (h_d + s_d) │                          │  → (h_d + s_d) │
            └────────┬───────┘                          └────────┬───────┘
                    │                                           │
                    └──────────────────┬────────────────────────┘
                                       │
                                 ┌─────┴──────────┐
                                 │  Pairwise head  │
                                 │  concat(a, b,   │
                                 │  a-b, a*b)      │
                                 │  → Dense →      │
                                 │  Sigmoid        │
                                 └─────────────────┘
```

**Two-tower embedding design:**

Each tower has two input pathways that merge into a single player embedding:

| Pathway | Input | Encoder | Output dim |
|---|---|---|---|---|
| **Sequence** | Last N matches per player (form trajectory) | LSTM / GRU / TCN — `hidden_size` IS the embedding, no separate projection | `hidden_dim` — tuned by Optuna (64–256) |
| **Static** | Player attributes: rank, age, style vector, bio embedding, handedness, height, etc. | Dense(ReLU) — projects arbitrary static features to a fixed dim | `static_dim` — tuned by Optuna (16–64) |
| **Full embedding** | `concat(seq_embedding, static_embedding)` | — | `hidden_dim + static_dim` |

**Key point:** the LSTM's hidden_size IS the sequence embedding dimension.
There is no separate projection layer:

```python
# hidden_dim = 128 means the embedding is 128-dimensional
lstm = nn.LSTM(input_size=100, hidden_size=hidden_dim, batch_first=True)
# h_n[-1] shape: (batch, hidden_dim) — this IS the player's sequence embedding
```

The final hidden state `h_n[-1]` is directly passed to the pairwise head.
No additional linear layer compresses it further. The same applies to GRU
and TCN — the encoder's output dimension is its `hidden_size` parameter.

The full player embedding captures both *who the player is* (static) and
*how they're playing right now* (sequence). This means the pairwise head
can learn interactions between a player's identity and their form — e.g.,
"an aggressive baseliner on a hot streak is more dangerous than a
counterpuncher on a hot streak."

**Pairwise comparison layer — why `concat(a, b, a-b, a*b)`:**

After each tower produces a full player embedding, the comparison head
combines them in a structured way:

| Component | What it encodes | Tennis example |
|---|---|---|
| `emb_a`, `emb_b` | Absolute player profile | "Djokovic is a top-5 aggressive baseliner in peak form" |
| `emb_a - emb_b` | Directional gap and magnitude | "Sinner's full profile is 0.3 units better than Rublev's" |
| `emb_a * emb_b` | Interaction / level of play | "When two serve-and-volleyers meet on grass, serve dominance amplifies" |

The difference alone misses the *absolute level* — two weak players with a
small gap are different from two strong ones with a small gap. The product
alone misses which player is favored. Together, the MLP can learn:

```python
# The MLP's Dense layer can learn non-linear combinations like:
if emb_a is high and emb_b is high and (emb_a - emb_b) is small:
    prediction ≈ 0.5          # evenly matched top players
if emb_a * emb_b is low and (emb_a - emb_b) is large:
    prediction ≈ 1.0          # one strong, one weak
```

This architecture is standard in pairwise comparison problems — it's how
TrueSkill, Bayesian skill rating, and production recommendation systems
model head-to-head outcomes. You're learning a similarity function over
player embeddings rather than a direct decision boundary.

**Classifier head — what happens after the concat:**

The `concat(a, b, a-b, a*b)` output feeds into a small MLP classifier.
This is where the actual compression and decision happens:

```python
# Input: concat(a, b, a-b, a*b) — shape (batch, (hidden_dim + static_dim) * 4)
# With hidden_dim=128, static_dim=32: (batch, (128+32)*4) = (batch, 640)

self.classifier = nn.Sequential(
    nn.Linear((hidden_dim + static_dim) * 4, 64),   # 640 → 64 — compress pairwise info
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(64, 1),                                # 64 → 1 — final logit
    # No sigmoid here — use BCEWithLogitsLoss which fuses sigmoid + BCE
)
```

The 640-dim pairwise vector is compressed to 64, then to 1 logit. The
intermediate 64-dim layer captures non-linear combinations of the
embedding differences — e.g., "emb_a is high AND (emb_a * emb_b) is
high AND (emb_a - emb_b) is small" → near-even match.

**Full forward pass summary:**

```python
# Player A tower
emb_a = encode_player(seq_a, lengths_a, static_a)
# emb_a = concat(LSTM(seq_a), Dense(static_a))
# shape: (batch, hidden_dim + static_dim), e.g. (256, 160)

# Player B tower (same encoder, shared weights)
emb_b = encode_player(seq_b, lengths_b, static_b)

# Pairwise comparison
combined = torch.cat([emb_a, emb_b, emb_a - emb_b, emb_a * emb_b], dim=-1)
# shape: (256, 640)

# Classifier → logit
logits = self.classifier(combined)  # (256, 1)

# Loss (done externally, not in forward):
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

The LSTM's `hidden_size` IS the sequence embedding dimension — no
additional projection layer between the LSTM output and the concat.
If `hidden_dim=128`, the sequence embedding is 128-dimensional, passed
directly into the pairwise comparison.

| Encoder | Why include | Input shape |
|---|---|---|
| **LSTM** | Proven for sequence modeling, handles variable-length via packing | `(batch, seq, features)` |
| **GRU** | Same as LSTM, fewer params, equivalent performance at this sequence length | `(batch, seq, features)` |
| **TCN** | Parallel convolution, trains faster, needs more data (80k is enough) | `(batch, features, seq)` — transpose required |

The encoder is selected as an Optuna categorical:

```python
encoder_type = trial.suggest_categorical("encoder", ["lstm", "gru", "tcn"])
```

**Input shapes:**
- Sequence: `(batch, seq_len, num_features)` — e.g. `(256, 10, 100)`
- Static: `(batch, num_static_features)` — e.g. `(256, 20)`

**Handling variable-length histories:**

Players have different numbers of past matches. New players may have only a few.

**LSTM/GRU** uses post-padding + `pack_padded_sequence`:

```python
# X shape: (batch, padded_seq_len, num_features)
# lengths: [20, 14, 20, 7, 10, ...]  — real match count per player
packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
output, (h_n, c_n) = lstm(packed)  # LSTM never sees padding
```

```
Player A: [m1, m2, m3, m4, m5, m6, m7, 0, 0, 0]   lengths=7
Player B: [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]  lengths=10
Player C: [m1, m2, m3, 0, 0, 0, 0, 0, 0, 0]        lengths=3
```

**TCN** uses attention masking instead (conv1d can't skip timesteps):

```python
mask = (X != 0).any(dim=-1)                    # (batch, seq) — True where real
tcn_out = tcn(X.permute(0, 2, 1))               # (batch, feat, seq)
tcn_out = tcn_out * mask.unsqueeze(1)           # zero out padded garbage
embedding = tcn_out[:, :, -1]                   # take last timestep
```

**Why this replaces the need for GBDT on static features:**

With this architecture, the neural network class now handles **both sequence
and static features** end-to-end. The pairwise head learns interactions
between form trajectory and player identity directly. This means:

- The NN model is no longer limited to sequences — it's a full player
  comparison model
- GBDT remains as a separate model class (pure tabular, no sequences)
- The stacking ensemble can compare: pure GBDT vs full two-tower NN vs
  both stacked — letting the data decide whether the NN's end-to-end
  embeddings add value over GBDT's tree-based approach

### Ensembling — Stacking the winners

Once the best model from each class is identified, combine them via a small
meta-model:

```python
# Base model predictions (each trained on same train/test split)
proba_lr     = best_linear.predict_proba(X_test_static)[:, 1]
proba_gbdt   = best_gbdt.predict_proba(X_test_static)[:, 1]
proba_seq    = best_nn.predict_proba(X_test_seq)[:, 1]

# Meta-model: logistic regression on the 3 probability scores
stack_features = np.column_stack([proba_lr, proba_gbdt, proba_seq])
meta = LogisticRegression().fit(stack_features, y_test)
```

This gives interpretable weights — the meta-coefficient tells you how much
each model class contributes.

If one class dominates (e.g., GBDT gets 0.8 weight while linear gets 0.1),
you can drop the weak class and simplify to a two-model ensemble.

---

## 1.5 Player Similarity — Content-Based Recommendation

A separate feature from match prediction — given a player, find the top 5
most similar players by playing style. This is an **unsupervised retrieval
problem**, not a classification task.

### Approach: bio embeddings + FAISS

Encode player bios using a sentence-transformer, combine with engineered
style features, then nearest-neighbor search via FAISS:

```python
from sentence_transformers import SentenceTransformer
import faiss

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
play_style = encoder.encode([
    "aggressive baseliner, powerful forehand, strong return",
    "serve-and-volleyer, prefers grass, quick at net",
    ...
])  # (n_players, 384)

# Combine with engineered style features (ace_rate, first_serve_pct, etc.)
style_features = np.column_stack([play_style, ace_rate, first_serve_pct, ...])
faiss.normalize_L2(style_features)

# Build index
index = faiss.IndexFlatIP(style_features.shape[1])
index.add(style_features)

# Query
query_vec = style_features[player_index["Novak Djokovic"]].reshape(1, -1)
scores, indices = index.search(query_vec, k=6)
top_5 = player_names[indices[0][1:]]  # skip self-match
```

No training needed — runs immediately with any bio corpus. Update the index
whenever new player bios are added.

---

### 2.1 Statistical Encoding in SQL (primary approach)

All feature engineering lives in BigQuery SQL during the gold layer transform.
No pandas feature transforms — keep it in SQL.

**Player skill (target encoding):**

```sql
SAFE_DIVIDE(
    SUM(match_won) OVER (
        PARTITION BY player
        ORDER BY match_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ),
    COUNT(*) OVER (
        PARTITION BY player
        ORDER BY match_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
) AS player_skill
```

Also `opponent_skill`. The difference `player_skill - opponent_skill` is
likely the single strongest predictor.

**Form features (short-term deviation from baseline):**

These capture recent performance relative to the player's own baseline —
hot streaks, slumps, fatigue, and quality of opposition.

```sql
-- Trend: recent momentum (positive = on a hot streak)
wins_last_5 / 5.0 - 0.5 * wins_last_10 / nullIf(matches_last_10, 0) AS form_momentum,

-- Weighted form: exponential decay, recent matches matter more
SUM(match_won * pow(0.9, row_num)) / SUM(pow(0.9, row_num)) OVER (
    PARTITION BY player ORDER BY match_date
    ROWS BETWEEN 19 PRECEDING AND 1 PRECEDING
) AS weighted_form,

-- Quality of opposition in recent matches
AVG(opponent_ranking) OVER (
    PARTITION BY player ORDER BY match_date
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
) AS avg_opponent_rank_last_10,

-- Surface-specific recent form (clay form != hard form)
SAFE_DIVIDE(
    SUM(match_won) OVER (
        PARTITION BY player, surface
        ORDER BY match_date
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    ),
    COUNT(*) OVER (
        PARTITION BY player, surface
        ORDER BY match_date
        ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
    )
) AS surface_win_rate_last_10,

-- Fatigue proxy: match density
COUNT(*) OVER (
    PARTITION BY player
    ORDER BY match_date
    ROWS BETWEEN 29 PRECEDING AND 1 PRECEDING
) AS matches_last_30_days,

-- Rest: time since last match
dateDiff('day', LAG(match_date) OVER (
    PARTITION BY player ORDER BY match_date
), match_date) AS days_since_last_match,
```

**Style features (rolled up per player + surface):**

```sql
AVG(ace_rate) OVER (
    PARTITION BY player
    ORDER BY match_date
    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
) AS player_avg_ace_rate,                -- serving aggressiveness

AVG(double_fault_rate) OVER (...) AS player_avg_df_rate,  -- serving risk

AVG(first_serve_pct) OVER (...) AS player_avg_first_serve, -- serving caution

AVG(clay_win_rate) OVER (...) AS player_clay_skill,
AVG(grass_win_rate) OVER (...) AS player_grass_skill,
AVG(hard_win_rate) OVER (...) AS player_hard_skill,

AVG(break_points_converted_pct) OVER (...) AS player_clutch, -- break point conversion
```

These capture a player's identity — serve-heavy, clay specialist, clutch, etc.
All computed as rolling windows to avoid data leakage.

**Sequence-form features (for LSTM):**

```sql
-- Each row becomes one step in the sequence
SELECT
    match_date,
    player,
    opponent,
    player_ranking,
    opponent_ranking,
    ace_rate,
    double_fault_rate,
    first_serve_pct,
    -- ... same features for opponent perspective
    match_won AS label
FROM gold.match_features
ORDER BY player, match_date
```

Export as a table: `(player, match_date, features..., label)` ordered by date.
The LSTM reads the last N rows per player, padding shorter histories with
zeros and using `pack_padded_sequence` to avoid processing padding.

### 2.2 Bio Embeddings (sentence-transformers)

If player bios are available (ATP website, Wikipedia, etc.), encode playing style
directly from text:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
bios = [
    "aggressive baseliner, powerful forehand, strong return of serve",
    "serve-and-volleyer, prefers grass courts, quick net reactions",
    "defensive counterpuncher, exceptional speed, clay court specialist",
    ...
]
embeddings = encoder.encode(bios)  # (n_players, 384)

# Reduce dimensionality for xgboost
from sklearn.decomposition import PCA
style_features = PCA(n_components=10).fit_transform(embeddings)
```

- Captures playing *style* in a way aggregates cannot — a new player's bio
  says they're aggressive even before they've played a match
- No cold-start problem: any new bio can be embedded immediately
- Add as 8-16 numeric columns to xgboost

### 2.3 When to Use What

| Feature Type | Where computed | Features | Target Model |
|---|---|---|---|---|
| Rolling stats (skill) | ClickHouse SQL | player_skill, ranking delta, ace_rate, surface win rates, etc. | xgboost |
| Form stats (recent deviation) | ClickHouse SQL | form_momentum, weighted_form, avg_opponent_rank_last_10, surface_win_rate_last_10, matches_last_30_days, days_since_last_match | xgboost |
| Form sequences | ClickHouse SQL | Match history ordered by date per player | LSTM (two-tower) |
| Bio embeddings | Python (pre-training) | 8-16 dim style vectors | xgboost |

---

## 3. PyTorch Integration

### When to add PyTorch

The current pipeline uses xgboost/lightgbm via Papermill + MLflow. Add PyTorch
only when you need something gradient boosting can't do:

1. **LSTM for form sequences** — most impactful addition
2. **Forecasting (time series)** — predict player ranking trajectories, career arc

### Training flow with PyTorch

```
notebooks/parameters/01_feature_engineering.ipynb   ← Preprocessing, feature engineering, bio embeddings, data export, train/test split
notebooks/parameters/02_tune_linear.ipynb   ← Optuna: LR, SVM, NB on static features
notebooks/parameters/02_tune_gbdt.ipynb     ← Optuna: XGBoost, LightGBM, CatBoost
notebooks/parameters/02_tune_nn.ipynb       ← Optuna: encoder as cat (LSTM/GRU/TCN)
notebooks/parameters/03_pick_best.ipynb     ← Select best per class
notebooks/parameters/04_stack_ensemble.ipynb ← Meta-model on 3 best predictions, final training on holdout set
notebooks/parameters/05_evaluate.ipynb      ← SHAP, ROC AUC, error analysis, final report, compare against production model and decide to promote or not
```

The first three can run in parallel since they're independent. After all three
finish, `02_pick_winners` compares them and `03_stack_ensemble` builds the
meta-model.

New notebooks follow the same pattern: Papermill takes params, runs Optuna,
logs to MLflow.

### Optuna with PyTorch

Optuna works well for neural net hyperparameters:

```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = build_lstm(hidden_dim, dropout, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(200):
        loss = train_epoch(model, optimizer)
        val_loss = validate(model)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss
```

Use `MedianPruner` to stop bad hyperparameter combos early.

### MLflow logging in notebooks

Same pattern as xgboost notebooks — log inline:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(trial.params)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_acc)

    # Save model
    mlflow.pytorch.log_model(model, "lstm_model")
```

BentoML can load PyTorch models via MLflow's pyfunc flavor or directly via
`bentoml.pytorch.get_model()`.

---

## 5. Key Decisions

| Decision | Rationale |
|---|---|---|
| Feature engineering in SQL, not pandas | Single source of truth, no pandas transforms to maintain, all logic in gold layer |
| Bio embeddings as pre-computed features | One-time cost, zero inference overhead, handles new players |
| Best-in-class comparison (linear / GBDT / NN) | Each class has a different inductive bias; the best from each is structurally different, making the final ensemble more valuable than picking a single winner |
| Encoder as Optuna hyperparameter (LSTM / GRU / TCN) | Same data pipeline, same two-tower architecture, swap only the encoder module. Fair comparison, single notebook. |
| MLflow for all model logging | Works with sklearn, xgboost, lightgbm, and PyTorch; BentoML loads via `mlflow.get()` |
| Optuna for all hyperparameter tuning | Consistent API across sklearn, xgboost, lightgbm, and PyTorch |
| No TabNet / TabTransformer | Overkill for this data size and feature type; rarely beats xgboost on tabular |
| No unsupervised models (kNN, clustering) | Target is fully supervised binary classification; unsupervised methods add nothing |

---

## 6. Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ ETL                                                              │
│ raw CSV → bronze → gold (rolling stats, style features)          │
│            → form_sequences (for NN)                              │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│ Pre-computation (Python)                                          │
│ Bio embeddings → (n_players, 10) matrix                           │
│ (one-time, saved as Parquet)                                      │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│ Model Class Tuning (Papermill + Optuna, runs in parallel)        │
│                                                                  │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐ │
│ │ Linear         │ │ GBDT           │ │ Neural Net             │ │
│ │ LR, SVM, NB    │ │ XGB, LGBM, Cat │ │ LSTM, GRU, TCN        │ │
│ └───────┬────────┘ └───────┬────────┘ └───────────┬────────────┘ │
└─────────┼──────────────────┼──────────────────────┼──────────────┘
          │                  │                      │
          └──────────────────┼──────────────────────┘
                             ▼
          ┌──────────────────────────────────────────┐
          │ Pick Winners & Train Finals              │
          │ Best linear, best GBDT, best NN          │
          └───────────────────┬──────────────────────┘
                              │
                              ▼
          ┌──────────────────────────────────────────┐
          │ Stack Ensemble                           │
          │ Meta-model on 3 probability scores       │
          └───────────────────┬──────────────────────┘
                              │
                              ▼
          ┌──────────────────────────────────────────┐
          │ Evaluate (SHAP, error analysis)          │
          └───────────────────┬──────────────────────┘
                              │
                              ▼
          ┌──────────────────────────────────────────┐
          │ Serving (BentoML)                        │
          │ Load from MLflow → REST API → k3d        │
          └──────────────────────────────────────────┘
```
