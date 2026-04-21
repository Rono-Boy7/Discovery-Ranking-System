# Discovery Ranking System

A two-stage discovery ranking project built on **MIND-small**, inspired by real-world recommendation and content discovery systems used in products like Pinterest, Quizlet, and other large-scale personalization platforms.

This project explores how to move from a simple lexical retriever to a more realistic multi-stage ranking pipeline:

- **Stage 1:** candidate retrieval
- **Stage 2:** reranking
- **Evaluation:** grouped offline ranking metrics such as **MRR@K**, **Recall@K**, and **NDCG@K**

The project intentionally includes both a **strong classical baseline** and a **neural baseline**, because good ML engineering is not about assuming “neural = better.” It is about building fair baselines, measuring carefully, and reporting honest results.

---

## Project Goal

The original goal of this project was to build a discovery ranking system that mirrors the shape of real recommendation and retrieval systems:

- candidate generation / retrieval
- reranking
- ranking metrics
- experiment tracking
- reproducible training pipelines

Rather than jumping straight into a large and complex neural system, this repo was built in layers:

1. **Data pipeline**
2. **TF-IDF retrieval baseline**
3. **Feature-based reranker**
4. **Neural two-tower retriever**
5. **Head-to-head retrieval comparison**

That progression matters. In real ML systems, strong baselines are essential because they tell you whether your more complex model is actually helping.

---

## Dataset

This project uses **MIND-small**, a news recommendation dataset designed for personalized content ranking.

Why MIND-small works well for this project:

- it contains **user histories**
- it contains **impression groups**
- each impression contains **clicked and non-clicked candidate articles**
- it provides article metadata such as:
  - category
  - subcategory
  - title
  - abstract

That structure makes it a good fit for discovery ranking because the task is not just “predict a class,” but:

> given a user’s recent reading history, rank a candidate set of items by likely relevance or click probability.

---

## What Was Built

### 1. Raw data parsing and preprocessing

The raw MIND files were parsed and normalized into reusable interim artifacts:

- `news` table
- `behaviors` table
- `candidates` table

This made the rest of the project faster and cleaner by avoiding repeated raw TSV parsing.

### 2. TF-IDF retrieval baseline

A lexical retrieval model was built using:

- user profile text formed from recent clicked history titles
- candidate article text formed from category, subcategory, title, and abstract
- **TF-IDF vectorization**
- cosine similarity scoring

This acts as the first-stage retriever.

### 3. Logistic reranker

A second-stage reranker was built using engineered numeric features such as:

- retrieval score
- lexical overlap
- title overlap
- category/subcategory affinity
- history length
- impression size
- text length features

The reranker uses **logistic regression** to learn a better final ordering over the candidate set.

### 4. Neural two-tower retriever

A first neural retrieval model was built using PyTorch:

- **user tower**
- **item tower**
- token embeddings
- masked mean pooling
- projection layers
- normalized dense embeddings
- dot-product scoring

This model learns user-item similarity in an embedding space instead of relying only on lexical term overlap.

### 5. Config-driven training and experiment artifacts

The project includes:

- CLI training scripts
- YAML-based configuration
- saved model artifacts
- saved experiment logs
- metric summaries
- scored dev samples
- retrieval/reranking previews

---

## System Design

## Stage 1: Retrieval

The purpose of retrieval is to quickly identify promising candidate items for a user.

In this repo, two retrieval approaches were implemented:

### TF-IDF retriever

This approach represents both sides as weighted bags of words:

- **user side:** recent clicked history text
- **item side:** article metadata/text

Similarity is computed using cosine similarity over TF-IDF vectors.

#### Strengths
- simple
- fast
- easy to debug
- highly interpretable
- very strong when exact or near-exact lexical overlap matters

#### Weaknesses
- heavily lexical
- limited semantic generalization
- weaker on synonyms, paraphrases, and broader concept-level matching
- does not learn from interaction behavior end to end

### Neural two-tower retriever

This approach learns dense embeddings for users and items:

- **user tower:** encodes recent user history text
- **item tower:** encodes candidate article text
- retrieval score = dot product between learned embeddings

#### Strengths
- learns dense representations
- naturally supports embedding-based retrieval
- is compatible with future ANN indexing systems such as FAISS
- can eventually capture semantic similarity beyond exact token overlap

#### Weaknesses
- more complex
- harder to debug
- requires more tuning
- may underperform a strong lexical baseline if data scale, training objective, or architecture are not yet strong enough

---

## Stage 2: Reranking

The reranker takes the candidate set from retrieval and tries to improve the **top of the ranked list**.

This repo uses a **logistic regression reranker** over handcrafted features.

This is not a deep reranker, but it is still meaningful because it demonstrates:

- two-stage ranking architecture
- feature engineering
- score fusion
- metric-driven refinement of ordering quality

In practice, the reranker improved the **top-ranked results**, which is exactly what a second-stage model should do.

---

## Why Compare Baselines Instead of Assuming “Neural = Better”

A big lesson of this project is that **more complex does not automatically mean better**.

It is tempting to assume a neural model must outperform a classical one, but that is often false, especially when:

- the dataset is not huge
- the neural model is still small or early-stage
- lexical overlap is already very informative
- the simpler model is well matched to the problem

That is why strong ML work always asks:

- What is the simplest reasonable baseline?
- Can the advanced model beat it fairly?
- If not, why not?

This repo was built with that philosophy.

Instead of claiming victory just because a neural retriever exists, the project trains both approaches on the same sampled setup and compares them using the same ranking metrics.

That makes the repo much stronger and much more credible.

---

## Evaluation Metrics

This project evaluates ranking quality at the **impression level** using:

### MRR@K
**Mean Reciprocal Rank**

Measures how high the first relevant clicked item appears.

Higher is better.

### Recall@K
Measures how many relevant clicked items were recovered in the top K.

Higher is better.

### NDCG@K
**Normalized Discounted Cumulative Gain**

Measures ranking quality while rewarding relevant items appearing near the top of the list.

Higher is better.

These metrics are much more appropriate than plain accuracy for ranking systems.

---

## Final Experimental Results

## A. Retrieval Comparison: TF-IDF vs Two-Tower

A direct comparison was run where both retrievers used the same sampled train/dev setup.

### Dataset size for this comparison

**Train**
- rows: 3,656
- positives: 766
- negatives: 2,890
- unique impressions: 500

**Dev**
- rows: 993
- positives: 205
- negatives: 788
- unique impressions: 120

### TF-IDF metrics

- **MRR@5:** 0.5356
- **Recall@5:** 0.7951
- **NDCG@5:** 0.5570

- **MRR@10:** 0.5456
- **Recall@10:** 0.9427
- **NDCG@10:** 0.6195

- **MRR@20:** 0.5462
- **Recall@20:** 0.9886
- **NDCG@20:** 0.6373

### Two-Tower metrics

- **MRR@5:** 0.4639
- **Recall@5:** 0.8057
- **NDCG@5:** 0.5209

- **MRR@10:** 0.4685
- **Recall@10:** 0.9182
- **NDCG@10:** 0.5681

- **MRR@20:** 0.4692
- **Recall@20:** 0.9856
- **NDCG@20:** 0.5945

### Two-Tower minus TF-IDF

- **MRR@5:** -0.0717
- **Recall@5:** +0.0106
- **NDCG@5:** -0.0361

- **MRR@10:** -0.0771
- **Recall@10:** -0.0245
- **NDCG@10:** -0.0515

- **MRR@20:** -0.0770
- **Recall@20:** -0.0030
- **NDCG@20:** -0.0427

### Interpretation

The **TF-IDF retriever outperformed the first two-tower model on most ranking metrics**.

The two-tower only slightly beat TF-IDF on **Recall@5**, but TF-IDF was stronger on:

- MRR@5
- NDCG@5
- MRR@10
- NDCG@10
- MRR@20
- NDCG@20

This does **not** mean two-tower retrieval is a bad idea.

It means:

- the first neural version is functioning
- the comparison is honest
- the lexical baseline remains stronger on this dataset/setup

That is exactly the kind of insight a good recommender-system project should surface.

---

## B. Two-Stage Ranking: Retrieval + Reranker

The reranker was evaluated against retrieval-only results.

### Retrieval-only metrics

- **MRR@5:** 0.5356
- **Recall@5:** 0.7951
- **NDCG@5:** 0.5570

- **MRR@10:** 0.5456
- **Recall@10:** 0.9427
- **NDCG@10:** 0.6195

- **MRR@20:** 0.5462
- **Recall@20:** 0.9886
- **NDCG@20:** 0.6373

### Reranker metrics

- **MRR@5:** 0.5544
- **Recall@5:** 0.8174
- **NDCG@5:** 0.5800

- **MRR@10:** 0.5638
- **Recall@10:** 0.9323
- **NDCG@10:** 0.6271

- **MRR@20:** 0.5638
- **Recall@20:** 0.9843
- **NDCG@20:** 0.6478

### Reranker minus retrieval-only

- **MRR@5:** +0.0189
- **Recall@5:** +0.0224
- **NDCG@5:** +0.0230

- **MRR@10:** +0.0183
- **Recall@10:** -0.0104
- **NDCG@10:** +0.0075

- **MRR@20:** +0.0176
- **Recall@20:** -0.0042
- **NDCG@20:** +0.0106

### Interpretation

The reranker improved the **top of the ranked list**, especially at K=5.

That is exactly what we want from a second-stage model.

Even though recall dipped slightly at deeper cutoffs, the reranker improved:

- how high clicked items appear
- top-list quality
- discounted ranking gain

This is a realistic and useful ranking outcome.

---

## What We Learned

### 1. Strong lexical baselines matter
TF-IDF was a strong retriever on this setup because article titles and user history titles share meaningful lexical signals.

### 2. Neural models are not automatically better
The two-tower worked, trained, and produced valid rankings, but it did not beat TF-IDF yet.

### 3. Two-stage systems are powerful
Even with a simple retrieval stage, a second-stage reranker improved the final ranking quality.

### 4. Honest comparisons make the repo stronger
This project is stronger because it includes:
- baseline
- neural upgrade
- direct comparison
- real conclusions

Rather than pretending the neural model won, the repo shows the actual outcome.

---

## Repository Structure

```text
discovery-ranking-system/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
├── scripts/
│   ├── preprocess.py
│   ├── train_retrieval.py
│   ├── train_reranker.py
│   └── compare_retrievers.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── retrieval/
│   ├── ranking/
│   ├── serving/
│   └── utils/
├── tests/
└── artifacts/