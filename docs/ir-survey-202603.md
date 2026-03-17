# Information Retrieval Methods Survey (2024-2025)

*Compiled March 2026. Covers SIGIR, NeurIPS, ICLR, EMNLP, ACL, ICML proceedings from 2024-2025.*

---

## 1. Dense Retrieval and Bi-Encoders

Dense retrieval methods encode queries and documents independently into fixed-dimensional dense vectors using a dual-encoder (bi-encoder) architecture, typically built on pre-trained Transformers such as BERT or increasingly LLM backbones. At inference time, document vectors are pre-computed and indexed, and retrieval reduces to approximate nearest neighbor (ANN) search in the embedding space, making it highly efficient for first-stage retrieval over millions of documents. The standard training objective is contrastive loss (e.g., InfoNCE), which pushes query-positive pairs together and query-negative pairs apart in the embedding space. The key strength of bi-encoders is their speed: documents are encoded once offline and queries are encoded in milliseconds, with sub-linear search via ANN libraries like FAISS. The principal weakness is that query and document never attend to each other during encoding, limiting the model's ability to capture fine-grained token-level interactions. As a result, dense retrievers often underperform cross-encoders on precision-oriented metrics, but they serve as the dominant first-stage retrieval method in modern pipelines due to their scalability.

### 1.1 Scaling and Foundations

This subsection covers work on understanding how dense retrieval performance scales with model size, training data, and compute budget, as well as foundational architectural choices for dense retrievers. These papers investigate scaling laws analogous to those found in language modeling, explore LLM backbones as replacements for BERT-scale encoders, and introduce corpus-aware or conversation-aware encoding strategies. The results inform practical decisions about model selection and resource allocation when building dense retrieval systems.

#### Paper: Scaling Laws For Dense Retrieval
- **cite_id**: fang2024scaling
- **Authors**: Yan Fang et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2403.18684
- **Core method**: Investigates power-law relationships between dense retrieval performance and scaling factors (model size, data quantity, annotation quality). Uses a pre-trained Transformer with a projection layer as a text encoder, mapping text to shared dense embeddings. Proposes contrastive entropy as a continuous evaluation metric replacing discrete ranking metrics, enabling stable scaling law fitting across model sizes from 0.5M to 87M non-embedding parameters.
- **Training**: Contrastive ranking loss on MS MARCO (English) and T2Ranking (Chinese) with 256 random negatives per step, 10,000 steps. No early stopping.
- **Key innovation vs prior work**: First systematic exploration of scaling laws for dense retrieval; proposes contrastive entropy as a continuous metric for curve fitting; derives joint scaling laws for practical budget allocation under model size and data constraints.
- **Re-implementation complexity**: Medium
- **Results highlight**: R-squared > 0.99 for model size scaling on T2Ranking; power-law exponent alpha=0.53 for both datasets.
- **Access status**: Full text read

#### Paper: Large Language Models as Foundations for Next-Gen Dense Retrieval: A Comprehensive Empirical Assessment
- **cite_id**: luo2024llmfoundations
- **Authors**: Kun Luo et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2408.12194
- **Core method**: Comprehensive empirical assessment of 15+ backbone LLMs and non-LLMs across in-domain accuracy, data efficiency, zero-shot generalization, lengthy retrieval, instruction-based retrieval, and multi-task learning. Evaluates models ranging from BERT-base to LLaMA-70B as dense retrieval backbones.
- **Training**: Standard contrastive training with MS MARCO and other retrieval datasets; evaluates varying training data sizes to measure data efficiency.
- **Key innovation vs prior work**: First systematic comparison of LLM-based vs. traditional backbones across six distinct retrieval capabilities; demonstrates that larger models and extensive pre-training consistently enhance in-domain accuracy and data efficiency.
- **Re-implementation complexity**: Low (evaluation study)
- **Results highlight**: Larger LLM backbones show significant advantage in zero-shot generalization and instruction-based retrieval. Systematic comparison of 15+ backbone models from BERT-base to LLaMA-70B across six retrieval capabilities.
- **Access status**: Abstract + search summaries (full paper at arXiv)

#### Paper: Contextual Document Embeddings
- **cite_id**: morris2025contextual
- **Authors**: John X. Morris, Alexander M. Rush
- **Venue**: ICLR 2025
- **arXiv/URL**: https://arxiv.org/abs/2410.02525
- **Core method**: Two complementary approaches to contextualize document embeddings: (1) adversarial contrastive training using clustering to partition data into pseudo-domains, maximizing within-batch difficulty; (2) a two-stage CDE architecture where stage-1 encoder M1 embeds contextual documents from corpus, and stage-2 encoder M2 encodes target document/query while receiving concatenated contextual embeddings as additional input tokens.
- **Training**: 234M weakly-supervised pairs + 1.8M supervised pairs (MS MARCO, HotpotQA, BGE meta-datasets). Adam optimizer, lr 2e-5, 1000 warmup steps, temperature tau=0.02.
- **Key innovation vs prior work**: Makes document embeddings corpus-aware, analogous to how contextualized word embeddings revolutionized NLP. Addresses domain shift by letting embeddings adapt to neighboring document context, unlike standard bi-encoders that embed documents independently.
- **Re-implementation complexity**: High
- **Results highlight**: cde-small-v1 achieves 65.00 MTEB average (vs 63.56 for bge-base-en-v1.5); NDCG@10 63.1 on BEIR (small setting) vs 59.9 for standard biencoder.
- **Access status**: Full text read

#### Paper: ChatRetriever: Adapting Large Language Models for Generalized and Robust Conversational Dense Retrieval
- **cite_id**: mao2024chatretriever
- **Authors**: Kelong Mao et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.13556
- **Core method**: Dual-learning method that adapts LLMs for conversational dense retrieval via contrastive learning while enhancing complex session understanding through masked instruction tuning on high-quality conversational instruction tuning data. Inherits strong generalization capability of LLMs to robustly represent complex conversational sessions.
- **Training**: Contrastive learning + masked instruction tuning on conversational instruction tuning data from multiple conversational search benchmarks.
- **Key innovation vs prior work**: Combines LLM generalization with retrieval-specific contrastive training via dual-learning, avoiding the brittle query rewriting approaches used in prior conversational search.
- **Re-implementation complexity**: Medium
- **Results highlight**: NDCG@3: QReCC 52.5 (vs best CDR 48.5), TopiOCQA 40.1 (vs 31.4), CAsT-19 52.1 (vs 47.0), CAsT-20 40.0 (vs 33.2), CAsT-21 49.6 (vs 37.4). On par with LLM-based rewriting (LLM4CS).
- **Access status**: Full text read

### 1.2 Embedding Models

This subsection covers general-purpose text embedding models designed to produce high-quality dense representations across diverse tasks including retrieval, classification, and semantic similarity. These models typically fine-tune large pre-trained language models (often decoder-only LLMs like Mistral or LLaMA) with contrastive objectives on large-scale, often synthetically generated, training data. Key innovations include instruction-tuned embeddings, unified embedding-generation architectures, and improved pooling strategies. These models are evaluated on broad benchmarks like MTEB and BEIR, and serve as drop-in encoders for dense retrieval pipelines.

#### Paper: Improving Text Embeddings with Large Language Models
- **cite_id**: wang2024e5mistral
- **Authors**: Liang Wang et al.
- **Venue**: ACL 2024
- **arXiv/URL**: https://arxiv.org/abs/2401.00368
- **Core method**: Uses GPT-4/3.5-Turbo to generate 500k synthetic training examples across diverse embedding tasks in 93 languages with a two-step prompting strategy. Fine-tunes Mistral-7B using LoRA (rank 16) with InfoNCE contrastive loss (temperature 0.02) on synthetic data plus 13 public datasets (~1.8M examples total).
- **Training**: Less than 1k training steps, batch size 2048, lr 1e-4, 32 V100 GPUs, ~18 hours. InfoNCE loss.
- **Key innovation vs prior work**: Eliminates need for complex multi-stage training pipelines and manually-curated labeled datasets; demonstrates that LLM-generated synthetic data alone can achieve competitive embedding quality.
- **Re-implementation complexity**: Medium
- **Results highlight**: MTEB average 66.6 (56 datasets); BEIR average 56.9 nDCG@10.
- **Access status**: Full text read

#### Paper: Generative Representational Instruction Tuning (GritLM)
- **cite_id**: muennighoff2025gritlm
- **Authors**: Niklas Muennighoff et al.
- **Venue**: ICLR 2025
- **arXiv/URL**: https://arxiv.org/abs/2402.09906
- **Core method**: Unifies embedding and generative tasks in a single LLM through instruction-based differentiation. For embeddings, uses bidirectional attention with mean pooling; for generation, uses causal attention with a language modeling head. Combines contrastive loss for embeddings and language modeling loss for generation: L_GRIT = lambda_Rep * L_Rep + lambda_Gen * L_Gen.
- **Training**: Embedding data from E5 dataset + S2ORC; generative data from filtered Tulu 2 instructions. Batch size 2048 (embedding) and 256 (generative), 1253 steps, BF16.
- **Key innovation vs prior work**: First model to achieve state-of-the-art on both MTEB embedding and generative benchmarks simultaneously, with performance matching specialized models in each domain. Speeds up RAG by >60% by eliminating separate retrieval/generation models.
- **Re-implementation complexity**: Medium
- **Results highlight**: MTEB average 66.8 (SOTA for open models at time); GritLM 8x7B achieves 65.7 on generation benchmarks.
- **Access status**: Full text read

#### Paper: NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models
- **cite_id**: lee2024nvembed
- **Authors**: Chankyu Lee et al.
- **Venue**: NeurIPS 2024
- **arXiv/URL**: https://arxiv.org/abs/2405.17428
- **Core method**: Three architectural innovations: (1) removes causal attention mask during contrastive training for bidirectional attention; (2) introduces latent attention layer with 512 trainable latent arrays as cross-attention pooling; (3) two-stage instruction-tuning where stage-1 uses contrastive training on retrieval datasets with in-batch negatives, and stage-2 blends retrieval and non-retrieval tasks without in-batch negatives.
- **Training**: ~2.8M retrieval samples from MS MARCO, HotpotQA, NQ, etc. plus non-retrieval tasks. LoRA fine-tuning (rank 16, alpha 32) on Mistral 7B. Hard-negative mining using E5-mistral as teacher.
- **Key innovation vs prior work**: Synthesis of bidirectional attention removal, latent attention pooling, and staged training enables decoder-only LLMs to surpass bidirectional models without proprietary synthetic data. Latent attention consistently outperforms mean/last-token pooling.
- **Re-implementation complexity**: Medium
- **Results highlight**: NV-Embed-v2 achieves 72.31 MTEB average (#1 on leaderboard as of Aug 2024); BEIR 62.65 nDCG@10.
- **Access status**: Full text read

#### Paper: Gecko: Versatile Text Embeddings Distilled from Large Language Models
- **cite_id**: lee2024gecko
- **Authors**: Jinhyuk Lee et al.
- **Venue**: arXiv 2024 (Google DeepMind)
- **arXiv/URL**: https://arxiv.org/abs/2403.20327
- **Core method**: Two-step distillation from LLMs into compact embedding models. Step 1: Generate diverse synthetic paired data using an LLM. Step 2: Retrieve candidate passages for each query and relabel positive and hard negative passages using the same LLM to improve data quality.
- **Training**: Contrastive training on LLM-relabeled synthetic data with hard negatives.
- **Key innovation vs prior work**: Novel two-step LLM distillation process where the LLM both generates and relabels training data, achieving strong performance in a compact model (256/768-dim embeddings).
- **Re-implementation complexity**: Medium (requires LLM API for data generation)
- **Results highlight**: Gecko-1B-768: MTEB average 66.31, retrieval nDCG@10 55.70, STS 85.06, classification 81.17. Gecko-1B-256 outperforms OpenAI text-embedding-3-large-256 (62.00 vs 64.37). FRet zero-shot alone yields 62.64 MTEB average.
- **Access status**: Full text read

#### Paper: M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation
- **cite_id**: chen2024bgem3
- **Authors**: Jianlv Chen et al.
- **Venue**: ACL 2024 Findings
- **arXiv/URL**: https://arxiv.org/abs/2402.03216
- **Core method**: Unifies three retrieval functionalities in one model: dense retrieval (CLS token), lexical/sparse retrieval (learned term weights), and multi-vector retrieval (ColBERT-style late interaction). Uses self-knowledge distillation where relevance scores from all three methods are integrated as teacher signal via s_inter = s_dense + s_lex + s_mul to enhance each individual method.
- **Training**: Three-stage: RetroMAE pre-training, unsupervised pre-training on 1.2B text pairs (194 languages, 25k steps), and fine-tuning with self-KD on 1.1M English + 386.6K Chinese supervised data. Based on XLM-RoBERTa-Large (8192 max length).
- **Key innovation vs prior work**: Self-knowledge distillation across heterogeneous retrieval functions (dense, sparse, multi-vector) enables mutual reinforcement. Supports 100+ languages with three retrieval modes in a single model.
- **Re-implementation complexity**: High
- **Results highlight**: MIRACL nDCG@10 70.0 (vs E5-mistral 62.2); MKQA Recall@100 75.5; MLDR nDCG@10 65.0.
- **Access status**: Full text read

#### Paper: jina-embeddings-v3: Multilingual Embeddings With Task LoRA
- **cite_id**: sturua2024jinav3
- **Authors**: Saba Sturua et al.
- **Venue**: arXiv 2024 (Jina AI)
- **arXiv/URL**: https://arxiv.org/abs/2409.10173
- **Core method**: XLM-RoBERTa-based model (570M params) with task-specific LoRA adapters (<3% of total params) for retrieval, clustering, classification, and text matching. Supports 8192-token contexts and Matryoshka representation learning for flexible dimensionality (1024 down to 32).
- **Training**: Three-stage: MLM pre-training on 89-language CulturaX, bi-directional InfoNCE on 1B+ text pairs, then five LoRA adapter training with task-specific losses. Synthetic data targeting four failure modes.
- **Key innovation vs prior work**: Task-specific LoRA adapters outperform instruction-based approaches while adding minimal overhead; flexible dimensionality via Matryoshka learning.
- **Re-implementation complexity**: Medium
- **Results highlight**: English retrieval 53.87 nDCG@10; multilingual retrieval 57.98 nDCG@10; outperforms OpenAI text-embedding-3-large.
- **Access status**: Full text read

#### Paper: Nomic Embed: Training a Reproducible Long Context Text Embedder
- **cite_id**: nussbaum2024nomic
- **Authors**: Zach Nussbaum et al.
- **Venue**: arXiv 2024 (Nomic AI)
- **arXiv/URL**: https://arxiv.org/abs/2402.01613
- **Core method**: Fully reproducible, open-source 137M parameter embedding model supporting 8192 context length. End-to-end training pipeline with open data, code, and weights under Apache 2.0 license. Based on NomicBERT with FlashAttention.
- **Training**: Multi-stage contrastive training on openly available datasets following E5 methodology.
- **Key innovation vs prior work**: First fully reproducible (open code, data, weights) long-context embedding model that outperforms OpenAI Ada-002 and text-embedding-3-small on both MTEB and long-context (LoCo) benchmarks.
- **Re-implementation complexity**: Low
- **Results highlight**: MTEB average 62.39 (vs Ada-002 60.99, text-embedding-3-small 62.26); retrieval subset nDCG@10: 52.8; LoCo benchmark average 85.53 (vs Ada-002 52.7).
- **Access status**: Full text read

#### Paper: LongEmbed: Extending Embedding Models for Long Context Retrieval
- **cite_id**: zhu2024longembed
- **Authors**: Dawei Zhu et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.12096
- **Core method**: Extends embedding models from 512 to 32k tokens using training-free strategies. For APE models, applies position interpolation/reorganization. For RoPE models, uses NTK-Aware Interpolation and SelfExtend. Also releases E5-Base-4k and E5-RoPE-Base models.
- **Training**: E5-RoPE-Base mirrors E5-Base training on MS-MARCO, NQ, NLI. Pre-training: 32 V100, lr 2e-4, 20k steps. Fine-tuning: 8 GPUs, lr 2e-5, 3 epochs.
- **Key innovation vs prior work**: Demonstrates training-free context window extension for embedding models by adapting LLM extension techniques; establishes RoPE's superiority over APE for length extrapolation in embedding models.
- **Re-implementation complexity**: Low
- **Results highlight**: E5-Mistral + NTK (32k): nDCG@10 average 75.3 on LongEmbed benchmark.
- **Access status**: Full text read

#### Paper: Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions
- **cite_id**: yoon2024matryoshka
- **Authors**: Jinsung Yoon et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2407.20243
- **Core method**: Learns a transformation function (shallow MLP with skip connections) to enhance Matryoshka properties of embeddings, enabling 2-12x dimensionality reduction. Operates in unsupervised (corpus-only) and supervised (query-corpus pairs) modes. Works with black-box APIs without model parameter access.
- **Training**: Combines top-k similarity loss, pairwise similarity loss, and reconstruction loss. Adam optimizer, lr 0.001, batch 128, max 5000 iterations. <10 min unsupervised, <1 hour supervised on single V100.
- **Key innovation vs prior work**: Post-hoc dimensionality reduction that works with any embedding API (including black-box); preserves pairwise and top-k similarity relationships.
- **Re-implementation complexity**: Low
- **Results highlight**: Supervised (Google Gecko) on BEIR: nDCG@10 = 0.5714 at dim 256 (vs original at higher dims).
- **Access status**: Full text read

### 1.3 Zero-Shot and Prompt-Based Retrieval

This subsection covers methods that leverage prompt engineering and instruction-following capabilities of LLMs to perform retrieval without task-specific fine-tuning. Rather than training a dedicated retrieval model, these approaches elicit dense or sparse representations directly from a pre-trained LLM by crafting appropriate prompts. The key advantage is zero-shot generalization: no labeled query-document pairs are needed, and the model can adapt to new domains or tasks via prompt modification alone. The tradeoff is typically lower peak performance compared to fine-tuned retrievers, and higher inference cost when using large LLM backbones.

#### Paper: PromptReps: Prompting Large Language Models to Generate Dense and Sparse Representations for Zero-Shot Document Retrieval
- **cite_id**: zhuang2024promptreps
- **Authors**: Shengyao Zhuang et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.18424
- **Core method**: Prompts LLMs to represent text using a single word, then extracts two representations simultaneously: dense (last layer hidden state of last token) and sparse (logits for next-token prediction). Sparse representation undergoes sparsification via ReLU, log-saturation, top-128 token retention, and quantization. Hybrid system combining both.
- **Training**: Zero-shot method requiring no training. Supervised fine-tuning variant available with InfoNCE loss on MS MARCO.
- **Key innovation vs prior work**: Generates both dense and sparse representations through a single LLM forward pass using prompt engineering; eliminates expensive contrastive pre-training (saves ~$2,300 and ~5.6 kgCO2e).
- **Re-implementation complexity**: Low
- **Results highlight**: BEIR average nDCG@10: 45.97 (Llama3-8B hybrid); supervised MRR@10: 42.58 on MS MARCO.
- **Access status**: Full text read

#### Paper: GENRA: Enhancing Zero-shot Retrieval with Rank Aggregation
- **cite_id**: katsimpras2024genra
- **Authors**: Georgios Katsimpras, Georgios Paliouras
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://aclanthology.org/2024.emnlp-main.431/
- **Core method**: LLMs generate informative passages capturing query intent, which guide retrieval of relevant documents. LLMs then assess relevance of each retrieved document. Final rankings are produced via rank aggregation combining individual rankings into a unified result.
- **Training**: Zero-shot approach; no fine-tuning required.
- **Key innovation vs prior work**: Combines LLM-based passage generation, relevance assessment, and rank aggregation for zero-shot retrieval without any training data.
- **Re-implementation complexity**: Low
- **Results highlight**: Improves over existing zero-shot approaches on BEIR benchmark datasets through LLM passage generation + relevance assessment + rank aggregation pipeline.
- **Access status**: Abstract + search summaries (PDF at ACL Anthology)

## 2. Sparse and Learned Sparse Retrieval

Sparse retrieval methods represent queries and documents as high-dimensional sparse vectors aligned with the vocabulary, extending the classic term-matching paradigm (BM25) by learning term weights with neural networks rather than relying on hand-crafted statistics like TF-IDF. The dominant architecture is SPLADE and its variants, which use a masked language model head to predict importance weights for each vocabulary term, producing sparse vectors that can be stored and searched using standard inverted indexes. Training typically combines contrastive loss with sparsity regularization (e.g., FLOPS regularizer) to control the density of the learned representations. The key strength is compatibility with existing inverted index infrastructure -- these models can be deployed on the same systems that serve BM25, with comparable query latency. Learned sparse models also retain the interpretability and exact-match capabilities of term-based retrieval while incorporating semantic understanding from pre-training. The main weakness relative to dense retrieval is that sparse models can struggle with purely semantic matching where no lexical overlap exists, though in practice they often achieve competitive or superior performance on standard benchmarks.

#### Paper: Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations (Seismic)
- **cite_id**: bruch2024seismic
- **Authors**: Sebastian Bruch et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.18812
- **Core method**: Seismic organizes inverted lists into geometrically-cohesive blocks using K-means clustering, where each block has a summary vector (coordinate-wise max). Query processing uses summary inner products for rapid block filtering. Combines static pruning (top-lambda entries) with dynamic pruning through block-level skipping, plus a forward index for exact scoring.
- **Training**: Indexing algorithm (no ML training); indexes pre-computed SPLADE/E-SPLADE/uniCoil-T5 embeddings.
- **Key innovation vs prior work**: Addresses distribution mismatch between learned sparse representations and BM25 that makes traditional WAND/MaxScore algorithms inefficient; introduces geometrically-cohesive blocks with per-block summaries leveraging "concentration of importance" property.
- **Re-implementation complexity**: High (systems-level implementation)
- **Results highlight**: 187 usec/query on MS MARCO (SPLADE, 90% recall) vs 4200 usec for SparseIvf; 1-2 orders of magnitude faster than prior methods.
- **Access status**: Full text read

#### Paper: DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities
- **cite_id**: nguyen2024dyvo
- **Authors**: Thong Nguyen et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2410.07722
- **Core method**: Augments LSR vocabularies with Wikipedia entities to resolve ambiguities from subword tokenization. A Dynamic Vocabulary head scores entity candidates using pre-computed entity embeddings matched against contextualized hidden states via dot product with ReLU gating and log scaling. Entity weights and word-piece weights are combined through separate summations.
- **Training**: Two stages: pre-train LSR on MS MARCO using KL distillation from cross-encoders (300k steps), then fine-tune on target datasets using synthesized queries and MonoT5-3B distillation. Entity embeddings frozen; only projection layers trained.
- **Key innovation vs prior work**: Rather than exhaustive entity scoring, uses few-shot LLM-based candidate generation (Mixtral, GPT4) to identify Wikipedia entities helpful for retrieval, complementing traditional entity linking.
- **Re-implementation complexity**: High
- **Results highlight**: Robust04 nDCG@10: 54.39; CODEC nDCG@10: 56.46.
- **Access status**: Full text read

#### Paper: SPLATE: Sparse Late Interaction Retrieval
- **cite_id**: formal2024splate
- **Authors**: Thibault Formal et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.13950
- **Core method**: Lightweight two-layer MLP adapter mapping frozen ColBERTv2 token embeddings to sparse vocabulary space via residual connections: w_iv = (h_i + MLP(h_i))^T E_v + b_v. Logits undergo max-pooling aggregation across tokens with log-saturation ReLU transformation. Only 0.6M trainable adapter parameters added to frozen ColBERTv2.
- **Training**: Distillation on MS MARCO passages over 3 epochs, batch size 24. marginMSE and KLDiv losses with 20 hard negatives per query from ColBERTv2 top-1000. Training completes in <2 hours on dual V100s.
- **Key innovation vs prior work**: Enables traditional sparse retrieval infrastructure for late interaction pipelines; eliminates need for specialized CUDA kernels (PLAID) while maintaining performance in CPU environments.
- **Re-implementation complexity**: Low
- **Results highlight**: MS MARCO Dev MRR@10: 40.0 (matching ColBERTv2 39.7); BEIR nDCG@10: 74.2; latency as low as 2.9ms.
- **Access status**: Full text read

#### Paper: Dynamic Superblock Pruning for Fast Learned Sparse Retrieval
- **cite_id**: carlson2025superblock
- **Authors**: Parker Carlson, Wentai Xie, Shanxiu He, Tao Yang
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2504.17045
- **Core method**: Structures the sparse index as a set of superblocks on a sequence of document blocks and conducts superblock-level selection to decide if some superblocks can be pruned before visiting their child blocks. Generalizes flat block or cluster-based pruning, enabling early detection of groups of documents unlikely to appear in the final top-k list.
- **Training**: N/A (indexing/retrieval algorithm)
- **Key innovation vs prior work**: Two-level pruning (superblock + block) that generalizes BMP; achieves significant speedups over Seismic, ASC, and BMP while maintaining rank-safe or near-rank-safe recall.
- **Re-implementation complexity**: Medium
- **Results highlight**: Under 99%+ recall on SPLADE (MS MARCO), SP is 2.3-3.8x faster than Seismic, 3.2-9.4x faster than ASC, and up to 2.9x faster than BMP. MRR@10 stays at 38.1 across configurations. Rank-safe k=10: 0.629ms (vs BMP 1.44ms, ASC 4.70ms).
- **Access status**: Full text read

#### Paper: Effective Inference-Free Retrieval for Learned Sparse Representations
- **cite_id**: mackenzie2025inferencefree
- **Authors**: Franco Maria Nardini, Thong Nguyen, Cosimo Rulli, Rossano Venturini, Andrew Yates
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2505.01452
- **Core method**: Proposes Li-Lsr (Learned Inference-free Retrieval), which learns a static relevance score for each vocabulary token, casting query encoding to a fast table lookup operation. Uses formula si = log(1 + ReLU[wT*EW(xi) + b]). Extended evaluation of regularization approaches for LSR with focus on effectiveness, efficiency, and out-of-domain generalization.
- **Training**: KL divergence loss with L1 regularization, single negative per query. Lighter regularization allowing more non-zero terms improves quality.
- **Key innovation vs prior work**: Eliminates expensive neural inference at query time by learning static per-token scores; demonstrates query encoding is the bottleneck limiting LSR performance.
- **Re-implementation complexity**: Medium
- **Results highlight**: Li-Lsr Big: MRR@10 ~40.8 on MS MARCO, nDCG@10 ~51.0 on BEIR; TREC DL19 nDCG@10 72.1, DL20 70.5. Surpasses SPLADE-v3-Doc by ~1.0 MRR@10 and ~1.8 nDCG@10 on BEIR.
- **Access status**: Full text read

## 3. Late Interaction / Multi-Vector Methods

Late interaction methods represent queries and documents not as single vectors but as sets of token-level embeddings, deferring the interaction computation to retrieval time. The canonical example is ColBERT, which encodes each token independently and computes relevance via MaxSim: for each query token, find the maximum cosine similarity with any document token, then sum across query tokens. This "late interaction" pattern is more expressive than single-vector bi-encoders because it preserves fine-grained token-level matching signals, while remaining more efficient than full cross-encoders because query and document are still encoded independently. The training objective is typically contrastive loss with in-batch negatives and hard negatives, applied to the MaxSim aggregated score. The primary weakness is storage: representing each document as a matrix of per-token vectors (e.g., 128 tokens x 128 dimensions) dramatically increases index size compared to single-vector approaches. Late interaction models sit between bi-encoders and cross-encoders in the expressiveness-efficiency tradeoff, and are used as either first-stage retrievers or lightweight rerankers depending on the deployment constraints.

#### Paper: WARP: An Efficient Engine for Multi-Vector Retrieval
- **cite_id**: scheerer2025warp
- **Authors**: Jan Luca Scheerer et al.
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2501.17788
- **Core method**: Optimizes XTR-trained multi-vector retrieval through four stages: query encoding, candidate generation with WARP_SELECT (dynamic similarity imputation based on cumulative cluster sizes), implicit decompression (avoids explicit vector reconstruction by reusing query-centroid scores), and two-stage reduction (token-level max + document-level sum with imputation).
- **Training**: N/A (retrieval engine for pre-trained models)
- **Key innovation vs prior work**: WARP_SELECT performs dynamic similarity imputation; implicit decompression eliminates costly vector reconstruction; achieves 3x speedup over PLAID and 41x over XTR/ScaNN reference implementation.
- **Re-implementation complexity**: High (systems-level C++ implementation)
- **Results highlight**: LoTTE Pooled latency: 171ms (vs 2156ms XTR/ScaNN); Success@5 70.3; 2-4x index size reduction.
- **Access status**: Full text read

#### Paper: AGRaME: Any-Granularity Ranking with Multi-Vector Embeddings
- **cite_id**: reddy2024agrame
- **Authors**: Revanth Gangi Reddy et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2405.15028
- **Core method**: Enables ranking at varying granularities (passage, sentence, proposition) using multi-vector embeddings while encoding at a single coarser level. Uses distinct query marker tokens to signal ranking granularity to the ColBERTv2 encoder. In-passage scores computed via MaxSim between query and passage token embeddings.
- **Training**: MS MARCO for passage-level + QA datasets for sentence-level supervision. Combined loss: L = L_psg (KL-divergence from cross-encoder) + L_sent (sentence-level KL-divergence weighted by passage relevance).
- **Key innovation vs prior work**: Distinct query markers signal granularity to a single encoder; multi-granular contrastive training adds sentence/proposition supervision without separate indexes per granularity.
- **Re-implementation complexity**: Medium
- **Results highlight**: NQ sentence-level P@1: 36.8 (vs ColBERTv2 27.4); proposition-level P@1: 47.7 on PropSegmEnt.
- **Access status**: Full text read

#### Paper: Towards Lossless Token Pruning in Late-Interaction Retrieval Models
- **cite_id**: sigir2025tokenpruning
- **Authors**: Yuxuan Zong, Benjamin Piwowarski
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2504.12778
- **Core method**: Uses a principled approach to define how to prune tokens without impacting the retrieval score between a document and a query. Introduces three regularization losses (document similarity, L1-norm, nuclear norm) that induce solutions with high pruning ratios, combined with two pruning strategies (linear programming-based and norm-based). Applied to ColBERTv2.
- **Training**: MS MARCO v1 (8.8M passages), AdamW (lr 1e-5), batch size 16, 320k steps on NVIDIA V100 GPUs (5-8 days). Base checkpoint: ColBERTv2.
- **Key innovation vs prior work**: Principled framework for lossless token pruning with theoretical guarantees; achieves near-lossless performance at 70% token removal, unlike heuristic or statistical pruning.
- **Re-implementation complexity**: Medium
- **Results highlight**: Preserves ColBERT performance using only 30% of tokens. In-domain: MRR@10 39.7 on MS MARCO dev-small at 40% tokens; TREC DL19 nDCG@10 73.8 at 32% tokens (vs 74.4 baseline); DL20 nDCG@10 73.3 at 32% tokens (vs 75.6). Out-of-domain: BEIR nDCG@10 45.7 at 37% tokens; LoTTE Success@5 71.3 at 35% tokens.
- **Access status**: Full text read

## 4. Generative Retrieval

Generative retrieval replaces the traditional encode-then-match paradigm with a sequence-to-sequence model that directly generates document identifiers given a query. The model (typically a T5-variant or similar encoder-decoder) is trained to map query text to a structured document ID -- either a numeric string, hierarchical cluster path, or learned token sequence -- using standard language modeling loss. No separate index is needed at inference time; the model's parameters implicitly memorize the corpus. This makes generative retrieval architecturally elegant and avoids the storage overhead of explicit vector indexes. However, the approach faces significant practical challenges: updating the corpus requires retraining or fine-tuning the entire model, scaling to millions of documents remains difficult, and performance on standard benchmarks has not yet consistently surpassed well-tuned dense or sparse retrievers. Generative retrieval is an active area of research that may become more viable as model capacity and training techniques improve, but it is currently more experimental than the encode-and-match alternatives.

#### Paper: Generative Retrieval as Multi-Vector Dense Retrieval
- **cite_id**: wu2024grmvdr
- **Authors**: Shiguang Wu et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.00684
- **Core method**: Demonstrates that generative retrieval and multi-vector dense retrieval share an equivalent framework for computing relevance as a sum of products of query and document vectors with an alignment matrix. GR's logits can be reformulated as sum(E_d^T Q * A), matching MVDR's unified framework. They differ in document encoding (simple vs contextualized vectors) and alignment direction.
- **Training**: NQ320K (307k pairs) and MS MARCO (366k pairs). GR uses cross-entropy token prediction loss; MVDR uses contrastive loss with in-batch negatives. T5 backbone with batch size 256.
- **Key innovation vs prior work**: Rigorous mathematical connection between generative retrieval and multi-vector dense retrieval through analysis of attention layer and prediction head, showing GR is a special case of MVDR.
- **Re-implementation complexity**: Medium
- **Results highlight**: MVDR (q->d) R@1: 61.3, R@10: 91.9 on NQ320K rerank setting.
- **Access status**: Full text read

#### Paper: Generative Retrieval via Term Set Generation (TSGen)
- **cite_id**: zhang2024tsgen
- **Authors**: Peitian Zhang et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2305.13859
- **Core method**: Replaces sequence-based document identifiers with term sets. A BERT encoder with MLP scores term importance via contrastive learning, selecting top-N discriminative terms per document. Permutation-invariant decoding via inverted index allows any term ordering to map to the same document. Iterative optimization where model generates preferred permutations as training targets.
- **Training**: Iterative optimization over T iterations. At each iteration, model generates favorable term permutations via constrained beam search, then trains on self-generated permutations. T5-base backbone.
- **Key innovation vs prior work**: Term set DocIDs allow any permutation to retrieve the correct document, solving the "false pruning problem" where a single mispredicted token causes sequence-based GR to fail.
- **Re-implementation complexity**: Medium
- **Results highlight**: NQ320K MRR@10: 0.771; MS300K MRR@10: 0.502.
- **Access status**: Full text read

#### Paper: Planning Ahead in Generative Retrieval (PAG)
- **cite_id**: zeng2024pag
- **Authors**: Hansi Zeng et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.14600
- **Core method**: Constructs dual identifiers (lexical bag-of-words + sequential quantized relevance representations) and guides autoregressive generation through simultaneous decoding.
- **Training**: [Details from abstract]
- **Key innovation vs prior work**: Dual-identifier approach combining lexical and relevance-based representations enables 22x latency speedup over prior GR methods while improving effectiveness.
- **Re-implementation complexity**: Medium
- **Results highlight**: 15.6% MRR improvement on MS MARCO over SOTA; 22x query latency speedup.
- **Access status**: Abstract + search summaries

#### Paper: Self-Retrieval: End-to-End Information Retrieval with One Large Language Model
- **cite_id**: tang2024selfretrieval
- **Authors**: Qiaoyu Tang et al.
- **Venue**: NeurIPS 2024
- **arXiv/URL**: https://arxiv.org/abs/2403.00801
- **Core method**: Consolidates indexing, retrieval, and reranking into a single LLM. Corpus internalized through self-supervised sentence-to-passage learning. Retrieval via constrained decoding generating document titles and passages using a trie structure. Self-assessment scoring ranks retrieved passages.
- **Training**: Three components: self-supervised corpus internalization, supervised query-passage pairs, and positive/negative passage pairs for reranking. Cross-entropy auto-regressive loss across all three tasks.
- **Key innovation vs prior work**: Direct generation of actual passage content (not DocIDs) with trie-constrained decoding ensuring generated text matches corpus exactly; unified end-to-end IR in a single LLM.
- **Re-implementation complexity**: High
- **Results highlight**: NQ320K R@1: 73.3, R@10: 92.6, MRR@100: 80.7; NQ passage H@1: 63.44 (Llama 2 7B).
- **Access status**: Full text read

#### Paper: Constrained Auto-Regressive Decoding Constrains Generative Retrieval
- **cite_id**: sigir2025constrained
- **Authors**: Shiguang Wu, Wenda Wei, Mengqi Zhang, Zhumin Chen, Jun Ma, Zhaochun Ren, Maarten de Rijke, Pengjie Ren
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2504.09935
- **Core method**: Provides theoretical analysis of constrained auto-regressive decoding in generative retrieval under a Bayes-optimal setting. Derives lower bounds on estimation error in terms of KL divergence between ground-truth and model-predicted step-wise marginal distributions. Analyzes how beam search achieves perfect top-1 precision but suffers from poor top-k recall in sparse relevance scenarios.
- **Training**: Theoretical analysis paper; experiments on TREC DL 2019/2020 with MS MARCO.
- **Key innovation vs prior work**: First rigorous theoretical analysis showing constraints cause marginal distribution mismatch; error proportional to Simpson diversity index of relevance distribution. Reveals fundamental precision-recall tradeoff in beam search for generative retrieval.
- **Re-implementation complexity**: Medium (theoretical + experimental)
- **Results highlight**: TREC DL 2019/2020: Recall@50 53.7-67.5% while Precision@1 69.8-90.5%, demonstrating the predicted precision-recall tradeoff of constrained decoding.
- **Access status**: Full text read

## 5. Reranking

Reranking is a second-stage technique that rescores a candidate list produced by a fast first-stage retriever (e.g., BM25 or a bi-encoder). Because rerankers only process a small candidate set (typically 100-1000 documents), they can afford to use much more expressive models that jointly attend to the full query-document pair. Traditional neural rerankers use cross-encoder architectures (e.g., a BERT model that takes the concatenated query and document as input and outputs a relevance score), trained with pointwise or pairwise ranking losses. More recently, LLMs have been applied to reranking in pointwise, pairwise, listwise, and setwise configurations. Cross-encoders are far more accurate than bi-encoders on a per-pair basis but are too slow to apply to the full corpus, which is why they are used exclusively as second-stage components. The overall pipeline quality depends on the first-stage recall: the reranker can only promote documents that the first stage retrieved.

### 5.1 LLM-Based Reranking

This subsection covers reranking methods that leverage large language models, exploiting their instruction-following and reasoning capabilities to assess document relevance. Approaches include pointwise scoring (LLM rates each document independently), pairwise comparison (LLM judges which of two documents is more relevant), listwise ranking (LLM sorts an entire list at once), and setwise ranking (LLM selects the most relevant from a candidate set). These methods often work in a zero-shot or few-shot setting without retrieval-specific fine-tuning. The tradeoff is cost: LLM-based rerankers are significantly more expensive per query than trained cross-encoders, but they can achieve strong performance without any labeled training data.

#### Paper: A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models
- **cite_id**: zhuang2024setwise
- **Authors**: Shengyao Zhuang et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2310.09497
- **Core method**: Instructs LLMs to select the most relevant document from a set of candidate documents (typically 3+) rather than pairwise comparisons. Enables sorting algorithms (heap sort, bubble sort) to compare multiple documents simultaneously, reducing LLM inferences from O(k*log2(N)) to O(k*logc(N)) where c>=3.
- **Training**: Zero-shot; no fine-tuning. Uses pre-trained Flan-T5 (780M, 3B, 11B) and LLaMA-2/Vicuna models.
- **Key innovation vs prior work**: Replaces pairwise with set-based comparisons in sorting algorithms, reducing LLM inference calls while maintaining or improving ranking quality.
- **Re-implementation complexity**: Low
- **Results highlight**: TREC DL19 nDCG@10: 0.678 (Flan-T5-large bubblesort); BEIR average: 0.483; 2x faster than pairwise.
- **Access status**: Full text read

#### Paper: FIRST: Faster Improved Listwise Reranking with Single Token Decoding
- **cite_id**: reddy2024first
- **Authors**: Revanth Gangi Reddy et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2406.15657
- **Core method**: Extracts output logits of candidate identifier tokens during generation of only the first passage identifier, ranking candidates by decreasing logit values. Uses alphabetic identifiers (A-Z) for single-token representation. Trains with joint language modeling + weighted RankNet loss: L = L_LM + lambda*L_Rank.
- **Training**: 40k GPT-4 labeled instances (5k MS MARCO queries, <=20 candidates each). Zephyr-beta 7B fine-tuned 3 epochs, batch 32, lr 5e-6, ~7 hours on 4 A100s.
- **Key innovation vs prior work**: LLM rerankers implicitly judge relevance in first-token logits without generating full ranking sequences; weighted RankNet loss aligned to logit-based ranking achieves both 50% inference speedup and improved accuracy.
- **Re-implementation complexity**: Low
- **Results highlight**: BEIR average nDCG@10: 78.8%; 50% inference speedup over standard listwise.
- **Access status**: Full text read

#### Paper: ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval
- **cite_id**: yoon2024listt5
- **Authors**: Soyoung Yoon et al.
- **Venue**: ACL 2024
- **arXiv/URL**: https://arxiv.org/abs/2402.15838
- **Core method**: Implements listwise reranking using Fusion-in-Decoder (FiD) architecture that encodes each query-passage pair separately with identifiers, then aggregates in the decoder to generate passage indices in increasing relevance order. Uses m-ary tournament sort with output caching for O(n+k*log(n)) complexity.
- **Training**: MS MARCO (532k queries, 8.8M passages) with binary relevance. 20 random negative groups per query. T5-base: 20k steps, lr 1e-4; T5-3B: 3k steps, lr 1e-5. Batch 256, bfloat16.
- **Key innovation vs prior work**: FiD separately encodes passages with identifiers, allowing decoder integration without positional bias (the "lost-in-the-middle" problem of LLM-based rerankers).
- **Re-implementation complexity**: Medium
- **Results highlight**: BEIR average nDCG@10: 50.9 (T5-base); 53.0 (T5-3B); +1.3 over RankT5.
- **Access status**: Full text read

#### Paper: Ranked List Truncation for Large Language Model-based Re-Ranking
- **cite_id**: meng2024rlt
- **Authors**: Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, Maarten de Rijke
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.18185
- **Core method**: Studies ranked list truncation (RLT) for LLM-based re-ranking from three perspectives: (i) assessing RLT with LLM re-ranking + lexical retrieval, (ii) impact of different first-stage retrievers on RLT, (iii) impact of different re-ranker types. Evaluates 8 RLT methods across 3 retrievers and 2 re-rankers on TREC DL 2019/2020.
- **Training**: Evaluates existing RLT methods on existing LLM rerankers; no new training.
- **Key innovation vs prior work**: Shows supervised RLT methods offer no clear advantage over simple fixed depths, especially with stronger retrievers (SPLADE++, RepLLaMA); ~30% of queries need no re-ranking with effective retrievers.
- **Re-implementation complexity**: Low
- **Results highlight**: BM25-RankLLaMA: DL19 nDCG@10 0.747, DL20 0.762 (fixed-k 1000). SPLADE++-RankLLaMA: DL19 0.773, DL20 0.778 (fixed-k 20). Oracle cut-offs always significantly better; supervised methods fail to consistently approach oracle.
- **Access status**: Full text read

#### Paper: Reason-to-Rank: Distilling Direct and Comparative Reasoning from Large Language Models for Document Reranking
- **cite_id**: ji2025reason2rank
- **Authors**: Yuelyu Ji et al.
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2410.05168
- **Core method**: Teacher-student distillation where GPT-4 (teacher) generates two types of textual explanations: direct relevance reasoning and comparative reasoning between document pairs. Student model (LLaMA 3.1 8B with LoRA) trained with combined pairwise loss + listwise KL divergence + generation cross-entropy loss (weights alpha=0.4, beta=0.4, gamma=0.2).
- **Training**: Distillation from GPT-4 reasoning into LLaMA 3.1 8B via LoRA on 32GB V100 GPUs.
- **Key innovation vs prior work**: Unifies pointwise and pairwise reasoning in a single distillation pipeline; student model provides both ranking decisions and interpretable justifications.
- **Re-implementation complexity**: Medium
- **Results highlight**: DL19 nDCG@10: 73.8; DL20 nDCG@10: 71.22.
- **Access status**: Full text read

#### Paper: Consolidating Ranking and Relevance Predictions of Large Language Models through Post-Processing
- **cite_id**: emnlp2024consolidating
- **Authors**: Le Yan, Zhen Qin, Honglei Zhuang, Rolf Jagerman, Xuanhui Wang, Michael Bendersky, Harrie Oosterhuis
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2404.11791
- **Core method**: Proposes constrained regression to find minimal perturbations of LLM-generated relevance labels while satisfying pairwise preferences from ranking prompts. Three variants: Allpair (all pairwise comparisons), SlideWin (O(kn) sliding window), TopAll (top-k vs all, also O(kn)). Consolidates LLM relevance labels with pairwise ranking abilities.
- **Training**: N/A (post-processing approach, no training required)
- **Key innovation vs prior work**: Moves beyond simple ensemble trade-off by jointly optimizing label accuracy and ranking performance via constrained regression; achieves superior performance on both dimensions simultaneously.
- **Re-implementation complexity**: Low
- **Results highlight**: TREC DL19 nDCG@10: 0.7236-0.7265; DL20: 0.7025-0.7054; TREC-Covid: 0.7943-0.8220. Relevance prediction ECE: DL19 0.1084-0.1199, DL20 0.0865-0.0966.
- **Access status**: Full text read

### 5.2 Efficient and Neural Reranking

This subsection covers work on making neural rerankers faster and more practical for deployment. While cross-encoders and LLM-based rerankers achieve high accuracy, their computational cost limits their applicability, particularly at scale or under latency constraints. Papers in this category address efficiency through model pruning, architecture simplification, distillation into smaller models, and novel training strategies that maintain ranking quality while reducing inference time. These methods are critical for closing the gap between the accuracy of expensive rerankers and the speed requirements of production retrieval systems.

#### Paper: GUITAR: Gradient Pruning toward Fast Neural Ranking
- **cite_id**: zhao2024guitar
- **Authors**: Weijie Zhao, Shulong Tan, Ping Li
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2312.16828
- **Core method**: Bi-level graph searching framework for fast neural ranking. First constructs a probable candidate set using gradient-based approximation (computing separation angle between neighbor vertices and gradient direction), then evaluates neural network only over the probable candidate set. Adaptive angle-based heuristic selects candidates within alpha*theta angle range.
- **Training**: N/A (inference-time acceleration algorithm)
- **Key innovation vs prior work**: Gradient-based candidate pruning reduces neural network evaluations by ~58% while maintaining recall; angle-based heuristic adaptively determines candidate set size.
- **Re-implementation complexity**: Medium
- **Results highlight**: 2.2-2.7x speedup over SL2G at moderate recalls; 2-4x at 95% recall. ~58% reduction in neural network evaluations on Twitch and Amazon datasets.
- **Access status**: Full text read

#### Paper: Neural Passage Quality Estimation for Static Pruning
- **cite_id**: chang2024passagequality
- **Authors**: Xuejun Chang, Debabrata Mishra, Craig Macdonald, Sean MacAvaney
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2407.12170
- **Core method**: Estimates passage quality (query-agnostic relevance) for corpus pruning using neural quality estimation. QT5 model uses "Document: [passage] Relevant: [true/false]" prompt format. Evaluates lexical, unsupervised neural, and supervised neural approaches to predict which passages are unlikely to be relevant to any query.
- **Training**: QT5 fine-tuned on MS MARCO training triples (query-ignored), cross-entropy loss with Adam optimizer, lr 5e-5, 10k iterations, batch size 16.
- **Key innovation vs prior work**: Neural quality estimation substantially outperforms lexical and unsupervised methods for static corpus pruning; QT5-Base consistently prunes 25-60% of passages while maintaining statistically equivalent effectiveness.
- **Re-implementation complexity**: Medium
- **Results highlight**: QT5-Base prunes 60% of corpus for BM25 (Dev), 30% across all pipelines (TAS-B, SPLADE, MonoELECTRA) while maintaining statistical equivalence. Random baseline can only prune 5-15%. QT5-Tiny achieves competitive results at 0.13ms/passage.
- **Access status**: Full text read

## 6. Hybrid Retrieval

Hybrid retrieval combines dense and sparse retrieval signals to exploit their complementary strengths. A common approach is score-level fusion: running BM25 (or a learned sparse model) and a dense retriever independently, then combining their scores via linear interpolation, reciprocal rank fusion, or a learned merging function. Other approaches fuse at the representation level, producing embeddings that contain both dense semantic and sparse lexical components. Hybrid methods consistently outperform either dense or sparse retrieval alone on standard benchmarks, because lexical matching captures exact term overlap that dense models sometimes miss, while dense representations handle synonymy and paraphrase that term-matching cannot. The main practical cost is running two retrieval systems, though learned sparse+dense fusion can sometimes be unified into a single model. Hybrid retrieval is used at the first stage of the pipeline and is the default recommendation when maximum recall is needed.

#### Paper: Revisiting Document Expansion and Filtering for Effective First-Stage Retrieval
- **cite_id**: mansour2024docexpansion
- **Authors**: Watheq Mansour, Shengyao Zhuang, Guido Zuccon, Joel Mackenzie
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://dl.acm.org/doi/10.1145/3626772.3657850
- **Core method**: Detailed reproducibility study of Doc2Query-- examining trade-offs in document expansion and filtering mechanisms. Successfully reproduces best-performing Doc2Query-- method, then shows filtering actually harms recall-based metrics. Explores whether two-stage "generate-then-filter" can be replaced with a single generation phase via reinforcement learning.
- **Training**: Evaluates existing Doc2Query expansion models with various filtering strategies; also explores RL-based single-stage generation.
- **Key innovation vs prior work**: Demonstrates that Doc2Query-- filtering harms recall metrics on various test collections; questions the necessity of the filtering stage in document expansion pipelines.
- **Re-implementation complexity**: Low
- **Results highlight**: Filtering in Doc2Query-- harms recall-based metrics across test collections; simple unfiltered expansion can match or exceed filtered approaches for recall-oriented evaluation.
- **Access status**: Full text read

#### Paper: MixGR: Enhancing Retriever Generalization for Scientific Domain through Complementary Granularity
- **cite_id**: cai2024mixgr
- **Authors**: Fengyu Cai et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2407.10691
- **Core method**: Improves dense retrievers' awareness of query-document matching across various granularities in a zero-shot approach. Fuses metrics from multiple granularities into a unified score reflecting comprehensive query-document similarity.
- **Training**: Zero-shot approach; no additional training required.
- **Key innovation vs prior work**: Multi-granularity fusion for zero-shot scientific document retrieval.
- **Re-implementation complexity**: Low
- **Results highlight**: Outperforms prior document retrieval by 24.7%, 9.8%, 6.9% nDCG@5 with unsupervised, supervised, and LLM-based retrievers respectively, averaged across five scientific retrieval datasets with multi-subquery queries.
- **Access status**: Abstract read (arXiv)

## 7. Retrieval-Augmented Generation (Retrieval Components)

This section focuses on the retrieval module within retrieval-augmented generation (RAG) systems, where the goal is to select the most relevant passages to feed to an LLM for answer generation. Unlike standalone retrieval, the retriever here is evaluated not only on passage relevance but on downstream generation quality -- a passage that leads the LLM to a correct answer matters more than one that is topically relevant but uninformative. Architectures range from off-the-shelf dense retrievers paired with readers, to jointly trained retrieve-and-read systems where the retriever's parameters are updated based on the reader's generation loss. Recent work explores unified models that perform both retrieval (ranking) and generation within a single LLM, and self-RAG approaches where the model decides when and what to retrieve. The key challenge is bridging the gap between retrieval relevance and generation utility.

#### Paper: RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs
- **cite_id**: yu2024rankrag
- **Authors**: Yue Yu et al.
- **Venue**: NeurIPS 2024
- **arXiv/URL**: https://arxiv.org/abs/2407.02485
- **Core method**: Implements retrieve-rerank-generate pipeline training a single LLM for both context ranking and answer generation. Ranking via probability of generating "True" for relevance prompts. Unified (x,c,y) format for all tasks. Two-stage training: SFT on 128K instruction examples, then unified instruction tuning with ~50k ranking pairs from MS MARCO plus retrieval-augmented QA.
- **Training**: Stage-I: 128 batch, 1000 steps, lr 5e-6. Stage-II: 64 batch, 3300 steps, lr 3e-7 (8B). Llama 3 backbone.
- **Key innovation vs prior work**: Adding a small fraction of ranking data to instruction tuning blend outperforms LLMs fine-tuned with 10x more ranking data alone; ranking and generation capabilities mutually enhance each other.
- **Re-implementation complexity**: Medium
- **Results highlight**: RankRAG-70B NQ EM: 54.2 (vs GPT-4 40.4); biomedical average: 78.06 (vs GPT-4 79.97); R@5: 80.3 on NQ.
- **Access status**: Full text read

#### Paper: xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token
- **cite_id**: cheng2024xrag
- **Authors**: Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, Dongyan Zhao
- **Venue**: NeurIPS 2024
- **arXiv/URL**: https://arxiv.org/abs/2405.13792
- **Core method**: Reinterprets document embeddings from dense retrieval as features from the retrieval modality, integrating them into LLM representation space via a modality bridge. Only the bridge is trainable (<0.1% of LLM params); both retriever and LLM remain frozen. Compresses retrieved documents from ~175 tokens to a single token.
- **Training**: Only modality bridge trained; retriever and LLM frozen. Allows reuse of offline-constructed document embeddings.
- **Key innovation vs prior work**: Treats retrieval embeddings as a modality (like vision tokens), enabling extreme compression while preserving plug-and-play retrieval augmentation. Improved robustness to irrelevant documents (82-85% resilience vs RAG's 75%).
- **Re-implementation complexity**: Medium
- **Results highlight**: Mistral-7b: NQ EM 39.10, TriviaQA 65.77, HotpotQA 34.05; Mixtral-8x7b: NQ 47.28, TriviaQA 74.14, HotpotQA 39.66. 3.53x FLOP reduction, 1.64x CUDA time improvement vs standard RAG.
- **Access status**: Full text read

#### Paper: RA-DIT: Retrieval-Augmented Dual Instruction Tuning
- **cite_id**: lin2024radit
- **Authors**: Xi Victoria Lin et al.
- **Venue**: ICLR 2024
- **arXiv/URL**: https://arxiv.org/abs/2310.01352
- **Core method**: Lightweight fine-tuning methodology that retrofits any LLM with retrieval capabilities via two distinct steps: (1) update pre-trained LM to better use retrieved information, (2) update retriever to return results preferred by the LM.
- **Training**: Dual instruction tuning on tasks requiring both knowledge utilization and contextual awareness.
- **Key innovation vs prior work**: Avoids expensive retrieval-specific pre-training modifications; each stage independently yields significant gains, and combining both leads to additional improvements.
- **Re-implementation complexity**: Medium
- **Results highlight**: RA-DIT 65B 0-shot: MMLU 64.6, NQ 35.2, TriviaQA 75.4 (+8.9% avg over RePlug). 5-shot: MMLU 64.9, NQ 43.9, TriviaQA 75.1. 64-shot: surpasses Atlas 11B by 4.1 pts avg. NQ 0-shot: +22% over base Llama 65B.
- **Access status**: Full text read

## 8. Retrieval Evaluation and Benchmarks

This section covers new datasets, evaluation metrics, and methodological frameworks for measuring retrieval system quality. Standard benchmarks like MS MARCO and BEIR have driven progress but have known limitations -- saturation on easy queries, narrow domain coverage, or reliance on shallow relevance judgments. The papers here introduce more challenging benchmarks that test reasoning-intensive retrieval, robustness to distribution shift, or fine-grained relevance distinctions. They also propose improved evaluation protocols, such as LLM-based relevance assessment, pairwise preference evaluation, and metrics that better correlate with downstream task performance. Rigorous evaluation infrastructure is essential because small improvements on saturated benchmarks may not reflect meaningful advances, while new benchmarks can expose real weaknesses and guide future research directions.

#### Paper: BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval
- **cite_id**: su2025bright
- **Authors**: Hongjin Su et al.
- **Venue**: ICLR 2025
- **arXiv/URL**: https://arxiv.org/abs/2407.12883
- **Core method**: First text retrieval benchmark requiring intensive reasoning (1,384 real-world queries across economics, psychology, mathematics, coding). Evaluates whether models can retrieve beyond surface-form matching.
- **Training**: N/A (benchmark)
- **Key innovation vs prior work**: Reveals massive performance gap: SFR-Embedding-Mistral scores 59.0 nDCG@10 on standard benchmarks but only 18.3 on BRIGHT. Incorporating explicit reasoning improves by up to 12.2 points.
- **Re-implementation complexity**: N/A (benchmark)
- **Results highlight**: Best model: 18.3 nDCG@10 (vs 59.0 on standard MTEB).
- **Access status**: Full text read (abstract)

#### Paper: Beyond Content Relevance: Evaluating Instruction Following in Retrieval Models
- **cite_id**: zhou2025beyondcontent
- **Authors**: Jianqun Zhou et al.
- **Venue**: ICLR 2025
- **arXiv/URL**: https://arxiv.org/abs/2410.23841
- **Core method**: InfoSearch benchmark evaluating retrieval models' instruction-following across six document-level dimensions (Audience, Keyword, Format, Language, Length, Source). 600 core queries yielding 1,598 instructed/reversed variants and 6,392 documents. Novel metrics SICR and WISE for instruction compliance.
- **Training**: N/A (benchmark/evaluation)
- **Key innovation vs prior work**: Novel evaluation framework and metrics specifically measuring instruction-following in retrieval models; reveals clear hierarchy: listwise rerankers >> pointwise rerankers >> dense retrieval >> sparse retrieval.
- **Re-implementation complexity**: N/A (benchmark)
- **Results highlight**: GPT-4o achieves WISE 33.5/SICR 32.1 (listwise reranking); GritLM achieves WISE -11.1 (dense).
- **Access status**: Full text read

#### Paper: Evaluating D-MERIT of Partial-annotation on Information Retrieval
- **cite_id**: emnlp2024dmerit
- **Authors**: Royi Rassin, Yaron Fairstein, Oren Kalinsky, Guy Kushilevitz, Nachshon Cohen, Alexander Libov, Yoav Goldberg
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2406.16048
- **Core method**: Curates D-MERIT, a passage retrieval evaluation set from Wikipedia with near-complete annotations (avg 50.44 relevant passages per query). Queries describe groups (e.g., "journals about linguistics") with relevant passages as evidence for entity membership. Evaluates how partial annotations affect retrieval system rankings.
- **Training**: N/A (benchmark/evaluation)
- **Key innovation vs prior work**: Demonstrates partial-annotation evaluation can produce misleading system rankings; provides empirical guidance for annotation completeness vs reliability trade-off.
- **Re-implementation complexity**: N/A (benchmark)
- **Results highlight**: D-MERIT: 1,196 queries, 60,333 evidence pieces. Single-relevant setup Kendall-tau: 0.936 (random selection) but 0.545-0.696 (biased selection). SPLADE++ Recall@20: 24.11%, DPR: 9.62%. When systems are significantly separated (p<0.01), single-relevant setup is reliable.
- **Access status**: Full text read

## 9. Multi-Modal Retrieval

Multi-modal retrieval extends retrieval methods beyond text-only settings to handle visual content such as images, document screenshots, slides, and infographics. These methods typically adapt dense retrieval or late interaction architectures by incorporating vision encoders (e.g., ViT, SigLIP) alongside text encoders, producing embeddings in a shared cross-modal space. Training objectives mirror those in text retrieval -- contrastive loss aligning query and relevant document representations -- but applied across modalities. A key emerging pattern is visual document retrieval, where documents are represented as page images rather than extracted text, avoiding the information loss from OCR and layout parsing. Multi-modal retrievers can serve as first-stage retrievers over visual corpora or as components in multi-modal RAG pipelines. The main challenges are the higher dimensionality of visual features and the need for paired cross-modal training data.

#### Paper: ColPali: Efficient Document Retrieval with Vision Language Models
- **cite_id**: faysse2024colpali
- **Authors**: Manuel Faysse et al.
- **Venue**: arXiv 2024 / ICLR 2025
- **arXiv/URL**: https://arxiv.org/abs/2407.01449
- **Core method**: Leverages PaliGemma-3B VLM to generate ColBERT-style multi-vector embeddings directly from document page images. Late interaction mechanism computes matching scores between query tokens and image patches, eliminating OCR/layout detection pipelines.
- **Training**: 127,460 query-page pairs (63% academic, 37% synthetic from web PDFs). Pairwise cross-entropy loss with hardest in-batch negatives. English-only training; zero-shot multilingual. PaliGemma-3B with LoRA (r=32, alpha=32).
- **Key innovation vs prior work**: Integrates late interaction with vision-language models for end-to-end document retrieval from images alone; dramatically reduces indexing complexity vs. traditional OCR pipelines.
- **Re-implementation complexity**: Medium
- **Results highlight**: ViDoRe average nDCG@5: 81.3; outperforms OCR-based pipelines significantly.
- **Access status**: Full text read

#### Paper: Unifying Multimodal Retrieval via Document Screenshot Embedding (DSE)
- **cite_id**: ma2024dse
- **Authors**: Xueguang Ma et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2406.11251
- **Core method**: Bi-encoder encoding document screenshots directly into dense vectors using Phi-3-vision (4B params). Screenshots divided into sub-images generating patch embeddings, concatenated with text prompt and processed through language model. Cosine similarity between normalized embeddings.
- **Training**: InfoNCE contrastive loss, tau=0.02. LoRA (rank 8, alpha 64) on 2 A100 GPUs. Wikipedia: batch 128, 1 epoch. Slides: batch 64, 2 epochs. Lr 1e-4.
- **Key innovation vs prior work**: Bypasses document parsing entirely by encoding original document appearance through screenshots, preserving layout, visual context, and all multimodal content simultaneously.
- **Re-implementation complexity**: Medium
- **Results highlight**: NQ top-1 accuracy: 46.2% (vs BM25 29.5%); SlideVQA nDCG@10: 75.3 (vs BM25 55.8).
- **Access status**: Full text read

#### Paper: E5-V: Universal Embeddings with Multimodal Large Language Models
- **cite_id**: jiang2024e5v
- **Authors**: Ting Jiang et al.
- **Venue**: arXiv 2024
- **arXiv/URL**: https://arxiv.org/abs/2407.12580
- **Core method**: Adapts MLLMs for universal multimodal embeddings using a single modality training approach (trained exclusively on text pairs), demonstrating improvements over multimodal training on image-text pairs while reducing training costs by ~95%.
- **Training**: Text-pair only training; no image-text pair training needed.
- **Key innovation vs prior work**: Single-modality (text-only) training achieves multimodal embedding performance, dramatically reducing cost and complexity.
- **Re-implementation complexity**: Medium
- **Results highlight**: STS average 86.00; Flickr30K image retrieval R@1 79.5% (vs CLIP ViT-L 67.3%); COCO R@1 52.0%; CIRR R@1 33.90% (vs iSEARLE-XL 25.40%); I2I-Flickr30K R@1 67.8% (vs CLIP 3.8%).
- **Access status**: Full text read

## 10. Knowledge Distillation for Retrieval

Knowledge distillation for retrieval trains smaller, faster retrieval models by transferring knowledge from larger, more accurate teacher models. The typical setup uses a cross-encoder or LLM reranker as the teacher to generate soft relevance labels (scores or rankings) for query-document pairs, which then supervise the training of a bi-encoder student via KL-divergence or margin-based losses. This is one of the most effective ways to improve bi-encoder quality: the student learns to mimic the teacher's fine-grained relevance judgments without incurring the teacher's inference cost at serving time. Distillation can also be applied iteratively, with the student's improved negatives feeding back to generate harder training signal. The approach is widely used in practice and is responsible for much of the gap between vanilla contrastive-trained bi-encoders and state-of-the-art dense retrievers.

#### Paper: MTA4DPR: Multi-Teaching-Assistants Based Iterative Knowledge Distillation for Dense Passage Retrieval
- **cite_id**: lu2024mta4dpr
- **Authors**: Qixi Lu et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://aclanthology.org/2024.emnlp-main.336/
- **Core method**: Transfers knowledge from teacher to student via multiple assistants in an iterative manner. With each iteration, the student learns from more performant assistants and more difficult data.
- **Training**: Iterative distillation with multiple teaching assistants of increasing capability.
- **Key innovation vs prior work**: Multi-assistant iterative distillation enabling a 66M student model to compete with much larger LLM-based DPR models.
- **Re-implementation complexity**: Medium
- **Results highlight**: 66M student model achieves SOTA among same-size models on multiple datasets, competitive with larger LLM-based DPR models through iterative multi-assistant distillation.
- **Access status**: Abstract + ACL Anthology page

## 11. Index Compression and Efficient Retrieval

This section covers techniques for reducing the storage footprint and computational cost of retrieval indexes while maintaining search quality. Methods include vector quantization (reducing the number of bits per dimension), dimensionality reduction (training models to produce shorter embeddings), Matryoshka-style representations (where truncated prefixes of an embedding remain useful), index pruning (removing redundant entries from inverted or vector indexes), and efficient ANN data structures. These techniques are critical for deploying retrieval systems at scale, where storing billions of full-precision 768-dimensional vectors is prohibitively expensive. The tradeoff is between compression ratio and retrieval accuracy -- aggressive compression degrades recall, so the best methods achieve high compression with minimal quality loss. These optimizations apply to both dense vector indexes and learned sparse indexes, and are often combined with quantization-aware training for best results.

#### Paper: 2D Matryoshka Training for Information Retrieval
- **cite_id**: wang2025matryoshka2d
- **Authors**: Shuai Wang et al.
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2411.17299
- **Core method**: Trains encoder model simultaneously across various layer-dimension setups, enabling flexible choice of both model depth and embedding dimensionality at inference time.
- **Training**: Multi-layer multi-dimension contrastive training with full-dimension loss.
- **Key innovation vs prior work**: Extends Matryoshka representation learning from embedding dimensions to also include model layers, creating a 2D flexibility space.
- **Re-implementation complexity**: Medium
- **Results highlight**: 2DMSE consistently outperforms standard MSE and BERT in sub-layers across all evaluated layers and dimensions. V2 variant demonstrates superior effectiveness. Optimal configuration: layer 12 with dimension 128. Full-dimension loss modification provides most substantial improvements at higher dimensions (>=128). Performance evaluated on MS MARCO (MRR@10), BEIR (nDCG@10), and STS tasks.
- **Access status**: Full text read

#### Paper: IGP: Efficient Multi-Vector Retrieval via Proximity Graph Index
- **cite_id**: sigir2025igp
- **Authors**: Zheng Bian, Man Lung Yiu, Bo Tang
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://www4.comp.polyu.edu.hk/~csmlyiu/conf/SIGIR25_IGP.pdf
- **Core method**: Leverages a Proximity Graph (PG) index, state-of-the-art in single-vector retrieval, to solve multi-vector retrieval (MVR). Develops an incremental next-similar retrieval technique for the PG index to facilitate high-quality candidate generation, producing only hundreds of candidates (vs. tens of thousands) while achieving high recall.
- **Training**: N/A (indexing algorithm)
- **Key innovation vs prior work**: Adapts proximity graph indexes (designed for single-vector score functions) to efficiently support multi-vector retrieval; achieves 2-3x query throughput improvements over state-of-the-art MVR approaches at same accuracy level.
- **Re-implementation complexity**: High (systems-level implementation)
- **Results highlight**: 2-3x query throughput improvement over existing MVR methods at equivalent accuracy. Produces only hundreds of document candidates vs. tens of thousands for prior methods.
- **Access status**: Full text read (PDF)

## 12. SIGIR 2025 Additional Papers

#### Paper: On the Scaling of Robustness and Effectiveness in Dense Retrieval
- **cite_id**: sigir2025scalingrobust
- **Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2505.24279
- **Core method**: Investigates scaling laws for both robustness (OOD and adversarial) and effectiveness in dense retrieval. Uses contrastive entropy as metric. Proposes Pareto training that dynamically adjusts optimization weights using distributionally robust optimization to balance robustness and effectiveness.
- **Training**: Various model and data size configurations on MS MARCO; Pareto training with adjusted optimization weights.
- **Key innovation vs prior work**: First to show robustness also follows power-law scaling; identifies that robustness is more sensitive to data size while effectiveness responds more to model size; proposes Pareto-efficient joint optimization.
- **Re-implementation complexity**: Medium
- **Results highlight**: Both OOD robustness and effectiveness follow power-law scaling (R-squared > 0.997). Pareto training achieves up to 2.5x improvement in scaling efficiency. Robustness more affected by data variations; effectiveness more responsive to model size.
- **Access status**: Full text read

#### Paper: TITE: Token-Independent Text Encoder for Information Retrieval
- **cite_id**: sigir2025tite
- **Authors**: Ferdinand Schlatt, Maik Fröbe, Matthias Hagen
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://downloads.webis.de/publications/papers/schlatt_2025d.pdf
- **Core method**: Uses attention-based pooling to iteratively reduce the sequence length of hidden states layer by layer so the final output is already a single sequence representation vector. Eliminates wasted computation from discarded non-CLS token embeddings in standard bi-encoders.
- **Training**: Standard bi-encoder training with modified architecture; models available on HuggingFace (webis/tite-2-late).
- **Key innovation vs prior work**: Token-independent encoding that avoids computing and then discarding non-CLS embeddings, achieving up to 3.3x speedup while maintaining effectiveness.
- **Re-implementation complexity**: Medium
- **Results highlight**: On par with standard bi-encoder retrieval models on TREC DL19, DL20, and BEIR while being up to 3.3x faster at encoding queries and documents.
- **Access status**: Full text read

#### Paper: Efficient Re-ranking with Cross-encoders via Early Exit
- **cite_id**: sigir2025earlyexit
- **Authors**: Francesco Busolin, Claudio Lucchese, Franco Maria Nardini, Salvatore Orlando, Raffaele Perego, Salvatore Trani, Alberto Veneri
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://dl.acm.org/doi/10.1145/3726302.3729962
- **Core method**: SEE (Selective Early Exit) method uses multiple classifiers positioned before transformer blocks that early-stop non-relevant documents. A filter processes all documents, reordering them based on embedding similarities, then progressively terminates processing for documents identified as non-relevant.
- **Training**: Cross-encoder with additional early exit classifiers trained on re-ranking data.
- **Key innovation vs prior work**: Enables early termination of non-relevant documents during cross-encoder processing, reducing computational costs while maintaining ranking quality.
- **Re-implementation complexity**: Medium
- **Results highlight**: Reduces cross-encoder computation by enabling early exit for non-relevant documents while maintaining nDCG effectiveness on TREC DL benchmarks.
- **Access status**: Abstract + code (https://github.com/veneres/SEE-SIGIR25)

#### Paper: From Vector Representations to Neural Representations: Learned Query-Specific Relevance Functions (Hypencoder)
- **cite_id**: sigir2025neuralrep
- **Authors**: Julian Killingback, Hansi Zeng, Hamed Zamani
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2502.05364
- **Core method**: Replaces vector inner products with query-dependent neural networks (Hypencoder). A hypernetwork generates weights for a small "q-net" conditioned on the query, which then scores documents. Proves theoretically that there exist document groups not linearly separable by standard similarity functions, motivating query-specific neural scoring.
- **Training**: Contrastive training with distillation; surpasses DRAGON despite DRAGON using 40x more training resources and 5-teacher curriculum distillation.
- **Key innovation vs prior work**: Fundamental shift from fixed similarity functions to query-dependent neural scoring; theoretical proof that vector inner products cannot separate all relevant document groups.
- **Re-implementation complexity**: High
- **Results highlight**: TREC DL19+20 nDCG@10: 0.736 (vs TAS-B 0.700, CL-DRD 0.701); MS MARCO RR@10: 0.386 (vs TAS-B 0.344); TREC DL-HARD nDCG@10: 0.630; only model with positive p-MRR on FollowIR.
- **Access status**: Full text read

#### Paper: Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information
- **cite_id**: sigir2025pointwise
- **Authors**: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, Ting Wang
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2506.10859
- **Core method**: GCCP (Global-Consistent Comparative Pointwise Ranking) creates an anchor document via spectral-based multi-document summarization as a reference point for comparing candidates, enabling implicit pairwise comparisons with pointwise efficiency. PAGC (Post-Aggregation with Global Context) linearly combines GCCP scores with existing pointwise methods in a training-free framework.
- **Training**: Zero-shot (no training); training-free post-aggregation.
- **Key innovation vs prior work**: Incorporates global context into pointwise ranking via anchor document comparison, bridging gap between pointwise efficiency and comparative effectiveness without training.
- **Re-implementation complexity**: Low
- **Results highlight**: PAGC-QSG (Flan-T5-xl): TREC DL19 nDCG@10 0.7068, DL20 0.6872, BEIR average 0.4916. Significantly outperforms prior pointwise methods with only ~0.3s additional latency per query.
- **Access status**: Full text read

#### Paper: Scaling Sparse and Dense Retrieval in Decoder-Only LLMs
- **cite_id**: sigir2025scalingparadigms
- **Authors**: Hansi Zeng, Julian Killingback, Hamed Zamani
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2502.15526
- **Core method**: Systematic comparative study of how sparse and dense retrieval paradigms scale with decoder-only LLM size (Llama-3 1B, 3B, 8B). Evaluates contrastive learning (CL), knowledge distillation (KD), and their combination across in-domain (MS MARCO, TREC DL) and out-of-domain (BEIR) benchmarks.
- **Training**: MS MARCO passages with CL, KD, and CL+KD objectives across Llama-3 1B/3B/8B scales with fixed compute budget.
- **Key innovation vs prior work**: First systematic comparison showing sparse retrieval significantly outperforms dense in zero-shot generalization (10.5% on BEIR with KD, 4.3% with CL at 8B scale); scaling behaviors emerge clearly only with CL.
- **Re-implementation complexity**: Medium
- **Results highlight**: Lion-SP-8B: MRR@10 0.417 (MS MARCO), nDCG@10 0.758 (TREC DL19+20), 0.552 (BEIR). Lion-DS-8B: MRR@10 0.417, nDCG@10 0.755 (DL), 0.501 (BEIR). Sparse outperforms dense by 10.5% on BEIR zero-shot.
- **Access status**: Full text read

#### Paper: Pre-training vs. Fine-tuning: A Reproducibility Study on Dense Retrieval Knowledge Acquisition
- **cite_id**: sigir2025pretrainfinetune
- **Authors**: Zheng Yao, Shuai Wang, Guido Zuccon
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2505.07166
- **Core method**: Reproducibility study extending prior work on whether retrieval knowledge is primarily gained during pre-training. Tests across representation approaches (CLS tokens vs mean pooling), backbone architectures (encoder-only BERT vs decoder-only LLaMA), and additional datasets (MS MARCO + NQ).
- **Training**: Various configurations of DPR, Contriever, RepLlama across pre-training and fine-tuning setups.
- **Key innovation vs prior work**: Shows that DPR's "decentralization" finding (pre-training primarily determines retrieval knowledge) does not generalize to mean-pooled (Contriever) or decoder-based (LLaMA) models; architecture-dependent behavior rather than universal principle.
- **Re-implementation complexity**: Medium
- **Results highlight**: DPR fine-tuning increases intermediate-layer neuron activations by 32-41% (supporting decentralization), but Contriever shows uniform patterns and RepLlama shows reduced activations, contradicting universality. Linear probing: BERT-CLS generally outperforms DPR variants.
- **Access status**: Full text read

## 13. Additional Notable Papers

#### Paper: Unsupervised Large Language Model Alignment for Information Retrieval via Contrastive Feedback (RLCF)
- **cite_id**: dong2024rlcf
- **Authors**: Qian Dong et al.
- **Venue**: SIGIR 2024
- **arXiv/URL**: https://arxiv.org/abs/2309.17078
- **Core method**: Proposes RLCF (Reinforcement Learning from Contrastive Feedback) to align LLMs for IR. Constructs unsupervised contrastive feedback signals from similar document groups and uses a group-wise reciprocal rank reward function within PPO to optimize LLMs for generating high-quality, context-specific responses.
- **Training**: Unsupervised alignment via PPO with contrastive feedback; no relevance labels needed.
- **Key innovation vs prior work**: Unsupervised LLM alignment specifically for IR using contrastive document group feedback, enabling LLMs to distinguish relevant documents from similar candidates.
- **Re-implementation complexity**: Medium
- **Results highlight**: Sparse retrieval: NDCG@10 avg 0.302 (3.4% improvement). Dense retrieval: NDCG@10 avg 0.245 (10.4% improvement; 26.5% on ArguAna). Document summarization: Rouge-diff 32.2 on LCSTS (vs vanilla 22.1), 32.5 on Gigaword (vs 11.9).
- **Access status**: Full text read

#### Paper: SPLADE-v3: New Baselines for SPLADE
- **cite_id**: lassance2024spladev3
- **Authors**: Carlos Lassance et al.
- **Venue**: arXiv 2024
- **arXiv/URL**: https://arxiv.org/abs/2403.06789
- **Core method**: Updated training structure for SPLADE learned sparse retrieval models, incorporating improved hard-negative mining, cross-encoder distillation, and training strategies. Represents the third iteration of the SPLADE family.
- **Training**: Hard-negative mining + distillation from cross-encoder teachers with MaxPooling aggregation.
- **Key innovation vs prior work**: Improved training recipe building on SPLADE-v2 with better negative sampling and distillation strategies.
- **Re-implementation complexity**: Medium
- **Results highlight**: SPLADE-v3: MRR@10 40.2 on MS MARCO (vs SPLADE++SelfDistil 37.6); BEIR nDCG@10 51.7 (vs 50.7); TREC DL19 72.3, DL20 75.4. DistilBERT variant: MRR@10 38.7, DL19 75.2, DL20 74.4.
- **Access status**: Full text read

#### Paper: Taxonomy-guided Semantic Indexing for Academic Paper Search (TaxoIndex)
- **cite_id**: kang2024taxoindex
- **Authors**: SeongKu Kang et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://arxiv.org/abs/2410.19218
- **Core method**: Extracts key concepts from papers and organizes them as a semantic index guided by an academic taxonomy, then leverages this index to identify academic concepts and link queries and documents. Plug-and-play framework enhancing existing dense retrievers.
- **Training**: Can be applied with limited training data as a plug-in to existing dense retrievers.
- **Key innovation vs prior work**: Taxonomy-guided concept extraction and indexing that enhances dense retrieval for academic search; significant improvements even with highly limited training data.
- **Re-implementation complexity**: Medium
- **Results highlight**: CSFCube (SPECTER-v2): nDCG@10 0.417 (vs full fine-tuning 0.368), MAP@10 0.198, Recall@50 0.633. DORIS-MAE (Contriever-MS): nDCG@10 0.421 (vs fine-tuning 0.407). Requires only 6.7% of backbone parameters.
- **Access status**: Full text read

#### Paper: Improve Dense Passage Retrieval with Entailment Tuning
- **cite_id**: ludai2024entailment
- **Authors**: Stella Lu Dai et al.
- **Venue**: EMNLP 2024
- **arXiv/URL**: https://github.com/stellaludai/EntailmentTuning
- **Core method**: Redefines relevance in QA-oriented retrieval by aligning it with entailment from NLI tasks. Augments dense retriever training with NLI data using a masked hypothesis prediction scheme, converting questions into narrative-form existence claims.
- **Training**: Standard dense retriever training augmented with NLI data and unified retrieval/NLI formats.
- **Key innovation vs prior work**: Integrates NLI entailment signal into dense retriever training via rule-based question-to-claim transformations.
- **Re-implementation complexity**: Low
- **Results highlight**: NQ passage retrieval MRR: BERT 64.51->67.24, RoBERTa 62.75->64.24, RetroMAE 66.12->67.75, Condenser 66.34->67.89. Top-1 hits improvement up to 3.32% for BERT. Downstream QA: +0.5% on NQ (T5-base).
- **Access status**: Full text read

#### Paper: QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking
- **cite_id**: sigir2025qder
- **Authors**: Shubham Chatterjee, Jeff Dalton
- **Venue**: SIGIR 2025
- **arXiv/URL**: https://arxiv.org/abs/2510.11589
- **Core method**: Unifies entity-oriented approaches with multi-vector models via "late aggregation" - maintaining individual token and entity representations throughout ranking, aggregating only at final scoring. Dual-channel architecture: text (BERT) + entity (Wikipedia2Vec) channels with dynamic attention, addition/multiplication interaction operations, and bilinear scoring with BM25 hybrid.
- **Training**: Multi-vector training with entity embeddings; combines knowledge graph semantics with fine-grained token representations.
- **Key innovation vs prior work**: Late aggregation of query-specific token and entity representations; particularly excels on difficult queries where traditional methods fail completely.
- **Re-implementation complexity**: Medium
- **Results highlight**: TREC Robust04 (title queries): nDCG@20 0.7694 (vs CEDR 0.5475, +36%), MAP 0.6082 (vs 0.3701, +64%), MRR 0.9751. TREC Core 2018: nDCG@20 0.6562. Achieves nDCG@20 0.70 on hard queries where baselines score 0.0.
- **Access status**: Full text read

---

## Papers Needing Manual Access

The following papers have limited access (PDF-only without extractable detailed result tables):

| cite_id | Title | Venue | Status |
|---------|-------|-------|--------|
| katsimpras2024genra | GENRA: Enhancing Zero-shot Retrieval | EMNLP 2024 | Abstract read; PDF at ACL Anthology not parseable for specific numbers |
| lu2024mta4dpr | MTA4DPR: Multi-Teaching-Assistants KD | EMNLP 2024 | Abstract read; PDF not parseable for detailed result tables |
| luo2024llmfoundations | Large Language Models as Foundations for Next-Gen Dense Retrieval | EMNLP 2024 | Abstract + search summaries; full paper available at arXiv |
