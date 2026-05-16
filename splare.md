Sections:
Abstract
1 Introduction
2 Background
    2.1 Sparse Autoencoders
    2.2 SPLADE
3 Method
    3.1 SPLARE
    3.2 Training
        Training LSR Models.
        Distillation vs Contrastive Learning.
        Sparsity.
4 Experimental Setup
    Training Data.
    Evaluation.
5 Analysis and Design Choices for SPLARE Models
    Performance and Layer Depth.
    How does the Width of the SAE Affect Retrieval Performance?
    Effectiveness–Efficiency Trade-Off.
    Comparison of Lexical and Latent Features.
6 Multilingual Models
    6.1 Comparing Latent Models to Lexicon-based Approaches
    6.2 Comparing to Top Models
    6.3 Interpretability: Mechanistic Interpretation of SPLARE
7 Related Works
    LLMs and Retrieval.
    Sparse Autoencoders and Retrieval.
8 Conclusion
References
Appendix A Experimental Setting
    English Setting.
    Multilingual Setting.
Appendix B Hyper-parameters
    SAE Choice.
Appendix C English-only SPLARE Full Results
Appendix D Full Results
Appendix E Latency Measures
Appendix F SPLADE Layer Ablation
Appendix G Retrieval Examples
## Contents
- 1 Introduction
- 2 Background
  - 2.1 Sparse Autoencoders
  - 2.2 SPLADE
- 3 Method
  - 3.1 SPLARE
  - 3.2 Training
    - Training LSR Models.
    - Distillation vs Contrastive Learning.
    - Sparsity.
- 4 Experimental Setup
  - Training Data.
  - Evaluation.
- 5 Analysis and Design Choices for SPLARE Models
  - Performance and Layer Depth.
  - How does the Width of the SAE Affect Retrieval Performance?
  - Effectiveness–Efficiency Trade-Off.
  - Comparison of Lexical and Latent Features.
- 6 Multilingual Models
  - 6.1 Comparing Latent Models to Lexicon-based Approaches
  - 6.2 Comparing to Top Models
  - 6.3 Interpretability: Mechanistic Interpretation of SPLARE
- 7 Related Works
  - LLMs and Retrieval.
  - Sparse Autoencoders and Retrieval.
- 8 Conclusion
- References
- Appendix A Experimental Setting
  - English Setting.
  - Multilingual Setting.
- Appendix B Hyper-parameters
  - SAE Choice.
- Appendix C English-only SPLARE Full Results
- Appendix D Full Results
- Appendix E Latency Measures
- Appendix F SPLADE Layer Ablation
- Appendix G Retrieval Examples

## Abstract

Abstract Sparse autoencoders (SAEs) provide a powerful mechanism for decomposing the dense representations produced by Large Language Models (LLMs) into interpretable latent features. We posit that SAEs constitute a natural foundation for Learned Sparse Retrieval (LSR), whose objective is to encode queries and documents into high-dimensional sparse representations optimized for efficient retrieval. In contrast to existing LSR approaches that project input sequences into the vocabulary space, SAE-based representations offer the potential to produce more semantically structured, expressive, and language-agnostic features. Building on this insight, we introduce SPLARE, a method to train SAE-based LSR models. Our experiments, relying on recently released open-source SAEs, demonstrate that this technique consistently outperforms vocabulary-based LSR in multilingual and out-of-domain settings. SPLARE-7B, a multilingual retrieval model capable of producing generalizable sparse latent embeddings for a wide range of languages and domains, achieves top results on MMTEB’s multilingual and English retrieval tasks. We also developed a 2B-parameter variant with a significantly lighter footprint.

## 1 Introduction

Embedding models have become a pivotal tool for search systems, enabling the better capture of semantic relationships between queries and documents across various domains and modalities. This trend has been further accelerated by the advent of Retrieval-Augmented Generation (RAG) [53] and agent-based systems, which impose even higher demands on retrieval performance and robustness. Recently, dense embedding models [80, 39], which map inputs into single dense vectors, have demonstrated impressive performance on the (M)MTEB benchmark [66, 23]. Specifically, embedding models relying on large (V)LLM backbones have become the de facto approach for generalist multilingual [51, 101, 88, 50, 56] or even multimodal models [30, 25, 94]—marking a shift away from encoder-only language models which have defined the state of the art for years [35, 39, 93].

Learned Sparse Retrieval (LSR) methods [26, 64, 70, 43] have achieved state-of-the-art performance on widely used English-centric benchmarks [86, 2, 14] and have demonstrated strong generalization when compared to dense embedding models [28, 60, 18]. Beyond their efficiency, these approaches provide a level of interpretability that is particularly valuable in production systems. Models such as SPLADE [26, 27, 49] operationalize this idea by representing documents and queries as sparse, weighted bag-of-words over the vocabulary space of their backbone model. While originally developed for encoder-only architectures such as BERT [20], recent work has explored adapting SPLADE to LLM backbones [78, 21, 95, 98, 84, 61]. However, these models remain limited to English-centric contexts and struggle to match state-of-the-art performance on more comprehensive benchmarks like MMTEB which place greater emphasis on generalization across novel domains and languages. Unlike dense retrieval, which models relevance within a continuous embedding space, LSR methods are inherently constrained by the fixed vocabulary of their underlying backbone, which incurs issues such as tokenization redundancy [52]. This limitation also makes it significantly harder to handle multilingual or cross-lingual retrieval [69, 68, 46]—and even more so when extending to multi-modal settings [71]. We hypothesize that this is a key reason why LSR models have recently fallen behind dense approaches(^1^11For instance, as of the time of writing, no sparse retrieval model is listed on the MTEB (Multilingual, v2) leaderboard.).

In the context of LLMs, Sparse Autoencoders (SAEs) [63, 34, 5] decompose dense token representations into sparse vectors of latent features. These features have been shown to exhibit desirable properties: they are largely mono-semantic (most features correspond to a single interpretable concept), multilingual (remaining largely language-agnostic), and even multimodal (generalizing across modalities in multimodal LLMs) [5, 85, 57, 34, 31, 19]. While SAEs have generated significant excitement for mechanistic interpretability, recent work has also highlighted their limitations, showing that they can struggle to transfer effectively to certain downstream tasks [38, 83].

In this work, we argue and empirically demonstrate that SAEs are a natural fit for LSR models: their learned latent features provide a semantically-grounded representation space for sparse retrieval which is particularly advantageous in domains or languages where vocabulary-based approaches may underperform. To this end, we propose a new LSR approach that represents queries and documents as sparse vectors over a latent vocabulary space, by replacing the standard language modeling (LM) head with pre-trained SAEs such as Llama Scope [31]. More specifically, our contributions are as follows:

- •
We introduce SPLARE—for SParse LAtent REtrieval—a new LSR approach relying on pre-trained SAEs;
- •
We conduct a systematic investigation of the advantages of using a latent vocabulary—compared to the standard LLM vocabulary—across a comprehensive set of benchmarks spanning diverse tasks, domains, and languages;
- •
Finally, we introduce a new 7B multilingual latent sparse retriever that supports 100+ languages through cross-lingual transfer and achieves competitive results on the MMTEB *retrieval* benchmark. SPLARE is the first LSR model to rival state-of-the-art dense approaches on MMTEB, *by fine-tuning only on a large open-source dataset, without additional pretraining or data augmentation.*

## 2 Background

We first provide some background on Sparse Autoencoders as well as Learned Sparse Retrieval. SPLARE can be understood as synthesizing these two research directions into a unified framework.

### 2.1 Sparse Autoencoders

Given activations $x\in\mathbb{R}^{d}$ from a language model, a Sparse Autoencoder (SAE) is a single hidden layer model, comprising an encoder and a decoder:

$$ $z=f(\boldsymbol{W}_{\text{enc}}x+\boldsymbol{b}_{\text{enc}}),\quad\hat{x}=\boldsymbol{W}_{\text{dec}}z+\boldsymbol{b}_{\text{dec}}$ (1) $$

where $z\in\mathbb{R}^{|\mathcal{W}|}$, with $|\mathcal{W}|>>d$ corresponding to the width of SAE, i.e., the number of features in the latent space. SAEs, as a class of autoencoders, are trained using a standard reconstruction objective $\mathcal{L}=\|\hat{x}-x\|^{2}$. Sparsity in the decomposition is induced through suitable activation functions $f$ such as ReLU [5], Top-K [63, 29] or JumpReLU [79], and/or regularization penalties such as $\ell_{1}$. Several works have demonstrated that SAEs can recover highly monosemantic features, many of which are language-agnostic—responding consistently to the same concepts across languages—and, in some cases, even multimodal [34, 5, 85, 57, 31, 15, 19]. Large Sparse Autoencoders are also notoriously hard and costly to train. Recently, high-quality large scale open-source SAEs have become available to the research community. In particular, we rely in this work on the Llama Scope series of models [31] which offers SAEs trained on Llama-3.1-8B and the Gemma Scope suite [57] which offers SAEs trained on Gemma-2-2B, 9B and 27B models.

### 2.2 SPLADE

Learned Sparse Retrieval (LSR) models aim to map input sequences into high-dimensional sparse representations for efficient retrieval. Among these approaches, the SPLADE family of approaches [26, 27, 49] has emerged as the state-of-the-art method, achieving performance comparable to or exceeding that of dense embedding models in many settings. Given an input sequence tokenized as $t=(t_{1},t_{2},\ldots,t_{n})$ and fed through all the layers of the transformer, SPLADE generates a sequence of logits $(v_{1},v_{2},\ldots,v_{n})$ by projecting each final hidden state $(h_{1},h_{2},\ldots,h_{n})$ onto the vocabulary space $\mathcal{V}$ using the language modeling head. The weights ($v_{ij})_{j\in\mathcal{V}}$ correspond to an unnormalized log-probability distribution over $\mathcal{V}$ for token $t_{i}$, where each output dimension $j$ is actually associated with the token it represents.
To obtain a single sequence-level representation, SPLADE first applies a term saturation function, before max-pooling over the sequence:

$$ $u_{j}=\max_{i=1\ldots n}\log\left(1+\textrm{ReLU}(v_{ij})\right),j\in\mathcal{V}$ (2) $$

Given these sparse representations $u\in\mathbb{R}^{|\mathcal{V}|}$ for queries and documents, relevance scores are computed as a sparse dot product $s(q,d)=<u^{q},u^{d}>$. This operation can be efficiently supported using inverted index structures together with specialized query processing techniques [87, 8, 103].

## 3 Method

### 3.1 SPLARE

Conceptually, SPLARE closely parallels SPLADE but operates in the latent representation space. Rather than projecting the final hidden states of the language model onto the vocabulary space via the LM head, SPLARE employs sparse autoencoders to transform representations from a selected layer into a sparse latent space, which can be interpreted as a latent vocabulary.

Let $(\boldsymbol{W}_{\text{enc}},\boldsymbol{b}_{\text{enc}})$ in Eq. [1](#S2.E1) denote the SAE’s encoder parameters at a given layer $l$ of the transformer(^2^22Note that we only rely on the encoder parameters, as we only aim to extract sparse features from representations. Also note that we consider SAEs trained on the residual streams of the transformer.). Similarly to SPLADE, we can obtain sequences of sparse latent logits $(w_{1},w_{2},\ldots,w_{n})$ by mapping the hidden states at layer $l$ with the SAE encoder. The weights ($w_{ij})_{j\in\mathcal{W}}\in\mathbb{R}^{|\mathcal{W}|}$ contain the sparse list of latent features associated with token $i$ in the input sequence. It can be used in place of the vocabulary decomposition to compute sequence-level representations for input queries or documents into a sparse set of latent features using the same type of pooling mechanism as in Eq. [2](#S2.E2)—which we refer to as SPLADE-pool.

### 3.2 Training

#### Training LSR Models.

The training procedure for LSR models mirrors that of dense embedding models. While contrastive learning [74, 12] is the de-facto approach to train state-of-the-art dense models [51, 101], we instead adopt a distillation-based approach using a cross-encoder teacher model [72] to train our sparse embeddings. Specifically, we optimize the Kullback–Leibler divergence between the teacher and student relevance distributions [59]. Given a query $q$, $(d_{1},d_{2},\ldots d_{m})$ which contains a positive document and a pool of hard negatives, $(s_{1},s_{2},\ldots s_{m})$ the corresponding teacher scores for documents $d_{i}$ with respect to $q$, and $\tau$ a temperature parameter, the training loss is given by:

$$ $\displaystyle\mathcal{L}_{\mathrm{KL}}$ $\displaystyle=\sum_{i=1}^{m}p_{i}\left(\log p_{i}-\log\hat{p}_{i}\right)$ (3) $\displaystyle\hat{p}_{i}$ $\displaystyle=\frac{e^{s(q,d_{i})/\tau}}{\sum_{j}e^{s(q,d_{j})/\tau}}$ $\displaystyle p_{i}$ $\displaystyle=\frac{e^{s_{i}}}{\sum_{j}e^{s_{j}}}$ $$

#### Distillation vs Contrastive Learning.

Distillation is a common toolbox to train retrieval models [32, 59], but has been overlooked in the context of LLM-based embeddings. State-of-the-art embedding models typically rely on contrastive learning, which is generally effective but suffers from well-known issues such as false negatives. As a result, many recent systems incorporate filtering mechanisms for negative samples—often using cross-encoders or LLMs—which can themselves be viewed as implicit forms of distillation [51, 17]. In addition, since our retrieval model mirrors SPLADE (with the vocabulary projection being the only difference), we follow the established training practices for SPLADE [27, 49].

#### Sparsity.

To encourage sparsity in query and document representations, LSR models are typically trained with a sparsity-inducing regularization term, analogous to that used in SAEs. We use the FLOPS loss [76] employed in SPLADE. The final loss is:

$$ $\mathcal{L}=\mathcal{L}_{\mathrm{KL}}+\lambda_{q}\ell^{q}_{\text{FLOPS}}+\lambda_{d}\ell^{d}_{\text{FLOPS}}$ (4) $$

The sparsity of LSR approaches plays a crucial role in determining both effectiveness and computational efficiency on retrieval benchmarks. However, the sparsity induced by $\mathcal{L}$ can vary significantly depending on the model configuration, backbone architecture, SAE suite, and dataset characteristics. Achieving a desired target sparsity would require continuous adjustment of $\lambda_{d,q}$. To mitigate this challenge and establish a more robust training setup, we additionally apply Top-K pooling *at inference time*. This strategy allows us to train a single model with moderate sparsity—using fixed, conservative values of $\lambda_{d,q}$—while systematically studying the effect of pooling without the need for re-training. Although some prior works have entirely replaced explicit sparsity regularization with Top-K pooling [48, 21], our initial experiments with this approach yielded inferior results. Note that Top-K acts as a strict upper bound: depending on the dataset, queries and documents may contain fewer active dimensions due to the inherent sparsity of the representations.

Finally, we note that while SPLARE is initialized with an SAE—which produces sparse token-level representations—sequence-level sparsity at initialization remains relatively high (e.g., a few thousands non-zero values). As a result, additional sparsity regularization is required to ensure the model achieves the desired efficiency. It is also worth noting that LSR models are usually hard to train and require a careful initialization of the projection head. While the LM head or a SAE can provide a suitable initialization, training an LSR model entirely from scratch is highly difficult and consistently results in much lower performance—when converging.

## 4 Experimental Setup

#### Training Data.

We conduct two large sets of experiments: § [5](#S5) contains various ablations and analyses for models trained on English data on the MS MARCO dataset [2]. In § [6](#S6), we further extend training to a larger set of publicly available data, including multilingual datasets. We do not prepend any special instructions or prefix to our input sequences—which could only likely yield further improvements. To ease reproducibility, we also refrain from any form of pre-finetuning or synthetic data generation [51, 30, 101], both of which have recently become common practice for achieving top results on the MTEB benchmark. We detail in Appendix [A](#A1) our two training settings.

#### Evaluation.

MTEB [66] and MMTEB [23] are the most widely adopted benchmarks for evaluating embedding models. Our evaluation focuses only on the *retrieval* subsets of these benchmarks, excluding other task categories. In addition to the English and Multilingual splits, we also report results on domain-specific subsets of MTEB, including Code, Medical, Law, and Chemical domains. Given SPLARE’s strong performance in multilingual settings, we further place particular emphasis on this aspect by including language-specific splits of MMTEB for five languages, as well as evaluations on the MIRACL [100] and XTREME-UP [81] datasets. The latter introduces a challenging cross-lingual retrieval task, requiring retrieval from an English corpus using queries from low-resource languages. We also report results on MS MARCO [2] and BEIR [86] (Appendix [C](#A3)).

While our approach is broadly applicable to any pre-trained SAE, we conduct the majority of our experiments using the Llama Scope model suite [31], built on Llama-3.1-8B [24]. During training, we fine-tune the backbone with LoRA adapters [33] while keeping SAE parameters frozen. Preliminary experiments indicated that this strategy not only improves performance but also simplifies training. Moreover, it preserves the interpretability of the latent feature space [58]. As in prior work [98, 3, 52], we enable bidirectional attention across all backbones and pretrain them with Masked Next Token Prediction. Following the exact procedure of [98], we mask 20% of tokens in the MS MARCO corpus and train for $10k$ steps which takes about five hours. Bidirectional attention is particularly important for LSR models since pooling occurs at every position of the input sequence, unlike dense models that rely on the <EOS> token. Full details of our experimental hyperparameters are provided in Appendix [B](#A2). Unless stated otherwise, retrieval evaluation is performed using Top-K pooling, with default values of $k=40$ for queries and $k=400$ for documents. For our multilingual models (§ [6](#S6)), we additionally rely on model averaging [92] from several training runs, which boosts generalization performance [51, 101].

We are mainly interested in comparing SPLARE to current state-of-the-art LSR methods, which are all vocabulary-based. To this end, we perform controlled comparisons with a SPLADE model built on the same Llama-3.1-8B backbone—following the methodology of [21, 98]—and trained under identical settings. We refer to this baseline as SPLADE-Llama.

## 5 Analysis and Design Choices for SPLARE Models

Figure: Figure 1: (Left) Performance across layers on Llama Scope (Llama-3.1-8B) and Gemma Scope (Gemma-2-2B). (Right) Performance with increasing SAE width on Gemma-2. Evaluation done with $\text{Top-K}=(40,400)$.
Refer to caption: https://arxiv.org/html/2603.13277/2603.13277v1/x1.png

We first conduct a series of ablation studies in a controlled, English-only setting. At this stage, our primary objective is to compare SPLARE’s latent representations with traditional vocabulary-based approaches (i.e., our SPLADE-Llama baseline). Specifically, we aim to address the following research questions:
(i) At which transformer layer depth do we obtain the most effective sparse latent representations for retrieval?
(ii) How does the width of the SAE affect retrieval performance?
(iii) What are the efficiency–effectiveness trade-offs introduced by the latent vocabulary?
(iv) Do the sparse latent features learned by the SAE yield improvements over equivalent SPLADE models?

#### Performance and Layer Depth.

We train SPLARE models at varying depths on Llama-3.1-8B, using SAEs from Llama Scope with two widths $|\mathcal{W}|\in\{32k,131k\}$, and on Gemma-2-2B, using Gemma Scope with width $|\mathcal{W}|=65k$, and report the average MTEB (English, v2) performance in Figure [1](#S5.F1) (Left). Interestingly, the highest performance is consistently achieved at about two-thirds of the model depth, i.e., around layer 20 (out of 32) for Llama Scope and 16 (out of 26) for Gemma Scope. These findings are consistent with prior work suggesting that intermediate transformer layers often yield richer representations for retrieval tasks [82, 102, 90]. A further advantage of using intermediate layers is the reduction in retriever size and, consequently, inference latency—an improvement over SPLADE models, which require processing through all layers of the LLM (see Appendix [F](#A6)).

While the best performance often appears at relatively deep layers, this choice is not critical: performance at earlier layers remains strong, and the selection ultimately reflects an effectiveness-efficiency trade-off. A practical rule of thumb is to select a layer around two-thirds of the model depth to maximize effectiveness. For the remainder of the paper, our main SPLARE models are trained at layer 26 of Llama-3.1-8B, yielding a 7B-parameter model (including the SAE parameters), but we also train a strong 2B model (layer 6) in § [6](#S6)(^3^33Unless otherwise specified, SPLARE refers to SPLARE-7B.).

#### How does the Width of the SAE Affect Retrieval Performance?

Unlike SPLADE models, the dimensionality of SPLARE’s feature space—determined by the SAE width $|\mathcal{W}|$—is not constrained by the LLM’s vocabulary size. To study the effect of SAE width on retrieval effectiveness, we train multiple SPLARE models using Gemma Scope, which offers a broader range of SAE configurations. Especially, we consider SAEs at layers 12 and 19 of Gemma-2-2B with widths $|\mathcal{W}|\in\{2^{14}\approx 16k,2^{15},\ldots,2^{20}\approx 1M\}$. We report the resulting average MTEB (English, v2) performance in Figure [1](#S5.F1) (Right). Our results show a roughly log-linear relationship between SAE width and retrieval effectiveness, providing a scaling mechanism for improved performance—something not possible with SPLADE’s fixed vocabulary size. Prior work has shown that SAEs can scale to widths as large as 14$M$ on very large LLMs [85], though such models remain proprietary. Llama Scope, while limited to $|\mathcal{W}|\in\{32k,131k\}$, exhibits the same scaling effect consistently across layers (Figure [1](#S5.F1), (Left)).
These experiments also highlight that the approach is transferable across different backbone architectures. Despite the availability of much wider SAEs in Gemma Scope, we observe that Llama Scope models achieve superior overall performance. Consequently, we report results using this model (with $|\mathcal{W}|=131k$) for all subsequent experiments.

Figure: Figure 2: (Left) Impact of pruning documents with Top-K (with $k=40$ for queries). (Right) MS MARCO index distribution for SPLARE and SPLADE (8.8$M$ documents).
Refer to caption: https://arxiv.org/html/2603.13277/2603.13277v1/x3.png

#### Effectiveness–Efficiency Trade-Off.

Sparse retrieval methods achieve efficiency through the use of dedicated inverted index structures and exact [103, 87] or approximate [6] query processing algorithms. In all cases, obtaining highly sparse representations is critical for achieving low-latency retrieval. While SPLADE has been successfully adapted to LLM backbones, efficiency considerations have generally been overlooked. As discussed in § [3.2](#S3.SS2), LSR models can easily become “dense” in practical scenarios, which undermines their efficiency.

We study the relationship between SPLARE performance and sparsity by capping, at inference time, the number of activated features for document vectors using Top-K pooling. Results are shown in Figure [2](#S5.F2) (Left). SPLARE exhibits substantially greater robustness to document pruning: when indexing only $\text{Top-K}=100$ document features, its performance drops by merely $2\%$, compared to over $6\%$ for SPLADE.
This difference can be partially attributed to SPLARE’s more compact and structured latent feature space as well as the fact that SPLADE models based on LLMs are inherently harder to sparsify. As we show in Appendix [E](#A5), this difference translates into lower query latency at a given accuracy level, when evaluated using Seismic [6, 7, 9]. For reference, performing retrieval with SPLARE ($\text{Top-K}=(40,400)$) on MS MARCO ($8.8M$ documents) requires about $5$*ms* per query only—without accounting model inference. Figure [2](#S5.F2) (Right) further illustrates the distributions of activated features after training. Notably, SPLARE utilizes a much larger portion of the available feature space, activating nearly all dimensions, in contrast to SPLADE, which relies on fewer than $100k$ dimensions (out of $128k$). Moreover, SPLARE exhibits a more balanced activation distribution across features. By comparison, SPLADE tends to over-activate a small subset of dimensions [62, 52].

#### Comparison of Lexical and Latent Features.

Finally, we compare the performance of SPLARE with existing top LSR methods trained on the English MS MARCO dataset. In particular, Lion-SP-8B [98] represents the most effective contemporary SPLADE adaptation for LLM-based retrieval. We show the results on Table [1](#S5.T1.2) for various splits of MTEB (English Models). First, notice that SPLADE-Llama (our baseline) already significantly outperforms Lion-SP-8B (e.g, +4.4 on MTEB English). We further observe that SPLARE consistently outperforms competing methods on both multilingual and several out-of-domain evaluation sets. In particular, it achieves an improvement of roughly two points on the multilingual split and shows superior performance on the Law and Medical retrieval benchmarks—though its advantage diminishes on the Code and Chemical splits. The observed multilingual generalization from English-only training is unsurprising, given the language-agnostic nature of SAE features. Meanwhile, the performance drop on the Code tasks is likely due to the highly domain-specific nature of code retrieval, which does not align well with the features learned by the SAE (a trend that is further supported by our observations in § [6](#S6)). To illustrate this behavior, we provide in Appendix [G](#A7) (Figures [G](#A7)-[G](#A7)) several examples where SPLARE underperforms compared to SPLADE on MTEB Code. In these cases, the top activated features appear overly generic rather than specialized to code semantics. This suggests that for highly domain-specific scenarios such as code retrieval, dedicated SAEs trained on code-focused corpora may be more appropriate. We leave this direction for future work.

**Table 1: Average performance on various MTEB splits. English models are trained on MS MARCO only (§ [5](#S5)). Multilingual models are trained on a large-scale multilingual training set (§ [6](#S6)). Evaluation done with $\text{Top-K}=(40,400)$.**
|  | English | Multilingual | Code | Medical | Law | ChemTEB |
| --- | --- | --- | --- | --- | --- | --- |
| English Models |  |  |  |  |  |  |
| SPLADE-v3 [49] | 50.7 | 38.1 | 44.5 | 44.2 | 40.4 | 75.6 |
| Lion-SP-8B [98] | 48.5 | 50.0 | 53.3 | 54.4 | 48.5 | 71.1 |
| SPLADE-Llama | 52.9 | 54.3 | 57.3 | 61.0 | 49.0 | 75.9 |
| SPLARE | 52.9 | 56.3 | 55.1 | 62.9 | 51.2 | 70.0 |
| Multilingual Models |  |  |  |  |  |  |
| SPLADE-Llama | 58.9 | 61.7 | 64.3 | 67.6 | 60.7 | 77.4 |
| SPLARE | 59.3 | 62.3 | 63.0 | 67.7 | 60.8 | 78.1 |

## 6 Multilingual Models

In § [5](#S5), we showed how the latent feature space of the SAE offers some advantages for LSR models—when compared to the vocabulary space—*in a controlled English-based setting*. In § [6.1](#S6.SS1), we further extend those findings for multilingual models, by training models on a large-scale multilingual dataset and broadening the evaluation to cover a more diverse set of benchmarks, as detailed in § [3.2](#S3.SS2). In § [6.2](#S6.SS2), we compare SPLARE to concurrent models on (M)MTEB and XTREME-UP.

### 6.1 Comparing Latent Models to Lexicon-based Approaches

Table [1](#S5.T1.2) (Multilingual models) reports the average performance of multilingual SPLARE and SPLADE across the various MTEB splits, with complete results available in Appendix [D](#A4). Overall, SPLARE consistently outperforms its vocabulary-based counterpart, except on the Code split—an outcome aligned with the discussion in § [5](#S5). Notably, the performance gap is more substantial on the Multilingual split of MTEB (+0.6 average nDCG@10). Examining the *multilingual-only* datasets within this split confirms that SPLARE systematically surpasses SPLADE (Table [8](#A4.T8.2), +0.9 points). This trend is confirmed by Table [2](#S6.T2.2), which underscores the advantage of latent-based LSR models in multilingual scenarios (e.g., +2.4 on XTREME-UP and +1.8 on MIRACL). Comprehensive results for MIRACL and XTREME-UP, together with comparisons to concurrent methods, are presented in Appendix [D](#A4) and Table [3](#S6.T3.4), respectively. SPLARE achieves particularly strong performance on the hidden test sets of MIRACL (Table [13](#A4.T13), de and *yo*) as well as on the low-resource languages of XTREME-UP.

**Table 2: Multilingual comparison of SPLARE and SPLADE ($\text{Top-K}=(40,400)$).**
|  | indic | sca | deu | fra | kor | XTREME-UP | MIRACL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPLADE-Llama | 91.9 | 70.4 | 57.3 | 65.6 | 74.8 | 56.2 | 69.9 |
| SPLARE | 92.3 | 70.8 | 57.1 | 64.8 | 76.0 | 58.6 | 71.7 |

### 6.2 Comparing to Top Models

Finally, we compare SPLARE to top models from the MTEB leaderboard in Table [3](#S6.T3.4). SPLARE reaches an average score of 62.3 (for the pooled version), making it among the top 10 models on MTEB(Multilingual, v2) retrieval and the top-1 LSR model. Notably, these results are achieved without relying on private or synthetic data and without any pre-finetuning. This is also particularly interesting, as open models like gte-Qwen2-7B instruct or NV-Embed-v2 rely on 3584-$d$ (resp. 4096-$d$) dense vectors to encode queries and documents, while SPLARE only needs $40$ features (resp. 400) to encode queries (resp. documents) in its high-dimensional feature space to reach high effectiveness (Top-K at inference time). We also observe an average gain of $+1.5$ points for the non-pooled version, albeit at the cost of higher retrieval complexity. On the other hand, extremely sparse models ($\text{Top-K}=(10,100)$) still offer competitive performance. Note that in practical retrieval scenarios, dense embeddings often require dimensionality-reduction techniques [44] and/or approximate nearest-neighbor search algorithms [36]—whose performance degradation is rarely reported on standard benchmarks. In contrast, sparse retrieval methods natively support efficient exact search without incurring such compromises. Finally, we also report results for a SPLARE model trained at layer 6 (SPLARE-2B). Although its performance is somewhat lower than that of the full SPLARE model (7B parameters), it remains strong—particularly on the XTREME-UP dataset. Importantly, this model is substantially more efficient and therefore offers a different, and often attractive, point on the effectiveness–efficiency trade-off curve.

**Table 3: Average MTEB retrieval performance of SPLARE (Multilingual) against top models. Multilingual (resp. Eng) refers to MTEB(Multilingual, v2) (resp. MTEB(eng, v2)). At the time of writing (February 27, 2026), SPLARE ranks in the top-10 models on MTEB(Multilingual, v2) retrieval. For XTREME-UP (MRR@10), we report results from [51]. Unless specified, evaluation for SPLARE is done with $\text{Top-K}=(40,400)$—corresponding to our default model SPLARE or SPLARE-2B.**
|  | English | Multilingual | XTREME-UP |
| --- | --- | --- | --- |
| Top Open Source models |  |  |  |
| e5-mistral-7b-instruct [88] | 57.6 | 55.8 | - |
| NV-Embed-v2 [50] | 62.8 | 56.7 | - |
| multilingual-e5-large-instruct [89] | 53.5 | 57.1 | 18.7 |
| GritLM-7B [67] | 55.0 | 58.3 | - |
| SFR-Embedding-Mistral [65] | 59.3 | 59.4 | - |
| Linq-Embed-Mistral [41] | 60.1 | 58.7 | 24.6 |
| gte-Qwen2-7B-instruct [56] | 58.1 | 60.1 | 17.4 |
| voyage-3-large [1] | 53.5 | 66.1 | 39.2 |
| jina-embeddings-v4 [30] | 56.2 | 66.4 | - |
| inf-retriever-v1 [96] | 64.1 | 66.5 | - |
| Qwen-3-Embedding-8B [101] | 69.4 | 70.9 | - |
| Commercial models |  |  |  |
| Cohere-embed-multilingual-v3.0 [13] | 55.7 | 59.2 | - |
| text-embedding-3-large [75] | 58.0 | 59.3 | 18.8 |
| gemini-embedding-001 [51] | 64.4 | 67.7 | 64.3 |
| SPLARE | 59.3 | 62.3 | 58.6 |
| SPLARE | no-pooling | 61.4 | 63.8 | 61.4 |
| SPLARE | $\text{Top-K}=(20,200)$ | 55.9 | 59.9 | 53.8 |
| SPLARE | $\text{Top-K}=(10,100)$ | 50.1 | 56.0 | 46.5 |
| SPLARE-2B | 55.9 | 59.1 | 41.6 |

### 6.3 Interpretability: Mechanistic Interpretation of SPLARE

Finally, we provide interpretability insights for SPLARE. We leverage Neuronpedia [58] to obtain explanations for individual SAE features—which, as a reminder, remain frozen during fine-tuning—and list the top features contributing to a document’s relevance with respect to a given query. For SPLADE, by contrast, we report the tokens with the highest relevance contributions. Figure [6.3](#S6.SS3) illustrates a cross-lingual example from XTREME-UP from Tamil to English. The features activated by SPLARE align well with meaningful concepts present in both the query and document. They correspond to coherent, language-agnostic concepts which combine into a comprehensive description of the data point. In contrast, SPLADE exhibits a higher degree of redundancy (e.g., separate activations for “Indian” and “indian”) and predominantly relies on Latin-script tokens— effectively defaulting to English subword representations—which provide less informative signals in this setting. Further examples are given in Appendix [G](#A7).

[Uncaptioned image]: https://arxiv.org/html/2603.13277/2603.13277v1/interpretability_examples/tamil_img.png

## 7 Related Works

#### LLMs and Retrieval.

Dense embedding models derived from LLMs have demonstrated substantial gains over traditional BERT-style encoders [51, 101]. Recent approaches such as LLM2Vec [3] or GritLM [67] highlight how LLMs can be effectively adapted into powerful text encoders by incorporating bi-directional attention. Beyond providing stronger backbone architectures, LLMs have also significantly advanced retrieval model training, enabling the generation of high-quality synthetic data and improved filtering of training samples [88, 50, 51, 101, 16].
Nonetheless, despite the impressive progress of dense embeddings, controlled evaluations have shown that they can still be outperformed by alternative architectures such as multi-vector models or sparse retrievers [98, 25, 10].

#### Sparse Autoencoders and Retrieval.

Sparse autoencoders have primarily been employed in Information Retrieval (IR) to approximate dense representations for efficient nearest-neighbor search. Given a dense embedding model, these approaches learn to map query and document vectors into sparse latent representations that preserve the structure of the original embedding space [47, 4, 77, 37, 91]. SAEs have also been used to interpret dense embeddings in both IR [73] and Recommender Systems [40, 42].
Most closely related to our work is [77], which shows that SAE-derived features can serve as effective indexing units. However, all prior studies train SAEs on top of an *already-trained dense retriever*. In contrast, our approach leverages pre-trained SAEs on the base LLM and fine-tunes an LSR model directly in a SPLADE-like fashion, allowing for tighter integration of relevance and sparsity when training the sparse representations.

## 8 Conclusion

In this work, we investigated two complementary research directions: Sparse autoencoders and Learned Sparse Retrieval models. We demonstrated that SAEs provide a natural foundation for LSR by yielding semantically rich and multilingual latent features that overcome the vocabulary dependence of traditional LSR approaches. Our experiments show that SAE-based LSR models consistently outperform vocabulary-based counterparts, particularly in multilingual and out-of-domain scenarios. Finally, we introduced SPLARE, a competitive 7B-parameter multilingual model capable of producing generalizable sparse latent embeddings, thereby paving the way for more robust, versatile, and cross-lingual retrieval across diverse domains and modalities.

## References

- AI [2025]
Voyage AI.
Voyage-3 large.
[https://blog.voyageai.com/2025/01/07/voyage-3-large/](https://blog.voyageai.com/2025/01/07/voyage-3-large/), 2025.
Accessed: 2025-09-24.
- Bajaj et al. [2018]
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang.
Ms marco: A human generated machine reading comprehension dataset, 2018.
- BehnamGhader et al. [2024]
Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy.
LLM2vec: Large language models are secretly powerful text encoders.
In *First Conference on Language Modeling*, 2024.
- Borges et al. [2023]
Luís Borges, Bruno Martins, and Jamie Callan.
Kale: Using a k-sparse projector for lexical expansion.
In *Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval*, page 13–22, New York, NY, USA, 2023. Association for Computing Machinery.
- Bricken et al. [2023]
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah.
Towards monosemanticity: Decomposing language models with dictionary learning.
*Transformer Circuits Thread*, 2023.
https://transformer-circuits.pub/2023/monosemantic-features/index.html.
- Bruch et al. [2024a]
Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini.
Efficient inverted indexes for approximate retrieval over learned sparse representations.
In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 152–162, 2024a.
- Bruch et al. [2024b]
Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini.
Pairing clustered inverted indexes with $\kappa$-nn graphs for fast approximate retrieval over learned sparse representations.
In *Proceedings of the 33rd International ACM Conference on Information and Knowledge Management (CIKM)*, pages 3642–3646. ACM, 2024b.
- Bruch et al. [2024c]
Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini.
Efficient inverted indexes for approximate retrieval over learned sparse representations.
In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, pages 152–162. ACM, 2024c.
- Bruch et al. [2025]
Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini, and Leonardo Venuta.
Investigating the scalability of approximate sparse retrieval algorithms to massive datasets.
In *Advances in Information Retrieval*, pages 437–445. Springer Nature Switzerland, 2025.
- Chen et al. [2024a]
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2024a.
- Chen et al. [2024b]
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation.
In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 2318–2335, Bangkok, Thailand, 2024b. Association for Computational Linguistics.
- Chen et al. [2020]
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
A simple framework for contrastive learning of visual representations.
In *International conference on machine learning*, pages 1597–1607. PmLR, 2020.
- Cohere [2023]
Cohere.
Introducing embed v3.
[https://cohere.com/blog/introducing-embed-v3](https://cohere.com/blog/introducing-embed-v3), 2023.
Accessed: 2025-09-24.
- Craswell et al. [2021]
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos.
Overview of the trec 2020 deep learning track, 2021.
- Cunningham and Conerly [2024]
Hoagy Cunningham and Tom Conerly.
Comparing topk and gated saes to standard saes.
*Transformer Circuits Thread*, 2024.
- Dai et al. [2023]
Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith Hall, and Ming-Wei Chang.
Promptagator: Few-shot dense retrieval from 8 examples.
In *The Eleventh International Conference on Learning Representations*, 2023.
- de Souza P. Moreira et al. [2025]
Gabriel de Souza P. Moreira, Radek Osmulski, Mengyao Xu, Ronay Ak, Benedikt Schifferer, and Even Oldridge.
Nv-retriever: Improving text embedding models with effective hard-negative mining, 2025.
- Déjean et al. [2023]
Hervé Déjean, Stephane Clinchant, Carlos Lassance, Simon Lupart, and Thibault Formal.
Benchmarking middle-trained language models for neural search.
In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 1848–1852, New York, NY, USA, 2023. Association for Computing Machinery.
- Deng et al. [2025]
Boyi Deng, Yu Wan, Baosong Yang, Yidan Zhang, and Fuli Feng.
Unveiling language-specific features in large language models via sparse autoencoders.
In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4563–4608, Vienna, Austria, 2025. Association for Computational Linguistics.
- Devlin et al. [2019]
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
BERT: Pre-training of deep bidirectional transformers for language understanding.
In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota, 2019. Association for Computational Linguistics.
- Doshi et al. [2024]
Meet Doshi, Vishwajeet Kumar, Rudra Murthy, Vignesh P, and Jaydeep Sen.
Mistral-splade: Llms for better learned sparse retrieval, 2024.
- Déjean et al. [2024]
Hervé Déjean, Stéphane Clinchant, and Thibault Formal.
A thorough comparison of cross-encoders and llms for reranking splade, 2024.
- Enevoldsen et al. [2025]
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemiński, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Veysel Çağatan, Akash Kundu, Martin Bernstorff, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświata, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Suppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal A Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Mariya Hendriksen, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri K, Maksimova Anna, Silvan Wehrli, Maria
Tikhonova, Henil Shalin Panchal, Aleksandr Abramov, Malte Ostendorff, Zheng Liu, Simon Clematide, Lester James Validad Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, and Niklas Muennighoff.
MMTEB: Massive multilingual text embedding benchmark.
In *The Thirteenth International Conference on Learning Representations*, 2025.
- et al. [2024]
Aaron Grattafiori et al.
The llama 3 herd of models, 2024.
- Faysse et al. [2025]
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, CELINE HUDELOT, and Pierre Colombo.
Colpali: Efficient document retrieval with vision language models.
In *The Thirteenth International Conference on Learning Representations*, 2025.
- Formal et al. [2021]
Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant.
Splade: Sparse lexical and expansion model for first stage ranking.
In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 2288–2292, New York, NY, USA, 2021. Association for Computing Machinery.
- Formal et al. [2022a]
Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant.
From distillation to hard negative sampling: Making sparse neural ir models more effective.
In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 2353–2359, New York, NY, USA, 2022a. Association for Computing Machinery.
- Formal et al. [2022b]
Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant.
Match your words! a study of lexical matching in neural information retrieval.
In *Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022, Stavanger, Norway, April 10–14, 2022, Proceedings, Part II*, page 120–127, Berlin, Heidelberg, 2022b. Springer-Verlag.
- Gao et al. [2025]
Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu.
Scaling and evaluating sparse autoencoders.
In *The Thirteenth International Conference on Learning Representations*, 2025.
- Günther et al. [2025]
Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang, and Han Xiao.
jina-embeddings-v4: Universal embeddings for multimodal multilingual retrieval, 2025.
- He et al. [2024]
Zhengfu He, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, Yu-Gang Jiang, and Xipeng Qiu.
Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders, 2024.
- Hofstätter et al. [2020]
Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury.
Improving efficient neural ranking models with cross-architecture knowledge distillation.
*arXiv preprint arXiv:2010.02666*, 2020.
- Hu et al. [2022]
Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-rank adaptation of large language models.
In *International Conference on Learning Representations*, 2022.
- Huben et al. [2024]
Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey.
Sparse autoencoders find highly interpretable features in language models.
In *The Twelfth International Conference on Learning Representations*, 2024.
- Izacard et al. [2022]
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave.
Unsupervised dense information retrieval with contrastive learning.
*Transactions on Machine Learning Research*, 2022.
- Johnson et al. [2019]
Jeff Johnson, Matthijs Douze, and Hervé Jégou.
Billion-scale similarity search with GPUs.
*IEEE Transactions on Big Data*, 7(3):535–547, 2019.
- Kang et al. [2025]
Hao Kang, Tevin Wang, and Chenyan Xiong.
Interpret and control dense retrieval with sparse latent features.
In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)*, pages 700–709, Albuquerque, New Mexico, 2025. Association for Computational Linguistics.
- Kantamneni et al. [2025]
Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda.
Are sparse autoencoders useful? a case study in sparse probing.
In *Forty-second International Conference on Machine Learning*, 2025.
- Karpukhin et al. [2020]
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
Dense passage retrieval for open-domain question answering.
In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online, 2020. Association for Computational Linguistics.
- Kasalický et al. [2025]
Petr Kasalický, Martin Spišák, Vojtěch Vančura, Daniel Bohuněk, Rodrigo Alves, and Pavel Kordík.
The future is sparse: Embedding compression for scalable retrieval in recommender systems.
In *Proceedings of the Nineteenth ACM Conference on Recommender Systems*, page 1099–1103, New York, NY, USA, 2025. Association for Computing Machinery.
- Kim et al. [2024]
Junseong Kim, Seolhwa Lee, Jihoon Kwon, Sangmo Gu, Yejin Kim, Minkyung Cho, Jy yong Sohn, and Chanyeol Choi.
Linq-embed-mistral:elevating text retrieval with improved gpt data through task-specific control and quality refinement.
Linq AI Research Blog, 2024.
- Klenitskiy et al. [2025]
Anton Klenitskiy, Konstantin Polev, Daria Denisova, Alexey Vasilev, Dmitry Simakov, and Gleb Gusev.
Sparse autoencoders for sequential recommendation models: Interpretation and flexible control, 2025.
- Kong et al. [2023]
Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Michael Bendersky.
Sparseembed: Learning sparse lexical representations with contextual embeddings for retrieval.
In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 2399–2403, New York, NY, USA, 2023. Association for Computing Machinery.
- Kusupati et al. [2022]
Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al.
Matryoshka representation learning.
In *Advances in Neural Information Processing Systems*, 2022.
- Kwiatkowski et al. [2019]
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.
Natural questions: A benchmark for question answering research.
*Transactions of the Association for Computational Linguistics*, 7:452–466, 2019.
- Lassance [2023]
Carlos Lassance.
Extending english ir methods to multi-lingual ir, 2023.
- Lassance et al. [2021]
Carlos Lassance, Thibault Formal, and Stéphane Clinchant.
Composite code sparse autoencoders for first stage retrieval.
In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 2136–2140, New York, NY, USA, 2021. Association for Computing Machinery.
- Lassance et al. [2023]
Carlos Lassance, Simon Lupart, Hervé Déjean, Stéphane Clinchant, and Nicola Tonellotto.
A static pruning study on sparse neural retrievers.
In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 1771–1775, New York, NY, USA, 2023. Association for Computing Machinery.
- Lassance et al. [2024]
Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant.
Splade-v3: New baselines for splade.
*arXiv preprint arXiv:2403.06789*, 2024.
- Lee et al. [2025a]
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping.
NV-embed: Improved techniques for training LLMs as generalist embedding models.
In *The Thirteenth International Conference on Learning Representations*, 2025a.
- Lee et al. [2025b]
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, and Tom Duerig.
Gemini embedding: Generalizable embeddings from gemini, 2025b.
- Lei et al. [2025]
Yibin Lei, Tao Shen, Yu Cao, and Andrew Yates.
Enhancing lexicon-based text embeddings with large language models.
In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 18986–19001, Vienna, Austria, 2025. Association for Computational Linguistics.
- Lewis et al. [2020]
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive nlp tasks.
In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, Red Hook, NY, USA, 2020. Curran Associates Inc.
- Li et al. [2023a]
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao.
Making large language models a better foundation for dense retrieval, 2023a.
- Li et al. [2025]
Chaofan Li, Minghao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Defu Lian, Yingxia Shao, and Zheng Liu.
Making text embedders few-shot learners.
In *The Thirteenth International Conference on Learning Representations*, 2025.
- Li et al. [2023b]
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang.
Towards general text embeddings with multi-stage contrastive learning, 2023b.
- Lieberum et al. [2024]
Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, Janos Kramar, Anca Dragan, Rohin Shah, and Neel Nanda.
Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2.
In *Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP*, pages 278–300, Miami, Florida, US, 2024. Association for Computational Linguistics.
- Lin [2023]
Johnny Lin.
Neuronpedia: Interactive reference and tooling for analyzing neural networks, 2023.
Software available from neuronpedia.org.
- Lin et al. [2020]
Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin.
Distilling dense representations for ranking using tightly-coupled teachers.
*arXiv preprint arXiv:2010.11386*, 2020.
- Lupart et al. [2023]
Simon Lupart, Thibault Formal, and Stéphane Clinchant.
Ms-shift: An analysis of ms marco distribution shifts on neural retrieval.
In *Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part I*, page 636–652, Berlin, Heidelberg, 2023. Springer-Verlag.
- Ma et al. [2025]
Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, and Songlin Hu.
Lightretriever: A llm-based hybrid retrieval architecture with 1000x faster query inference, 2025.
- Mackenzie et al. [2023]
Joel Mackenzie, Andrew Trotman, and Jimmy Lin.
Efficient document-at-a-time and score-at-a-time query evaluation for learned sparse representations.
*ACM Trans. Inf. Syst.*, 41(4), 2023.
- Makhzani and Frey [2013]
Alireza Makhzani and Brendan Frey.
K-sparse autoencoders.
*arXiv preprint arXiv:1312.5663*, 2013.
- Mallia et al. [2021]
Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto.
Learning passage impacts for inverted indexes.
In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 1723–1727, 2021.
- Meng et al. [2024]
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz.
Sfr-embedding-mistral:enhance text retrieval with transfer learning.
Salesforce AI Research Blog, 2024.
- Muennighoff et al. [2023]
Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers.
MTEB: Massive text embedding benchmark.
In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 2014–2037, Dubrovnik, Croatia, 2023. Association for Computational Linguistics.
- Muennighoff et al. [2024]
Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela.
Generative Representational Instruction Tuning, 2024.
arXiv:2402.09906 [cs].
- Nair et al. [2022]
Suraj Nair, Eugene Yang, Dawn J. Lawrie, James Mayfield, and Douglas W. Oard.
Learning a sparse representation model for neural clir.
In *DESIRES*, pages 53–64, 2022.
- Nair et al. [2023]
Suraj Nair, Eugene Yang, Dawn Lawrie, James Mayfield, and Douglas W. Oard.
Blade: Combining vocabulary pruning and intermediate pretraining for scaleable neural clir.
In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 1219–1229, New York, NY, USA, 2023. Association for Computing Machinery.
- Nguyen et al. [2023]
Thong Nguyen, Sean MacAvaney, and Andrew Yates.
A unified framework forlearned sparse retrieval.
In *Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part III*, page 101–116, Berlin, Heidelberg, 2023. Springer-Verlag.
- Nguyen et al. [2024]
Thong Nguyen, Mariya Hendriksen, Andrew Yates, and Maarten de Rijke.
Multimodal learned sparse retrieval with probabilistic expansion control.
In *Advances in Information Retrieval: 46th European Conference on Information Retrieval, ECIR 2024, Glasgow, UK, March 24–28, 2024, Proceedings, Part II*, page 448–464, Berlin, Heidelberg, 2024. Springer-Verlag.
- Nogueira and Cho [2020]
Rodrigo Nogueira and Kyunghyun Cho.
Passage re-ranking with bert, 2020.
- O’Neill et al. [2024]
Charles O’Neill, Christine Ye, Kartheik G. Iyer, and John F Wu.
Towards interpretable scientific foundation models: Sparse autoencoders for disentangling dense embeddings of scientific concepts.
In *Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges*, 2024.
- Oord et al. [2018]
Aaron van den Oord, Yazhe Li, and Oriol Vinyals.
Representation learning with contrastive predictive coding.
*arXiv preprint arXiv:1807.03748*, 2018.
- OpenAI [2024]
OpenAI.
text-embedding-3-large and new embedding models.
[https://openai.com/index/new-embedding-models-and-api-updates/](https://openai.com/index/new-embedding-models-and-api-updates/), 2024.
Accessed: 2025-09-25.
- Paria et al. [2020]
Biswajit Paria, Chih-Kuan Yeh, Ian EH Yen, Ning Xu, Pradeep Ravikumar, and Barnabás Póczos.
Minimizing flops to learn efficient sparse representations.
*arXiv preprint arXiv:2004.05665*, 2020.
- Park et al. [2025]
Seongwan Park, Taeklim Kim, and Youngjoong Ko.
Decoding dense embeddings: Sparse autoencoders for interpreting and discretizing dense retrieval, 2025.
- Qiao et al. [2025]
Jingfen Qiao, Thong Nguyen, Evangelos Kanoulas, and Andrew Yates.
Leveraging decoder architectures for learned sparse retrieval, 2025.
- Rajamanoharan et al. [2024]
Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda.
Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders, 2024.
- Reimers and Gurevych [2019]
Nils Reimers and Iryna Gurevych.
Sentence-bert: Sentence embeddings using siamese bert-networks.
In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 2019.
- Ruder et al. [2023]
Sebastian Ruder, Jonathan H. Clark, Alexander Gutkin, Mihir Kale, Min Ma, Massimo Nicosia, Shruti Rijhwani, Parker Riley, Jean-Michel A Sarr, Xinyi Wang, John Wieting, Nitish Gupta, Anna Katanova, Christo Kirov, Dana L. Dickinson, Brian Roark, Bidisha Samanta, Connie Tao, David I. Adelani, Vera Axelrod, Isaac Caswell, Colin Cherry, Dan Garrette, Reeve Ingle, Melvin Johnson, Dmitry Panteleev, and Partha Talukdar.
XTREME-UP: A user-centric scarce-data benchmark for under-represented languages.
In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 1856–1884, Singapore, 2023. Association for Computational Linguistics.
- Skean et al. [2025]
Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv.
Layer by layer: Uncovering hidden representations in language models.
In *Forty-second International Conference on Machine Learning*, 2025.
- Smith et al. [2025]
Lewis Smith, Senthooran Rajamanoharan, Arthur Conmy, Callum McDougall, Tom Lieberum, János Kramár, Rohin Shah, and Neel Nanda.
Negative results for saes on downstream tasks and deprioritising sae research (gdm mech interp team progress update 2).
[https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/sae-progress-update-2-draft](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/sae-progress-update-2-draft), 2025.
- Soares et al. [2023]
Livio Soares, Daniel Gillick, Jeremy Cole, and Tom Kwiatkowski.
NAIL: Lexical retrieval indices with efficient non-autoregressive decoders.
In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 2574–2589, Singapore, 2023. Association for Computational Linguistics.
- Templeton et al. [2024]
Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan.
Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet.
*Transformer Circuits Thread*, 2024.
- Thakur et al. [2021]
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych.
BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.
In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.
- Tonellotto et al. [2018]
Nicola Tonellotto, Craig Macdonald, Iadh Ounis, et al.
Efficient query processing for scalable web search.
*Foundations and Trends® in Information Retrieval*, 12(4-5):319–500, 2018.
- Wang et al. [2024a]
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei.
Improving text embeddings with large language models.
In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 11897–11916, Bangkok, Thailand, 2024a. Association for Computational Linguistics.
- Wang et al. [2024b]
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei.
Multilingual e5 text embeddings: A technical report.
*arXiv preprint arXiv:2402.05672*, 2024b.
- Wang et al. [2025]
Shuai Wang, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon.
2d matryoshka training for information retrieval.
In *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 3125–3134, New York, NY, USA, 2025. Association for Computing Machinery.
- Wen et al. [2025]
Tiansheng Wen, Yifei Wang, Zequn Zeng, Zhong Peng, Yudi Su, Xinyang Liu, Bo Chen, Hongwei Liu, Stefanie Jegelka, and Chenyu You.
Beyond matryoshka: Revisiting sparse coding for adaptive representation.
In *Forty-second International Conference on Machine Learning*, 2025.
- Wortsman et al. [2022]
Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt.
Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time, 2022.
- Xiong et al. [2020]
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk.
Approximate nearest neighbor negative contrastive learning for dense text retrieval.
*arXiv preprint arXiv:2007.00808*, 2020.
- Xu et al. [2025a]
Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Osmulski, Yauhen Babakhin, Zhiding Yu, Benedikt Schifferer, and Even Oldridge.
Llama nemoretriever colembed: Top-performing text-image retrieval model, 2025a.
- Xu et al. [2025b]
Zhichao Xu, Aosong Feng, Yijun Tian, Haibo Ding, and Lin Lee Cheong.
Csplade: Learned sparse retrieval with causal language models, 2025b.
- Yang et al. [2025]
Junhan Yang, Jiahe Wan, Yichen Yao, Wei Chu, Yinghui Xu, and Yuan Qi.
inf-retriever-v1 (revision 5f469d7), 2025.
- Yang et al. [2018]
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning.
HotpotQA: A dataset for diverse, explainable multi-hop question answering.
In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium, 2018. Association for Computational Linguistics.
- Zeng et al. [2025]
Hansi Zeng, Julian Killingback, and Hamed Zamani.
Scaling sparse and dense retrieval in decoder-only llms.
In *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 2679–2684, New York, NY, USA, 2025. Association for Computing Machinery.
- Zhang et al. [2021]
Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin.
Mr. TyDi: A multi-lingual benchmark for dense retrieval.
In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pages 127–137, Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics.
- Zhang et al. [2023]
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin.
MIRACL: A multilingual retrieval dataset covering 18 diverse languages.
*Transactions of the Association for Computational Linguistics*, 11:1114–1131, 2023.
- Zhang et al. [2025]
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025.
- Zhuang et al. [2025]
Shengyao Zhuang, Shuai Wang, Fabio Zheng, Bevan Koopman, and Guido Zuccon.
Starbucks-v2: Improved training for 2d matryoshka embeddings, 2025.
- Zobel and Moffat [2006]
Justin Zobel and Alistair Moffat.
Inverted files for text search engines.
*ACM Comput. Surv.*, 38(2):6–es, 2006.

## Appendix A Experimental Setting

We detail below the training sets used for the English and Multilingual settings.

#### English Setting.

For our ablation study, we restrict training to the MS MARCO dataset, given the computational cost associated with training 7B-parameter models. Our experimental setup closely follows that of SPLADE-v3 [49]. For each training query, we mine hard negatives using a SPLADE model and derive distillation targets from reranking scores produced by an open-source DeBERTa-v3 reranker [22]. This controlled setting is designed to enable a direct and fair comparison between SPLARE and its vocabulary-based counterpart, SPLADE-Llama.

#### Multilingual Setting.

In this more compute-intensive setting, we use the same training set employed for the bge-multilingual-gemma2 model [55](^4^44hanhainebula/bge-multilingual-gemma2-data). This corpus includes several English-centric public datasets (e.g., MS MARCO [2], NQ [45], and HotPotQA [97]), a large collection of Chinese retrieval datasets, and two multilingual benchmarks: MIRACL [100] and Mr.TyDi [99]. Since we rely on distillation for training, we only keep samples from this dataset which were annotated using the BGE multilingual reranker [11, 54](^5^55BAAI/bge-reranker-v2-m3 reranker). After filtering, the final training set comprises approximately $1.3M$ queries with hard negatives. Notably, some of these datasets correspond to training splits of several MTEB benchmark tasks. While this may constrain the strict evaluation of generalization, this practice has become standard in prior work on general-purpose embedding models [50, 88, 3].

## Appendix B Hyper-parameters

Table [4](#A2.T4) gives the hyper-parameters used to train and evaluate SPLARE models and other baselines. Note that the temperature parameter $\tau$ is critical and needs to be adapted to each SAE suite. For instance, the optimal $\tau$ is different between Llama Scope or Gemma Scope. This depends on the scale of the logits and the initial sparsity of the SAE. For ill-suited $\tau$, it can happen that models actually diverge—for instance, collapse of the $\ell_{0}$.
To determine the optimal temperature, we ran a grid search over the values $\{1,10,20,40,50,80,100\}$, and used NanoBEIR’s nDCG@10 as an evaluation criterion for all models(^6^66[https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6)).

#### SAE Choice.

Gemma Scope contains multiple SAEs for the same layer and width, but with different $\ell_{0}$. In practice, we observed that the initial SAE’s $\ell_{0}$ had no critical effect on final performance—most likely because we fine-tune the backbone LLM. We use SAEs with $\ell_{0}$ closest to 100 throughout the paper. Additionally, Llama and Gemma Scope contain residual SAEs as well as MLP and attention stream SAEs—we only use residual SAEs in this study.

**Table 4: Hyper-parameters.**
| Component | Value |
| --- | --- |
| LoRA rank $r$ | 64 |
| Max training sequence length (english models) | 128 |
| Max training sequence length (multilingual models) | 512 |
| Epochs | 1 |
| Batch size (w/ gradient accumulation) | 128 |
| Learning rate | $5\times 10^{-5}$ |
| Warmup ratio | 0.01 |
| Nb negatives per query | 8 |
| $\lambda_{d}$ | 0.0001 |
| $\lambda_{q}$ | 0.0001 |
| $\tau$ SPLARE - Llama Scope | 80 |
| $\tau$ SPLARE - Gemma Scope | 50 |
| $\tau$ SPLADE-Llama | 10 |
| Evaluation max context size | 1024 |
| Adam $\beta$s | 0.9, 0.999 |

## Appendix C English-only SPLARE Full Results

We evaluate models from Section [5](#S5) (trained on English data only) on several benchmarks, and provide results in Table [1](#S5.T1.2).
Table [5](#A3.T5) additionally reports evaluation results comparing SPLARE with SPLADE-Llama. We report MRR@10 on MS MARCO [2] and nDCG@10 on TREC DL datasets [14] as well as on all BEIR datasets [86].

**Table 5: Full results (nDCG@10 unless specified) on BEIR, MS MARCO and TREC DL for English-based SPLARE and SPLADE-Llama models. Evaluation done with $\text{Top-K}=(40,400)$.**
| Dataset | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| Arguana | 16.0 | 16.2 |
| Climate-FEVER | 18.3 | 18.0 |
| DBPedia | 44.3 | 44.8 |
| FEVER | 76.0 | 75.8 |
| FiQA-2018 | 42.4 | 42.3 |
| HotpotQA | 66.8 | 67.6 |
| NFCorpus | 37.3 | 36.4 |
| NQ | 61.6 | 61.2 |
| Quora | 87.3 | 87.9 |
| SCIDOCS | 17.5 | 17.3 |
| SciFact | 72.5 | 72.9 |
| TREC-COVID | 84.7 | 82.4 |
| Touché-2020 | 27.2 | 26.9 |
| Average | 50.2 | 50.0 |
| MS MARCO (MRR@10) | 40.8 | 40.0 |
| TREC DL ’19 | 77.4 | 76.3 |
| TREC DL ’20 | 77.3 | 75.9 |

## Appendix D Full Results

Tables [6](#A4.T6.2)—[10](#A4.T10.2) provide the full results of several MTEB datasets: English, Multilingual, and various domains and languages.

Table [13](#A4.T13) compares the SPLARE results on the MIRACL dataset with top multilingual dense retrievers—baseline results are taken from [11]. On this benchmark, SPLARE obtains an average score of 71.7, +0.2 points above M3-embeddings (hybrid: dense + sparse) [10]. Notably, SPLARE is state-of-the-art in English, Finnish, Hindi, Japanese, Russian, Swahili, German and Yoruba, once again indicating its ability to generalize to diverse languages. Note in particular that German and Yoruba are the “secret” languages of MIRACL which were released later *without associated training data*.

**Table 6: Full results of SPLARE and SPLADE-Llama on MTEB(Eng, v2). Evaluation done with $\text{Top-K}=(40,400)$.**
| Task Name | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| ArguAna | 67.4 | 65.9 |
| CQADupstackGamingRetrieval | 59.6 | 59.3 |
| CQADupstackUnixRetrieval | 43.7 | 44.9 |
| ClimateFEVERHardNegatives | 33.4 | 35.9 |
| FEVERHardNegatives | 90.6 | 91.1 |
| FiQA2018 | 57.8 | 57.2 |
| HotpotQAHardNegatives | 76.0 | 73.2 |
| SCIDOCS | 20.8 | 20.3 |
| TRECCOVID | 83.4 | 82.6 |
| Touche2020Retrieval.v3 | 60.8 | 58.4 |
| Average | 59.3 | 58.9 |

**Table 7: Full results of SPLARE and SPLADE-Llama on MTEB(Multilingual, v2). Evaluation done with $\text{Top-K}=(40,400)$.**
| Task Name | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| AILAStatutes | 38.9 | 36.3 |
| ArguAna | 67.4 | 65.9 |
| BelebeleRetrieval | 83.9 | 83.8 |
| CovidRetrieval | 83.3 | 81.5 |
| HagridRetrieval | 98.9 | 98.8 |
| LEMBPasskeyRetrieval | 48.2 | 48.2 |
| LegalBenchCorporateLobbying | 95.1 | 94.9 |
| MIRACLRetrievalHardNegatives | 72.4 | 70.5 |
| MLQARetrieval | 83.8 | 81.5 |
| SCIDOCS | 20.8 | 20.3 |
| SpartQA | 5.3 | 5.2 |
| StackOverflowQA | 88.4 | 90.0 |
| StatcanDialogueDatasetRetrieval | 30.8 | 30.5 |
| TRECCOVID | 83.4 | 82.6 |
| TempReasonL1 | 2.4 | 3.8 |
| TwitterHjerneRetrieval | 73.2 | 73.4 |
| WikipediaRetrievalMultilingual | 91.7 | 90.6 |
| WinoGrande | 53.1 | 52.8 |
| Average | 62.3 | 61.7 |

**Table 8: Full results of SPLARE and SPLADE-Llama on the *multilingual-only* datasets of MTEB(Multilingual, v2). Evaluation done with $\text{Top-K}=(40,400)$.**
| Task Name | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| BelebeleRetrieval | 83.9 | 83.8 |
| MIRACLRetrievalHardNegatives | 72.4 | 70.5 |
| MLQARetrieval | 83.8 | 81.5 |
| StatcanDialogueDatasetRetrieval | 30.8 | 30.5 |
| TwitterHjerneRetrieval | 73.2 | 73.4 |
| WikipediaRetrievalMultilingual | 91.7 | 90.6 |
| Average | 72.6 | 71.7 |

**Table 9: Full results of SPLARE and SPLADE-Llama on MTEB domain specific datasets. Evaluation done with $\text{Top-K}=(40,400)$.**
| Task Name | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| Code |  |  |
| AppsRetrieval | 28.2 | 30.7 |
| COIRCodeSearchNetRetrieval | 62.1 | 70.0 |
| CodeEditSearchRetrieval | 74.2 | 75.0 |
| CodeFeedbackMT | 56.3 | 55.5 |
| CodeFeedbackST | 78.0 | 78.8 |
| CodeSearchNetCCRetrieval | 61.3 | 64.4 |
| CodeSearchNetRetrieval | 85.3 | 86.8 |
| CodeTransOceanContest | 86.7 | 88.5 |
| CodeTransOceanDL | 36.3 | 33.4 |
| CosQA | 31.1 | 30.7 |
| StackOverflowQA | 88.4 | 90.0 |
| SyntheticText2SQL | 68.2 | 67.5 |
| Average | 63.0 | 64.3 |
| Medical |  |  |
| CUREv1 | 61.2 | 57.2 |
| CmedqaRetrieval | 30.2 | 31.5 |
| MedicalQARetrieval | 74.7 | 75.7 |
| NFCorpus | 38.8 | 38.5 |
| PublicHealthQA | 86.7 | 86.2 |
| SciFact | 77.5 | 77.6 |
| SciFact-PL | 73.6 | 74.0 |
| TRECCOVID | 83.4 | 82.6 |
| TRECCOVID-PL | 82.8 | 84.8 |
| Average | 67.7 | 67.6 |
| Law |  |  |
| AILACasedocs | 39.0 | 41.1 |
| AILAStatutes | 38.9 | 36.3 |
| GerDaLIRSmall | 34.7 | 35.8 |
| LeCaRDv2 | 62.3 | 61.3 |
| LegalBenchConsumerContractsQA | 87.4 | 87.9 |
| LegalBenchCorporateLobbying | 95.1 | 94.9 |
| LegalQuAD | 62.2 | 60.8 |
| LegalSummarization | 66.9 | 67.1 |
| Average | 60.8 | 60.7 |
| ChemTEB |  |  |
| ChemHotpotQARetrieval | 86.2 | 85.3 |
| ChemNQRetrieval | 70.0 | 69.6 |
| Average | 78.1 | 77.4 |

**Table 10: Full results of SPLARE and SPLADE-Llama on MTEB language-specific benchmarks. Evaluation done with $\text{Top-K}=(40,400)$.**
| Task Name | SPLARE | SPLADE-Llama |
| --- | --- | --- |
| MTEB(Indic, v1) |  |  |
| BelebeleRetrieval | 87.1 | 87.0 |
| XQuADRetrieval | 97.5 | 96.9 |
| Average | 92.3 | 91.9 |
| MTEB(Scandinavian, v1) |  |  |
| DanFeverRetrieval | 42.8 | 41.7 |
| NorQuadRetrieval | 23.6 | 26.5 |
| SNLRetrieval | 98.0 | 98.0 |
| SweFaqRetrieval | 79.4 | 77.0 |
| SwednRetrieval | 84.1 | 82.5 |
| TV2Nordretrieval | 94.0 | 94.0 |
| TwitterHjerneRetrieval | 73.2 | 73.4 |
| Average | 70.8 | 70.4 |
| MTEB(deu, v1) |  |  |
| GerDaLIR | 18.9 | 19.3 |
| GermanDPR | 87.5 | 86.7 |
| GermanQuAD-Retrieval | 97.2 | 96.8 |
| XMarket | 24.8 | 26.3 |
| Average | 57.1 | 57.3 |
| MTEB(fra, v1) |  |  |
| AlloprofRetrieval | 56.4 | 58.4 |
| BSARDRetrieval | 64.4 | 59.9 |
| MintakaRetrieval | 46.2 | 57.7 |
| SyntecRetrieval | 90.7 | 87.4 |
| XPQARetrieval | 66.4 | 64.6 |
| Average | 64.8 | 65.6 |
| MTEB(kor, v1) |  |  |
| Ko-StrategyQA | 82.7 | 83.1 |
| MIRACLRetrieval | 69.4 | 66.5 |
| Average | 76.0 | 74.8 |

**Table 11: Multi-lingual retrieval performance on MIRACL dev (nDCG@10). Baseline results are taken from [11, 51]. ^† denotes the two hidden test sets of MIRACL. Evaluation for SPLARE and SPLADE-Llama done with $\text{Top-K}=(40,400)$.**
| Model | ar | bn | en | es | fa | fi | fr | hi | id | ja | ko | ru | sw | te | th | zh | de^† | yo^† | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baselines (Prior Work) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| BM25 | 39.5 | 48.2 | 26.7 | 7.7 | 28.7 | 45.8 | 11.5 | 35.0 | 29.7 | 31.2 | 37.1 | 25.6 | 35.1 | 38.3 | 49.1 | 17.5 | 12.0 | 56.1 | 31.9 |
| mDPR | 49.9 | 44.3 | 39.4 | 47.8 | 48.0 | 47.2 | 43.5 | 38.3 | 27.2 | 43.9 | 41.9 | 40.7 | 29.9 | 35.6 | 35.8 | 51.2 | 49.0 | 39.6 | 41.8 |
| mContriever | 52.5 | 50.1 | 36.4 | 41.8 | 21.5 | 60.2 | 31.4 | 28.6 | 39.2 | 42.4 | 48.3 | 39.1 | 56.0 | 52.8 | 51.7 | 41.0 | 40.8 | 41.5 | 43.1 |
| mE5large | 76.0 | 75.9 | 52.9 | 52.9 | 59.0 | 77.8 | 54.5 | 62.0 | 52.9 | 70.6 | 66.5 | 67.4 | 74.9 | 84.6 | 80.2 | 56.0 | 56.4 | 78.3 | 66.6 |
| E5${}_{\mathrm{mistral\text{-}7b}}$ | 73.3 | 70.3 | 57.3 | 52.2 | 52.1 | 74.7 | 55.2 | 52.1 | 52.7 | 66.8 | 61.8 | 67.7 | 68.4 | 73.9 | 74.0 | 54.0 | 54.1 | 79.7 | 63.4 |
| Gemini Embedding | 78.3 | 79.0 | 58.7 | 57.0 | 60.9 | 78.0 | 55.6 | 65.4 | 54.3 | 75.1 | 68.9 | 73.4 | 81.0 | 80.5 | 80.8 | 65.7 | 59.8 | 88.8 | 70.1 |
| M3-Emb (Sparse) | 67.1 | 68.9 | 43.8 | 38.6 | 45.1 | 65.4 | 35.3 | 48.2 | 48.9 | 56.1 | 61.5 | 44.5 | 57.9 | 79.1 | 70.9 | 36.1 | 32.5 | 70.0 | 53.9 |
| M3-Emb (All) | 80.2 | 81.5 | 59.6 | 59.7 | 63.4 | 80.4 | 61.2 | 63.3 | 59.0 | 75.2 | 72.1 | 71.7 | 79.6 | 88.1 | 83.7 | 64.9 | 59.8 | 83.5 | 71.5 |
| SPLADE-Llama | 78.0 | 77.5 | 58.8 | 56.0 | 59.8 | 79.9 | 58.5 | 61.7 | 57.1 | 75.3 | 66.1 | 72.8 | 80.1 | 82.4 | 81.0 | 61.8 | 58.2 | 92.5 | 69.9 |
| SPLARE-7B | 79.7 | 79.9 | 60.9 | 58.5 | 62.1 | 81.4 | 60.5 | 65.5 | 57.6 | 75.9 | 69.5 | 74.1 | 81.8 | 83.2 | 83.1 | 64.1 | 62.5 | 90.0 | 71.7 |
| SPLARE-2B | 75.4 | 67.4 | 53.3 | 55.0 | 54.8 | 75.5 | 56.0 | 59.1 | 54.9 | 67.9 | 67.4 | 65.3 | 69.7 | 77.8 | 76.7 | 59.0 | 56.1 | 78.0 | 65.0 |

**Table 12: XTREME-UP dataset (MRR@10) - Average Scores. Baselines taken from [51]. Evaluation for SPLARE done with $\text{Top-K}=(40,400)$.**
| Model | MRR@10 |
| --- | --- |
| SPLARE | 58.6 |
| SPLARE (Eng Only) | 41.6 |
| SPLADE-Llama | 56.2 |
| SPLADE-Llama (Eng Only) | 30.5 |
| Gemini Embedding | 64.3 |
| Gemini Embedding (Eng Only) | 49.3 |
| Gecko i18n Embedding | 35.0 |
| voyage-3-large | 39.2 |
| Linq-Embed-Mistral | 24.6 |
| multilingual-e5-large-instruct | 18.7 |
| gte-Qwen2-7B-instruct | 17.4 |
| text-embedding-3-large | 18.8 |

**Table 13: XTREME-UP dataset (MRR@10) - Full scores. Evaluation done with $\text{Top-K}=(40,400)$.**
| Model | Avg. | as | bho | brx | gbm | gom | gu | hi | hne | kn | mai | ml | mni | mr | mwr | or | pa | ps | sa | ta | ur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Eng Only |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SPLARE | 42.6 | 44.4 | 49.4 | 10.2 | 48.2 | 39.2 | 40.9 | 57.0 | 49.5 | 46.5 | 52.6 | 47.6 | 20.1 | 50.8 | 49.8 | 26.3 | 44.4 | 37.3 | 43.5 | 45.6 | 48.6 |
| SPLADE-Llama | 30.5 | 31.7 | 46.8 | 10.7 | 45.8 | 30.7 | 24.2 | 54.7 | 44.5 | 19.0 | 48.7 | 21.8 | 20.0 | 45.2 | 48.2 | 7.6 | 23.3 | 2.2 | 38.5 | 27.8 | 19.5 |
| Multilingual |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SPLARE | 58.6 | 63.3 | 63.1 | 14.2 | 60.8 | 58.8 | 65.1 | 67.2 | 63.2 | 64.7 | 65.0 | 68.9 | 27.4 | 64.5 | 63.3 | 54.5 | 66.1 | 52.7 | 63.8 | 63.3 | 61.2 |
| SPLADE-Llama | 56.2 | 61.2 | 58.8 | 14.2 | 59.4 | 56.5 | 62.4 | 64.1 | 62.5 | 62.2 | 64.5 | 63.2 | 29.7 | 60.5 | 59.6 | 53.2 | 62.2 | 50.7 | 62.7 | 59.0 | 57.6 |

## Appendix E Latency Measures

We provide per-query retrieval latency as measured on MS MARCO (retrieval from a collection of $8.8M$ documents) for SPLARE and SPLADE-Llama in Figure [4](#A5.F4). To measure this, we first index the collection using Seismic [8], and then perform single-threaded retrieval on the saved index. Building a highly optimized sparse retrieval setup is difficult in general; here we use the Seismic library with default hyperparameters—given in Table [14](#A5.T14).

In this simple setup, retrieval takes around 5*ms* per query with maximal accuracy for SPLARE. In low-latency regime ($<$ 4*ms*), SPLARE can be used with higher accuracy compared to SPLADE.

Figure: Figure 4: Retrieval Latency (*ms*) when pooling documents (Left) or query (Right) representations with Top-K. In low-latency settings, SPLARE enables higher accuracy for a given level of latency.
Refer to caption: https://arxiv.org/html/2603.13277/2603.13277v1/x5.png

**Table 14: Seismic retrieval parameters used to measure latency.**
| Parameter | Value |
| --- | --- |
| k | 1000 |
| query_cut | 30 |
| heap_factor | 0.5 |
| n_knn | 0 |
| sorted | False |
| num_threads | 1 |

## Appendix F SPLADE Layer Ablation

In § [5](#S5), we showed that SPLARE models are typically more effective at intermediate layers, yielding a latency advantage over SPLADE. In principle, however, SPLADE models can also be trained using intermediate representations by simply applying the LM head to these layers. Table [15](#A6.T15) reports results obtained with this training procedure.

**Table 15: Training SPLADE-Llama models at intermediate layers leads to strong deterioration. At layer $<$ 22, models collapse during training.**
| SPLADE-Llama at intermediate layers |  |  |  |  |
| --- | --- | --- | --- | --- |
| Layer No. | 18 | 22 | 26 | 31 |
| MTEB(Eng, v2) | 0. | 43.6 | 44.5 | 52.9 |

## Appendix G Retrieval Examples

We provide in Figures [G](#A7)—[G](#A7) multiple examples of scores and explanations obtained for positive documents for some queries on English, Multilingual and multi-domain datasets. We also provide examples on the code domain (Figures [G](#A7)—[G](#A7)), which highlight some of the limitations of SPLARE on specific domains which might require dedicated SAEs. Notably, in Figure [G](#A7) which shows a Tamil example, *the document and query representations coincide for only 6 tokens*, further highlighting SPLADE multilingual limitations.
Note that the explanations we used, taken from Neuronpedia, are mostly annotated by LLMs provided with examples of context with features activations. As such, these explanations may remain inaccurate or incomplete.

[Uncaptioned image]: https://arxiv.org/html/2603.13277/2603.13277v1/interpretability_examples/tamil_img_2.pngs