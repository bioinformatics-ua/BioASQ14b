# Prompt

The paper you wrote is well structured and the English is very good. However, it contains several incorrect statements and some missing information. Below, I will point out the main issues and suggest corrections, for the sections. Don't forget to check these errors in other sections, as fixing them may create inconsistencies in the paper.

## 3. Methodology

### 3.1 Retrieval

#### 3.1.1 BM25 Retrieval

- `pg_textsearch` is an external PostgreSQL extension for full-text search, not a built-in feature
- Add a reference for pg_textsearch, https://github.com/timescale/pg_textsearch
- The BM25 index doesn't have a name
- Don't focus too much on performance improvements, but on the fact that now it is much more usable and maintainable

#### 3.1.4 Query expansion: HyDE and Context-1

You must rewrite this entire section. Actually, make it in two of them, one for HyDE and another for Context-1.

For Context-1, search on the web how it works and what are its main goals (for example, https://www.trychroma.com/research/context-1). It does not generate any refined representation, it uses LLMs to improve the normal RAG pipeline, to make it more effective. It is not a query expansion technique. Search first before writing, and make sure you understand how it works. And make this corresponding text a bit bigger.

### 3.1.5 Reranker training pipeline

Put a table in the appendix with the different models.

### 3.1.6 Fusion strategies

I told you to focus more on RRF than weighted sum, but you completely ignored weighted sum. You should explain what RRF is and how it works, the same for weighted sum, and why RRF is better than weighted sum.

### 3.1.7 Snippets generation & model fine-tuning

The approach you wrote is not the one we implemented. I haven't told you, but we fine-tuned a 31B Gemma 4 model using LoRA for snippet generation. The input is the question and the document title and abstract, and the output is the snippets from that document. We trained it using the snippets provided by BioASQ training data. Check details at @src/bioasq/snippets/

### 3.1.8 Future work: SPLARe, ColBERT

I was dumb, sorry. I mean SPLADE, not SPLARe

## 3.2 Generation

### 3.2.1 LLM as a judge

What you wrote is good, but I want you to expand it a bit more, and also explain the motivation behind using an LLM as a judge, and how it compares to majority voting. Also, explain how the LLM is used to evaluate the outputs of different models, and how it combines them in a more intelligent way. Check @src/bioasq/phase_b/evaluation/llm_judge.py for details.

In the discussion section, write about this as well, but in the sense on why we didn't use it as much as the agents quorum approach. Reasons include:

- Less flexibility
- Less intuitive to generate results
- Lack of infrastructure ready for it
- Overall more expensive

### 3.2.2 Agent Quorum

It is perfect, but I want you to expand it. Explain that different agents have different personalities and ways of thinking (e.g. skeptical, evidence-based, etc.). Show an example of a system prompt or so. Check @src/bioasq/phase_b/quorum/ for details.

Also, for different batches, we have different versions of the agent quorum (different models), check @docs/2026/submissions/ for details and make table(s) for showing them.

### 3.2.4 Prompting and Model selection

Remove this section, it is useless.

## 4. Results

Add a subsection for our internal retrieval testing, such as the usage of Context-1, RRF, etc. Don't add anything, just make it a small introduction and a TODO, since we are yet to gather the results.

## 5. Discussion

### 5.1 Impact of Architectural Changes

Again, the TEI shit won't make it much faster, it is fast but hold on, so don't emphasize it that much. Talk about being an API-based workflow, which allows to span multiple TEI instances.

### 5.3 Agent Quorum

As stated before, check the used agents for each batch at @docs/2026/submissions/ , this allows you to do a much better reasoning.

State that small models (e.g., Gemma 4 E2B and E4B) perform better at all metrics except lists. Explain why (being smaller, they tend to not create an overly complex representation of the input and thus do not overfit or overthink about the input and the context)

### 5.5 Future Work

- Again, sorry. I meant SPLADE, not SPLARe
- The agents already have roles, so don't say that.
