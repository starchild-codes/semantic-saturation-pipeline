# Semantic Saturation in Online Discussions — Frozen Protocol

## Research question
How quickly do online discussion threads reach semantic saturation—that is, what fraction of a thread must occur before most of its eventual semantic content is already represented?

## Dataset
Cornell ConvoKit `reddit-corpus-small`.

## Unit of analysis
A Reddit conversation thread.

The root submission/post is excluded. Only actual comments are analyzed.

## Cleaning
Remove only:
- empty comments
- `[deleted]`
- `[removed]`

No minimum word-count filter is used.

## Eligibility
A thread must contain at least **20 usable comments** after cleaning.

## Representation
Exactly one pretrained sentence embedding model:

`sentence-transformers/all-mpnet-base-v2`

No alternative embedding models are used for the paper.

## Semantic coverage

For a thread containing embeddings \(e_1,\dots,e_n\), define semantic coverage after the first \(k\) comments as:

\[
C(k)=\frac{1}{n}\sum_{j=1}^{n}\max_{i\leq k}\cos(e_i,e_j)
\]

The final value is normalized so that \(C(n)=1\).

## Saturation metrics

\[
S_{80}=\min\left\{\frac{k}{n}: C(k)\geq0.80\right\}
\]

\[
S_{90}=\min\left\{\frac{k}{n}: C(k)\geq0.90\right\}
\]

## Hypotheses

### H1
Semantic coverage increases sublinearly with thread progression, such that online discussions reach 80% semantic coverage before 80% of their comments have occurred.

### H2
Longer discussions exhibit greater proportional semantic redundancy, reflected in a lower \(S_{80}\).

## Frozen analyses
Exactly four outputs are part of the paper:

1. Distribution of \(S_{80}\)
2. Distribution of \(S_{90}\)
3. Aggregate semantic-coverage curve across normalized thread position
4. Spearman correlation between thread length and \(S_{80}\)

## Interpretation
The metric is called **semantic coverage**.

It is NOT described as:
- Shannon information
- factual information
- knowledge coverage
- percentage of all ideas

## Explicit DO NOT ADD list
Do not add:
- another embedding model
- human annotation
- LLM judging
- topic modeling
- clustering comparisons
- sentiment
- toxicity
- political/non-political splits
- subreddit-by-subreddit hypothesis testing
- scores/upvotes
- author characteristics
- reply-tree structure
- temporal speed
- prediction models
- summarization experiments
- compression algorithms
- semantic half-life
- TF-IDF comparisons
- post-hoc ablations

Unexpected findings are reported as findings or future work. They do not trigger new experiments for this paper.
