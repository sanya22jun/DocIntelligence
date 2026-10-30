# Project Report Outline
## AI-Powered Cross-Document Semantic Contradiction Detection System
### CrossDoc Intelligence

A chapter-by-chapter skeleton for the written submission. Each section
lists what to write and which file or command produces the evidence, so
every claim in the report can be traced back to something in the repo.

---

## Front matter

- Title page, certificate, declaration, acknowledgements (institute format)
- Abstract (150–200 words): the consistency problem across document sets,
  the embedding-plus-reasoning approach, what was built, headline
  evaluation result, and the stated limitations
- Table of contents, list of figures, list of tables

---

## 1. Introduction

1.1 **Background** — organisations keep overlapping policy documents that
drift apart across revisions and departments.

1.2 **Problem statement** — a 2023 leave policy grants 20 days while the
2024 handbook grants 15; one permits carry-over, the other forfeits it.
Keyword search cannot surface these because the wording is nearly
identical and only the *meaning* conflicts.

1.3 **Why this is hard** — contradiction is not lexical difference.
Statements can disagree in wording but agree in meaning ("20 days" vs
"four weeks"), or agree in wording but differ in scope ("managers get 25
days" vs "interns get 15 days"). Both must be handled.

1.4 **Objectives**
- Extract comparable statements from PDF/TXT documents with provenance
- Find semantically related statements across different documents
- Classify each related pair as contradiction, support or neutral
- Explain every conflict with citable sources
- Keep API cost controlled and measurable

1.5 **Scope and limitations** — prototype, English, pairwise,
cross-document only, no OCR.

---

## 2. Literature / technology review

2.1 Lexical retrieval and why it fails here (TF-IDF, BM25 — no negation
or scope awareness)

2.2 Sentence embeddings and semantic similarity; the distinction between
*relatedness* and *agreement* — the core reason similarity is used only
for candidate selection in this system

2.3 Natural Language Inference: entailment / contradiction / neutral as
the classical three-way task this project mirrors

2.4 Large language models as zero-shot NLI classifiers; structured
(schema-constrained) output

2.5 Comparison table: rule-based vs embedding-only vs LLM-reasoning
approaches, with the failure mode of each

---

## 3. System analysis and design

3.1 **Requirements** — functional and non-functional, including the cost
ceiling

3.2 **Architecture** — the pipeline diagram from `README.md`:
upload → extract → clean → segment → embed → shortlist → classify →
report

3.3 **Module design** — one subsection per service, from
`app/services/`:

| Module | Responsibility |
|---|---|
| `document_processor.py` | validation, PDF/TXT extraction in memory |
| `text_preprocessor.py` | conservative cleaning, sentence segmentation |
| `ingestion.py` | orchestration of the extraction stage |
| `openai_service.py` | the only API client; errors, retries, usage |
| `embedding_service.py` | batching, de-duplication, caching |
| `similarity_service.py` | cosine candidate selection |
| `contradiction_service.py` | prompt, schema, validation, recovery |
| `analysis_pipeline.py` | stage orchestration and report assembly |
| `chat_service.py` | grounded Q&A: retrieval, citation validation |
| `export_service.py` | JSON and CSV export |

3.4 **Data design** — the statement record (document_id,
document_name, page_number, sentence_id, text) and the result record;
why `page_number` is null for TXT

3.5 **Interface design** — dashboard sections, with screenshots

3.6 **Design decisions and trade-offs** — the table in
`README.md § Design decisions`: in-memory processing, failures reported
rather than dropped, vectors kept out of the stored payload, no
fabricated evidence, model-reported (not calibrated) confidence

---

## 4. Implementation

4.1 **Environment** — Python 3.11, Flask 3.1, PyMuPDF, spaCy
`en_core_web_sm`, NumPy, OpenAI SDK, Bootstrap 5

4.2 **Text extraction** — PyMuPDF page-by-page; encoding fallback chain
for TXT; how extraction failures are surfaced

4.3 **Preprocessing** — what is removed (line-break hyphenation, running
page markers, dot leaders) and, importantly, **what is deliberately
preserved**: numbers, dates, negations, casing, punctuation. Include the
worked example of a heading being separated from the sentence below it.

4.4 **Segmentation** — spaCy per paragraph; the statement filter
(minimum length/words, alphanumeric density, all-caps headings) and why
it protects the API budget

4.5 **Embeddings** — `text-embedding-3-small`, batching, the
de-duplication and caching strategy

4.6 **Candidate selection** — L2 normalisation so cosine is a single
matrix multiply; cross-document masking; top-K plus threshold plus cap.
State explicitly that similarity is never used as evidence.

4.7 **Contradiction classification** — the prompt design, including the
negative instructions that suppress false positives; the strict JSON
schema; the six conflict categories

4.8 **Trustworthiness mechanisms** — this is a strong section for a viva:
- evidence and source references come from stored statements, never the
  model
- highlight spans accepted only when verbatim in the statement
- results for unrequested pairs discarded
- omitted pairs retried, then reported as unclassified, never guessed

4.9 **Cost and reliability engineering** — caps, batching, request
budget, error taxonomy, partial-result preservation

4.10 **Grounded document chat** — retrieval over the same embeddings,
context assembly including relationships already found, strict-schema
answering, and citation validation against the retrieved set. Worth its
own subsection: it is retrieval-augmented generation built on the
artefacts the analysis already produced, and it demonstrates the same
no-fabrication discipline in a second setting.

4.11 **Frontend** — real server-driven stage progress (not a simulated
bar), filters, export, chat panel with cited sources

---

## 5. Testing

5.1 **Strategy** — unit, integration and HTTP-level tests; a stub
OpenAI client so the whole engine is testable offline at zero cost;
`TestConfig` forces an empty API key so the suite cannot touch the real
API

5.2 **Coverage summary** — the module/count table (see
`python -m pytest -q`)

5.3 **Adversarial and failure testing** — malformed JSON, truncated
output, omitted pairs, invented pairs, every SDK error class, oversized
and wrong-type uploads, path traversal, non-string ids

5.4 **Frontend security testing** — `tests/js/text-utils.test.js`:
escaping and highlight safety against hostile document text and hostile
model output

5.5 **Bugs found and fixed during testing** — worth a short table; it
demonstrates real engineering:
- headings gluing onto the following sentence (segmentation)
- all-caps titles passing as statements
- a non-string `analysis_id` causing a server error instead of a 400
- the test suite silently reading the developer's real API key

---

## 6. Evaluation

6.1 **Methodology** — a hand-labelled set of 42 statement pairs
(`evaluation/dataset.py`): 16 contradictions across all five categories,
12 support pairs including unit-equivalent phrasings, 14 neutral pairs of
which 8 are **hard negatives** designed so that comparing numbers alone
gives the wrong answer. Every label carries a written justification.

6.2 **Metrics** — precision, recall, F1 per class; macro and weighted
averages; accuracy reported two ways (over classified pairs, and with
unclassified pairs charged as errors); confusion matrix; one-vs-rest
contradiction detection. Implemented in `evaluation/metrics.py` and unit
tested against hand-computed values.

6.3 **Baseline** — the stub classifier is effectively a
"numbers differ ⇒ contradiction" rule. Report its score as the floor the
model must beat; it also demonstrates the dataset discriminates.

6.4 **Results** — the confusion matrix and metric table from the live
run, plus the hard-negative slice and the indicative category agreement.
Command: `python -m evaluation.run_evaluation --mode live`

6.5 **Retrieval evaluation** — separately, the recall of the similarity
gate: of the 11 relationships deliberately planted in the sample
documents, how many were shortlisted for the model. Classification
quality is irrelevant for a pair that never reaches the classifier.
Command: `--mode live --retrieval`

6.6 **Error analysis** — walk through the actual misclassifications with
the statement text and the model's explanation; say which failure mode
each represents (scope confusion, unit equivalence, category boundary)

6.7 **Cost measurement** — measured tokens and the estimated cost from
the run's `usage` block; compare against the projection; state clearly
that the cost figure is an estimate from list prices, not a bill

6.8 **Threats to validity** — small hand-built set; the author wrote both
the documents and the labels; a single run of a non-deterministic model;
figures are indicative, not a benchmark

---

## 7. Results and discussion

7.1 Screenshots: upload workspace, stage progress, overview tiles,
conflict cards with highlighting, filters, exported CSV

7.2 A worked demonstration: the three sample documents in, the planted
conflicts out, with the model's explanations quoted

7.3 A chat transcript showing a grounded answer with citations, a
follow-up question, and a question the documents cannot answer (the
honest refusal is the point worth demonstrating)

7.4 What worked well, and what the system gets wrong

7.5 Where similarity-based selection is the bottleneck versus where
classification is

---

## 8. Conclusion and future work

8.1 Summary against the objectives in §1.4

8.2 Future work, each with a reason:
- OCR for scanned PDFs
- self-consistency checking within a single document
- multi-statement (not just pairwise) conflict detection
- a persistent store and a real task queue
- clustering to group related conflicts into themes
- chat retrieval over passages rather than single statements, for
  questions whose answer spans several statements
- human-in-the-loop confirmation to build a calibration set, which is
  what would turn model confidence into a calibrated probability

---

## Appendices

- A: Installation and run instructions (`README.md`)
- B: API reference with status codes
- C: The labelled evaluation dataset
- D: Full evaluation output (`evaluation/results/*.txt`)
- E: Sample documents and the planted relationship key
- F: Test output (`python -m pytest -q`, `node tests/js/text-utils.test.js`)
- G: Source listing

---

## Figures and tables to prepare

| # | Item | Source |
|---|---|---|
| Fig 1 | Pipeline architecture | `README.md § Architecture` |
| Fig 2 | Candidate-selection funnel | report `stages` block |
| Fig 3 | Dashboard screenshot | running app |
| Fig 4 | Conflict card close-up | running app |
| Fig 5 | Confusion matrix | `evaluation/results/*.json` |
| Tab 1 | Module responsibilities | §3.3 |
| Tab 2 | Conflict categories | `contradiction_service.py` |
| Tab 3 | Cost controls | `README.md § API cost controls` |
| Tab 4 | Per-class metrics | evaluation output |
| Tab 5 | Planted relationships | `evaluation/planted.py` |
| Tab 6 | Bugs found in testing | §5.5 |

## Honesty checklist before submission

- [ ] Every reported metric comes from a run whose output file is in
      `evaluation/results/`
- [ ] Stub-mode figures are never presented as model performance
- [ ] Confidence is described as model-reported, not calibrated
- [ ] Cosine similarity is described as candidate selection, not evidence
- [ ] Cost is described as an estimate from list prices
- [ ] Limitations section is not quietly trimmed
