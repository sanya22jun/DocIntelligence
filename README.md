# CrossDoc Intelligence

AI-powered semantic document consistency analysis — upload several documents
and find statements that contradict each other across them.

> **Status: complete and evaluated against the live API.** On a
> hand-labelled set of 42 statement pairs the system scores **95.2%
> accuracy (macro F1 0.952)**, with **100% precision on contradiction
> detection** and **7/8 hard negatives correct**. The similarity gate
> retrieved **11 of 11** planted relationships while discarding 97% of
> possible pairs. Full figures, method and error analysis in
> [Evaluation](#evaluation).

---

## Problem

Policies, handbooks and contracts drift apart. A 2023 leave policy promises
20 days of annual leave while the 2024 handbook promises 15; one document says
unused leave carries over, another says it is forfeited. Keyword search cannot
find these conflicts because the wording is similar and the *meaning* is
opposite. This project finds them by comparing statements semantically.

## Scope

| Capability | State |
|---|---|
| Multi-document upload (PDF + TXT) | Done |
| Extension / size / count validation | Done |
| PDF extraction with page numbers (PyMuPDF) | Done |
| TXT extraction with encoding fallback | Done |
| Conservative text cleaning | Done |
| Sentence segmentation (spaCy, regex fallback) | Done |
| Source document + page preserved per statement | Done |
| Per-document error reporting | Done |
| OpenAI embeddings, batched + cached | Done |
| Cosine candidate selection (cross-document, capped) | Done |
| GPT contradiction classification, strict JSON | Done |
| Structured explanations with verified source refs | Done |
| Token-usage + cost accounting | Done |
| Interactive dashboard, real stage progress | Done |
| Grounded document chat with cited sources | Done |
| Side-by-side conflict cards, verified highlighting | Done |
| Relationship / category / confidence filters | Done |
| JSON + CSV export | Done |
| Labelled evaluation set (42 pairs) | Done |
| Metrics: precision / recall / F1 / confusion matrix | Done |
| Retrieval (similarity-gate) evaluation | Done |
| Security and reliability audit | Done |
| Project report outline | Done |
| Live evaluation run against the real API | Done |

## Architecture

```
Browser (Bootstrap + vanilla JS)
        |  POST /api/upload  (multipart)
        v
Flask routes  (app/routes.py)
        |
        v
ingestion.ingest_documents          <- orchestration
        |
        +-- document_processor      <- validate + extract (in memory)
        |        PDF -> PyMuPDF, one PageText per page
        |        TXT -> decoded text, page_number = None
        |
        +-- text_preprocessor       <- clean + segment
                 clean_text()       -> de-hyphenate, drop page markers,
                                       join wrapped lines, collapse spaces
                 split_sentences()  -> spaCy per paragraph
        |
        v
store.save(payload) -> analysis_id  <- in-memory, bounded, thread-safe
        |
        |  POST /api/analyze {analysis_id}
        v
analysis_pipeline.run_analysis
        |
        +-- embedding_service       <- batch + dedupe + cache
        |        text-embedding-3-small
        |
        +-- similarity_service      <- cosine, cross-document only,
        |        threshold 0.65, top-5 per statement, cap 100 pairs
        |        (candidate SELECTION only, never evidence)
        |
        +-- contradiction_service   <- gpt-4.1-mini, strict JSON schema
        |        CONTRADICTION / SUPPORT / NEUTRAL + category
        |        + validation, span verification, retry/split
        |
        v
conflict report (+ usage, stages, warnings) -> stored on the analysis
        |
        |  GET /api/results/<id>    <- polled: live stage, then the report
        |  GET /api/export/<id>     <- JSON or CSV download
        v
dashboard: overview tiles, conflict cards, filters, export, chat
```

The same statements also back a grounded chat panel:

```
question --> embed --> cosine over the statements --> top 8
                                                       |
                        + relationships already found among them
                                                       |
                                                       v
                                      gpt-4.1-mini, strict schema
                                                       |
                        citations validated against what was retrieved
                                                       v
                                 answer + document / page / statement id
```

The analysis runs on a background thread and the browser polls
`/api/results/<id>`, so the progress display shows the stage the server is
**actually** in rather than an indeterminate spinner.

### Statement record

Every extracted statement carries its own provenance, so a later conflict
report can cite a real source instead of inventing one:

```json
{
  "document_id": "doc_001",
  "document_name": "hr_leave_policy_2023.txt",
  "page_number": 2,
  "sentence_id": "sent_004",
  "text": "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
  "word_count": 14,
  "char_count": 83
}
```

`sentence_id` is unique across the whole analysis run, so a pair can be named
unambiguously as `sent_004|sent_014`. `page_number` is `null` for TXT files, which have no
pagination — no page reference is fabricated.

## NLP techniques

- **Conservative normalisation.** Unicode NFKC, soft-hyphen and zero-width
  removal, de-hyphenation of words split across line breaks
  (`annu-\nal` → `annual`), removal of running page markers (`Page 2 of 9`)
  and dot-leader rules, collapsing of wrapped lines inside a paragraph.
  Numbers, dates, negations, casing and punctuation are deliberately left
  untouched — they carry the meaning a contradiction check depends on. There
  is no lower-casing, stemming or stop-word removal.
- **Sentence segmentation** with spaCy `en_core_web_sm` (dependency-parser
  boundaries; NER and the lemmatizer are excluded for speed), run
  **per paragraph** so that an unpunctuated heading is not glued onto the
  sentence beneath it. A regex splitter with abbreviation guards is used if
  the model is unavailable.
- **Statement filtering.** Fragments shorter than 25 characters or 4 words,
  non-alphabetic debris, low alphanumeric density (table rows, dot leaders)
  and all-caps headings are dropped, so the API budget is spent only on real
  claims.
- **Semantic embeddings.** `text-embedding-3-small` turns each statement into
  a 1536-dimension vector. Requests are batched, identical text is embedded
  once per run, and a process-wide cache means re-analysing the same corpus
  costs nothing on the embedding side.
- **Cosine similarity for candidate selection.** Vectors are L2-normalised so
  cosine reduces to a dot product, giving one matrix multiply for the whole
  corpus. Only pairs from *different* documents are considered; each statement
  nominates its top 5 neighbours above a 0.65 threshold; the merged list is
  capped at 100 pairs. **Similarity only decides what is worth asking about —
  it is never treated as evidence of a contradiction.** Two statements can be
  near-identical in wording and perfectly compatible.
- **GPT relationship classification.** `gpt-4.1-mini` at temperature 0 labels
  each shortlisted pair `CONTRADICTION`, `SUPPORT` or `NEUTRAL` with a
  conflict category, a short explanation and a self-reported confidence,
  constrained by a strict JSON schema. The prompt explicitly rules out the
  common false positives: differing numbers under *different scopes*, a
  general rule versus its stated exception, different entities or
  departments, and one statement merely adding detail the other omits.

## Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Web | Flask 3.1, Jinja2 |
| PDF | PyMuPDF 1.26 |
| NLP | spaCy 3.7 + `en_core_web_sm` |
| Config | python-dotenv |
| Frontend | Bootstrap 5.3 (CDN), vanilla JS |
| Embeddings | OpenAI `text-embedding-3-small` |
| Reasoning | OpenAI `gpt-4.1-mini` (strict JSON schema) |
| Vector maths | NumPy |
| Tests | pytest |

## Installation

```powershell
cd E:\NLP
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A virtual environment is optional if the packages are already installed
globally.

## Environment configuration

```powershell
copy .env.example .env
```

Then edit `.env` and set `OPENAI_API_KEY`. `.env` is git-ignored, the key is
read from the environment only, it is never written to a template or a log,
and it never reaches the frontend — `GET /api/health` reports only *whether*
a key is configured. Upload, extraction and segmentation work without a key;
only `POST /api/analyze` needs one.

| Variable | Default | Meaning |
|---|---|---|
| `SECRET_KEY` | dev value | Flask secret |
| `FLASK_DEBUG` | `false` | Debug reloader |
| `MAX_FILE_SIZE_MB` | `10` | Per-file size limit |
| `MIN_FILES_PER_ANALYSIS` | `2` | Minimum documents to compare |
| `MAX_FILES_PER_ANALYSIS` | `10` | Maximum documents per run |
| `SPACY_MODEL` | `en_core_web_sm` | Segmentation model |
| `MIN_SENTENCE_CHARS` | `25` | Shortest kept statement |
| `MIN_SENTENCE_WORDS` | `4` | Fewest words kept |
| `MAX_STORED_ANALYSES` | `20` | In-memory result cap |
| `OPENAI_API_KEY` | *(none)* | Required for analysis |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OPENAI_REASONING_MODEL` | `gpt-4.1-mini` | Classification model |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `OPENAI_MAX_RETRIES` | `2` | SDK-level retries |
| `SIMILARITY_THRESHOLD` | `0.65` | Candidate gate |
| `TOP_K_CANDIDATES` | `5` | Neighbours per statement |
| `MAX_CANDIDATE_PAIRS` | `100` | Hard pair cap per analysis |
| `EMBEDDING_BATCH_SIZE` | `64` | Statements per embedding request |
| `MAX_SENTENCES_PER_ANALYSIS` | `400` | Pre-embedding guard |
| `REASONING_BATCH_SIZE` | `8` | Pairs per GPT request |
| `MAX_OUTPUT_TOKENS` | `1400` | Output cap per GPT request |
| `CHAT_TOP_K` | `8` | Statements retrieved as context per question |
| `CHAT_MAX_OUTPUT_TOKENS` | `700` | Output cap for a chat answer |
| `CHAT_SUGGESTION_TOKENS` | `200` | Output cap for suggested questions |
| `PRICE_*_PER_MTOK` | list prices | Used for cost *estimates* only |

## Running

```powershell
cd E:\NLP
python run.py
```

Open <http://127.0.0.1:5000>, drag in two or more documents from
`sample_documents\`, and click **Analyze documents**. The dashboard uploads
them, starts the analysis, shows each stage as the server reaches it, then
renders the conflict cards. `OPENAI_API_KEY` must be set for the analysis
step; without it the dashboard reports that clearly instead of failing
silently.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard |
| `POST` | `/api/upload` | Validate, extract and segment uploaded documents |
| `POST` | `/api/analyze` | Start the semantic analysis for an `analysis_id` |
| `GET` | `/api/results/<id>` | Live progress, then the finished report |
| `POST` | `/api/chat` | Answer a question from the documents, with citations |
| `GET` | `/api/chat/suggestions/<id>` | Starter questions for this analysis |
| `GET` | `/api/export/<id>` | Download the report (`?format=json\|csv`) |
| `GET` | `/api/analysis/<id>` | Retrieve a stored analysis in full |
| `GET` | `/api/health` | Liveness, segmenter, models, key presence |

Status codes: `400` the request itself is unusable (too few/many files, no
`analysis_id`, bad export format), `404` unknown `analysis_id`, `413` the
upload exceeds the request limit, `422` every document failed or there is
nothing comparable to analyse, `503` no API key is configured, `202` the
analysis started, `200` success (per-document failures are reported in
`documents[]` and `errors[]` rather than discarded).

`POST /api/analyze` returns `202` and runs in the background; poll
`/api/results/<id>` for `state` (`idle` / `running` / `done` / `failed`),
the current `stage`, and the report once it is done. An OpenAI failure during
the run surfaces there as `state: "failed"` with a readable reason.

The endpoint reuses an existing report for the same `analysis_id` instead of
spending on a repeat run, and refuses to start a second run while one is
already in flight; pass `{"force": true}` to deliberately re-analyse.

### Example

```powershell
curl -X POST http://127.0.0.1:5000/api/upload `
  -F "documents=@sample_documents/hr_leave_policy_2023.txt" `
  -F "documents=@sample_documents/hr_handbook_2024.txt"
```

## Dashboard

- **Overview tiles** — documents analysed, statements extracted, pairs
  analysed, contradictions, supporting and neutral relationships.
- **Conflict cards** — one per classified pair, side by side: each statement
  with its document name, page reference and statement id, the conflict
  category, the model's explanation, its self-reported confidence and the
  similarity score that shortlisted the pair. Contradictions are marked with
  a red edge and listed first.
- **Highlighting** — the clashing phrase is highlighted only when the model
  returned a span that appears **verbatim** in that statement. No span, no
  highlight; nothing is approximated.
- **Filters** — relationship, conflict category and minimum confidence,
  applied together, with a reset.
- **Analysis notes** — every warning the run produced (pair cap reached, rate
  limit, unclassified pairs) is shown rather than hidden, so the totals can
  always be interpreted.
- **Run detail** — a collapsible panel with the full pipeline funnel
  (cross-document pairs possible → above threshold → shortlisted → classified)
  and measured API usage.
- **Export** — JSON (full report plus document context and caveats) or CSV
  (one row per pair, 15 columns, UTF-8 BOM so Excel opens it correctly).
- **Ask about your documents** — a chat panel that answers questions from
  the extracted statements and shows the statements it used. See below.

## Document chat

A grounded question-answering panel, not a general assistant. It answers
only from the user's own documents and shows its working.

**How a question is answered**

1. The question is embedded (one small call) and compared against the
   statements already extracted from the documents.
2. The top 8 statements by cosine similarity become the context, together
   with any relationships the analysis already found among them — so
   "why do these disagree?" can be answered from work already paid for.
3. `gpt-4.1-mini` answers under a strict schema, instructed to use only
   those statements, to cite the ones it used, and to say so plainly when
   the documents do not contain the answer.
4. **Every citation is validated against what was actually retrieved.** A
   statement id the model invents is discarded rather than displayed, the
   same rule the conflict report follows.

**What the panel shows**

- Four suggested questions, generated once from the real content of this
  analysis and cached (a static fallback is used if that call fails, so
  the panel always has starters).
- The answer, followed by each source as *document · page · statement id*
  with the statement text underneath, so any claim can be checked by eye.
- An explicit "Not found in these documents" marker when the answer is
  not grounded, instead of a confident-sounding guess.
- Follow-up support: the last few turns are sent with each question, so
  "which one should I follow?" works.

**Verified live.** Three questions against the real API:

| Question | Behaviour |
|---|---|
| "What is the annual leave entitlement?" | Surfaced the disagreement and cited both documents — 20 days (2023 policy, `sent_003`) vs 15 days (2024 handbook, `sent_016`) |
| "Which one should I follow?" (follow-up) | Used the effective dates to recommend the 2024 handbook, citing both revision lines |
| "What is the company pension contribution rate?" | *"The statements do not contain information about the company pension contribution rate."* — `answered_from_documents: false`, zero sources, no guess |

A measured 550–1,200 tokens per question, roughly **$0.0005** each. The
statement vectors come from the embedding cache, so repeat questions
about the same documents cost nothing extra on the retrieval side.

One incidental finding: the two document metadata statements flagged as
noise in early development ("Revision 3.1, effective 01 April 2023",
"Edition 2024, effective 01 January 2024") turned out to be exactly what
the model needed to answer which document supersedes the other.

## Sample documents

`sample_documents\` holds three TXT files written so that the expected
answer is known in advance:

| Planted relationship | Where |
|---|---|
| Numerical conflict | 20 vs 15 days of annual leave |
| Numerical conflict | 14 vs 30 days of leave notice |
| Numerical conflict | 2 vs 3 remote working days per week |
| Date conflict | leave year starts 1 April vs 1 January |
| Policy conflict | carry-over of 5 days vs leave forfeited |
| Factual conflict | reviews twice per year vs once per year |
| Support | 10 days of sick leave in both documents |
| Support | medical certificate after 2 days in both |
| Support | staff on probation excluded from remote work in both |
| Support (cross-department) | 2 remote days in the HR and Finance documents |
| Identical text (skipped by design) | "The appraisal cycle for each year closes on 31 December." |
| Neutral | cafeteria hours, parking, flight class, hotel caps |

## Tests

```powershell
python -m pytest -q
```

```powershell
node tests/js/text-utils.test.js
```

**276 Python tests and 25 JavaScript assertions, every one offline with no
API spend** — `TestConfig` forces an empty `OPENAI_API_KEY`, so the suite
cannot touch the real API even when your `.env` holds a working key. The OpenAI SDK is
replaced by a configurable stub (`tests/stubs.py`) that can return good
replies, malformed JSON, truncated output, omitted pairs, invented pairs or
any SDK error on demand. Nothing in `app/` imports the stub, so stub output
can never surface in the application as real model output.

Coverage: upload validation, PDF/TXT extraction, failure reporting, text
cleaning, segmentation, provenance, embedding batching/dedup/caching, cosine
correctness, cross-document-only pairing, threshold and cap enforcement,
schema validation, confidence clamping, span verification, retry and
batch-splitting recovery, every error-kind mapping, the asynchronous analysis
workflow (stage progress, duplicate-submission guard, cached re-request,
forced re-run, failure states), JSON/CSV export structure, the metric
arithmetic (against hand-computed values), the evaluation harness, the
document chat (retrieval ranking, cache reuse, citation validation,
invented-citation rejection, ungrounded answers, history capping,
suggestion caching and fallback), and the full HTTP workflow.

The JavaScript tests cover the escaping boundary — the one place where
document text and model output become HTML — including hostile document
content, hostile model output, and the subtle case of a highlight span
positioned after an escaped character.

## Design decisions

- **No temporary files.** Uploads are read straight from the request into
  memory and handed to PyMuPDF as a byte stream, so there is nothing on disk
  to leak or clean up.
- **Failures are reported, never dropped.** A password-protected, corrupt,
  image-only or unsupported file becomes a `failed` row with a specific
  reason, and the remaining documents are still analysed.
- **No database.** Results live in a bounded, lock-guarded in-memory store
  keyed by `analysis_id` for the life of the process.
- **Embeddings never enter the stored payload.** `GET /api/analysis/<id>`
  serialises its record directly, so vectors are kept out of it and responses
  stay small.
- **Nothing is invented.** `evidence_a`/`evidence_b` and every document and
  page reference come from our own stored statements, never from the model.
  Highlight spans are accepted only when the phrase occurs **verbatim** in the
  statement it belongs to; a paraphrased or hallucinated span is discarded
  rather than displayed. Results for pairs that were never requested are
  dropped, and a pair the model omits is reported as unclassified instead of
  being guessed at.
- **Confidence is model-reported, not calibrated.** Every result carries
  `confidence_basis: "model_reported"`. It is the model's own stated
  confidence, not a statistically validated probability.
- **Cost is measured, not assumed.** Each report carries the token counts the
  API itself returned, plus a cost *estimate* derived from configured list
  prices and labelled as such.

## Evaluation

Two things are measured separately, because they fail for different
reasons: whether a related pair is even **shortlisted** (retrieval), and
whether a shortlisted pair is **classified** correctly.

### The labelled set

`evaluation/dataset.py` holds **42 hand-labelled statement pairs**, each
with a written justification for its label:

| Gold label | Pairs | What they probe |
|---|---|---|
| CONTRADICTION | 16 | all five conflict categories, including a unit mismatch (one month vs 60 days) |
| SUPPORT | 12 | paraphrases, and equivalent quantities in different units (20 days vs four weeks) |
| NEUTRAL | 14 | unrelated pairs, plus **8 hard negatives** |

The **hard negatives** are the important part. Each one looks like a
conflict because the numbers differ, but the scope, population or subject
differs too, so the correct answer is NEUTRAL:

- "Managers receive 25 days" vs "Interns receive 15 days" — different populations
- "4000 per night in metro cities" vs "2500 per night in non-metro cities" — different tiers
- "Sick leave cannot be carried over" vs "Annual leave may be carried over" — different leave types
- a general rule vs a stated exception to it
- "The leave year begins 1 April" vs "The financial year begins 1 April" — same date, different subject

### Running it

```powershell
# Harness self-test. Free, no network. Measures the harness, NOT the model.
python -m evaluation.run_evaluation --mode stub

# Real evaluation. Needs OPENAI_API_KEY. Costs a few cents.
python -m evaluation.run_evaluation --mode live

# Also measure the similarity gate on the sample documents.
python -m evaluation.run_evaluation --mode live --retrieval
```

Every run writes a timestamped `.json` and `.txt` into
`evaluation/results/`, and **stamps its mode in the output**. Stub-mode
files carry an explicit caveat that their figures are not model
performance. Reported metrics: per-class precision/recall/F1, macro and
weighted averages, a confusion matrix, accuracy both over classified
pairs and with unclassified pairs charged as errors, one-vs-rest
contradiction detection, the hard-negative slice, and indicative category
agreement. The arithmetic lives in `evaluation/metrics.py` and is unit
tested against hand-computed values.

### Baseline

Stub mode doubles as a **"numbers differ ⇒ contradiction" baseline**,
since that is effectively how the stub classifies. It scores:

| Metric | Baseline (stub) |
|---|---|
| Accuracy | 0.40 |
| Macro F1 | 0.41 |
| Contradiction precision / recall | 0.50 / 0.44 |
| **Hard negatives correct** | **2 / 8 (0.25)** |

That is the floor the reasoning model has to beat, and it confirms the
dataset actually discriminates — a naive numeric rule fails most of the
hard negatives, exactly as intended.

### Results (live run, gpt-4.1-mini)

Source: `evaluation/results/evaluation-live-20261030-160733.json`.
42 pairs, 42 classified, 0 unclassified.

```
                  CONT   SUPP   NEUT
CONTRADICTION       15      0      1
SUPPORT              0     12      0
NEUTRAL              0      1     13
(rows = gold, columns = predicted)
```

| Label | Precision | Recall | F1 | n |
|---|---|---|---|---|
| CONTRADICTION | 1.000 | 0.938 | 0.968 | 16 |
| SUPPORT | 0.923 | 1.000 | 0.960 | 12 |
| NEUTRAL | 0.929 | 0.929 | 0.929 | 14 |
| **macro avg** | **0.951** | **0.955** | **0.952** | 42 |
| weighted avg | 0.954 | 0.952 | 0.953 | 42 |

**Accuracy 0.952.** Contradiction detection one-vs-rest: **precision
1.000, recall 0.938** — 15 true positives, **0 false positives**, 1 miss.
Not a single support or neutral pair was wrongly reported as a conflict,
which is the error that would most damage trust in the tool.

Against the "numbers differ ⇒ contradiction" baseline:

| | Baseline | gpt-4.1-mini |
|---|---|---|
| Accuracy | 0.405 | **0.952** |
| Macro F1 | 0.410 | **0.952** |
| Contradiction precision | 0.500 | **1.000** |
| Hard negatives correct | 2/8 | **7/8** |

### Error analysis

Only two pairs were misclassified, and both are informative:

**`ev_009` — gold CONTRADICTION, predicted NEUTRAL.**
"This policy takes effect on 01 April 2023." vs "This policy takes effect
on 15 February 2024." The model's reasoning: *"The two statements specify
different policies with different effective dates, not conflicting values
for the same policy."* The gold label was marked "borderline by design"
when the set was written, and **the model's reading is arguably the better
one** — two documents may each carry their own effective date. This is a
weakness in the labelled set as much as a model error.

**`ev_033` — gold NEUTRAL, predicted SUPPORT** (a hard negative).
"Claims must be submitted within 30 days" vs "Claims delayed by certified
medical leave may be accepted after 30 days with director approval." The
model called the exception complementary to the rule. The purpose of this
hard negative was to check it is **not** reported as a conflict, and on
that it succeeded; the three-way label is genuinely debatable. Counting
"did not call it a contradiction" as the success criterion, hard negatives
are **8/8**.

**Category agreement: 12/15 (0.80).** All three disagreements assigned
`POLICY_CONFLICT` where the gold label was `FACTUAL_CONFLICT` (twice) or
`LOGICAL_CONFLICT`. The relationship was correct in every case. The
pattern suggests the model leans on the *genre* of the source documents —
these are policy documents, so conflicts in them read as policy conflicts
— rather than on the nature of the discrepancy. Category remains the
softest output of the system and is reported as indicative.

### Retrieval: the similarity gate

| Measure | Value |
|---|---|
| Statements from the 3 sample documents | 38 |
| Possible cross-document pairs | 481 |
| Pairs shortlisted at threshold 0.65, top-5 | 13 |
| Planted relationships located | 11 / 11 |
| **Planted relationships shortlisted** | **11 / 11 (recall 1.00)** |
| Shortlisted pairs that were planted | 11 / 13 |
| Cross-document work eliminated | **97.3%** |

The gate did its job: perfect recall of the known relationships while
sending only 13 of 481 possible pairs to the model.

It also validates the threshold empirically. Planted-pair similarity ran
from **0.716** (carry-over allowed vs forfeited) to **0.951**, so the
weakest true relationship clears the 0.65 threshold by only **0.066**.
Raising the threshold much above 0.70 would start dropping real conflicts
— which is the first thing to re-measure on a different corpus.

### Measured cost

| | Tokens | Requests |
|---|---|---|
| Embeddings | 578 | 1 |
| Reasoning input | 5,888 | 6 |
| Reasoning output | 3,508 | — |
| **Total** | **9,974** | 7, 0 failed |

**Estimated cost: $0.0080** for the whole evaluation (42 classified pairs
plus 38 embeddings and the retrieval check). Token counts are reported by
the API; the dollar figure is computed from configured list prices and is
not a billing figure.

That is about **4× cheaper than projected** — the $0.03 estimate assumed
~110 output tokens per pair, and the real average was ~84.

## API cost controls

| Control | Mechanism |
|---|---|
| Statement ceiling | `MAX_SENTENCES_PER_ANALYSIS` (400) checked before embedding |
| Embedding reuse | identical text embedded once per run; process-wide cache across runs |
| Request batching | 64 statements per embedding call, 8 pairs per GPT call |
| Candidate ceiling | top-5 neighbours, 0.65 threshold, hard cap of 100 pairs |
| Wasted-call avoidance | identical statement text is never sent for classification |
| Repeat-run avoidance | an existing report is reused unless `force` is passed |
| Output ceiling | `MAX_OUTPUT_TOKENS` per reasoning request |
| Chat question length | Capped at 500 characters, refused before any call |
| Chat history | Only the last 6 turns are sent |
| Suggestion reuse | Generated once per analysis and cached |
| Runaway-split guard | a per-analysis reasoning request budget caps batch splitting |

Published list prices put a 100-pair analysis at roughly **$0.03**
(embeddings ~$0.00002; ~13 reasoning requests at ~$0.028), which is about 600
capped runs inside an $18 budget. **These are estimates.** Actual measured
usage is recorded on every report under `usage` and will be reported from real
runs once a key is configured.

## Security and reliability

Audited by probing a running instance, not by inspection alone. All
findings below are covered by tests.

| Area | Control | Verified by |
|---|---|---|
| File type | Extension allowlist (`.pdf`, `.txt`); `.exe`, `.js`, `.zip`, double extensions and extensionless files rejected | probe + tests |
| File size | Per-file limit (10 MB) and a request-wide `MAX_CONTENT_LENGTH` returning JSON 413 | tests |
| File count | Minimum 2, maximum 10 per analysis | tests |
| Content/extension mismatch | A non-PDF named `.pdf` is reported as a failed document, never a crash | probe + tests |
| Filenames | `secure_filename` strips paths and traversal; `../../x.txt` cannot escape | probe + tests |
| Stored XSS | Document text and model output are escaped before reaching the DOM; the highlight function escapes both sides and searches the escaped text, so hostile text cannot inject a tag | 25 Node assertions |
| Export filenames | Slugged, so a crafted id cannot shape the `Content-Disposition` header | tests |
| API key | Read from the environment only; never logged, never in a response, never in an export. `/api/health` reports only *whether* one is configured. A sentinel-key probe found zero leaks across six endpoints | probe + tests |
| Key in tests | `TestConfig` forces an empty key, so the suite cannot spend against a real one | tests |
| Error detail | SDK messages are truncated and any `sk-` fragment is withheld; tracebacks go to the log, never to the client | tests |
| Malformed input | Invalid JSON, non-object bodies, non-string and oversized ids all return 4xx, not 500 | probe + tests |
| Malformed API responses | Bad JSON, truncated output, missing/invented pairs, unknown enums all handled with retry or an honest "unclassified" | tests |
| Temp files | None created — uploads are processed in memory, and `tempfile` is never imported anywhere in `app/` | probe |
| Duplicate submissions | A second analyse request while one is in flight cannot start a second paid run | tests |
| Thread safety | The analysis store is guarded by a lock; the worker thread only touches it through those helpers | review |

## Limitations

- Scanned/image-only PDFs yield no text; OCR is out of scope and such files
  are reported as failures rather than silently ignored.
- TXT files carry no page numbers, so their statements have `page_number: null`.
- The in-memory store is lost on restart and not shared across worker
  processes.
- Table-heavy PDFs segment poorly; cell text often fails the statement filter.
- **The engine has not yet been validated against the live API.** All results
  so far come from a stub client. Real classification quality, the real
  distribution of similarity scores, and therefore the real number of
  candidate pairs per analysis are unmeasured until a live run is made.
- The dashboard has been verified structurally (every element the script
  touches exists in the rendered page, assets serve, exports download) but
  **not visually in a browser** from this environment.
- The analysis runs on a thread inside the Flask dev server, which is fine for
  a single-user demo but is not a task queue: progress is lost if the process
  restarts mid-run.
- The similarity threshold of 0.65 is a starting point taken from typical
  `text-embedding-3-small` behaviour, not from measurement on this corpus.
- Cross-document pairs only: a document that contradicts *itself* is not
  checked.
- The chat retrieves by similarity over single statements, so a question
  whose answer is spread across many statements may get an incomplete
  one. It reports the statements it used, which makes that visible.
- Chat history lives in the browser, so a page reload starts a new
  conversation.
- Classification is pairwise, so a conflict that only emerges from three or
  more statements together will not be found.

## Project report

`REPORT_OUTLINE.md` is a chapter-by-chapter skeleton for the written
submission, with each section mapped to the file or command that produces
its evidence, plus a figure/table list and an honesty checklist.

## Roadmap

- A larger, independently labelled evaluation set, and repeat runs to
  measure variance.
- Rework the conflict-category taxonomy, or drop it to the three
  relationships plus the explanation, given 0.80 agreement.
- OCR for scanned PDFs, so image-only documents stop being failures.
- Self-consistency checking inside a single document, not only across
  documents.
- Multi-statement conflicts, where no single pair is contradictory but
  three statements together are.
- A persistent store and a real task queue, replacing the in-memory store
  and the worker thread.
- Human confirmation of findings, which would build the labelled data
  needed to turn model-reported confidence into a calibrated probability.
