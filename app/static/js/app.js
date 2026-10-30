/* CrossDoc Intelligence dashboard.
   Upload -> analyze -> poll real server progress -> render conflicts.
   Every value shown here comes from the API; nothing is simulated. */
(function () {
  "use strict";

  var cfg = window.APP_CONFIG || {
    allowedExtensions: ["pdf", "txt"],
    maxFileSizeMb: 10,
    minFiles: 2,
    maxFiles: 10,
    stages: []
  };

  var POLL_INTERVAL_MS = 700;
  var POLL_TIMEOUT_MS = 180000;

  var STAGE_LABELS = {
    embedding: "Generating embeddings",
    candidate_selection: "Finding related statements",
    reasoning: "Analysing pairs for contradictions",
    report: "Building the conflict report"
  };

  var RELATIONSHIP_CLASS = {
    CONTRADICTION: "pill-contradiction",
    SUPPORT: "pill-support",
    NEUTRAL: "pill-neutral"
  };

  var el = {};
  [
    "dropzone", "fileInput", "fileList", "fileListWrap", "fileCount",
    "clearFiles", "analyzeBtn", "analyzeLabel", "analyzeSpinner",
    "statusText", "alertArea", "emptyState", "results", "statRow",
    "warningPanel", "progressPanel", "stageList", "progressMessage",
    "relFilter", "catFilter", "confFilter", "resetFilters", "resultMeta",
    "conflictList", "exportJson", "exportCsv", "runDetailBody",
    "docFilter", "textFilter", "sentenceMeta", "sentencesBody",
    "modelFooter", "chatSuggestions", "chatLog", "chatForm", "chatInput",
    "chatSend", "chatSendLabel", "chatSpinner", "chatError"
  ].forEach(function (id) { el[id] = document.getElementById(id); });

  var selected = [];
  var state = {
    analysisId: null,
    documents: [],
    sentences: [],
    results: [],
    busy: false,
    pollTimer: null,
    // Chat history is kept client-side and sent with each question so
    // follow-ups like "why?" have context. The server stays stateless.
    chatHistory: [],
    chatBusy: false
  };

  /* ================= helpers ================= */

  // Shared with text-utils.js, which is unit tested on its own. Every
  // piece of document text and model output reaches the DOM through
  // escapeHtml or highlight, so those two are kept testable.
  var textUtils = window.CrossDocText;
  var escapeHtml = textUtils.escapeHtml;
  var highlight = textUtils.highlight;
  var extensionOf = textUtils.extensionOf;
  var formatSize = textUtils.formatSize;
  var titleCase = textUtils.titleCase;

  function showAlert(kind, message, items) {
    var list = "";
    if (items && items.length) {
      list = "<ul>" + items.map(function (item) {
        return "<li>" + escapeHtml(item) + "</li>";
      }).join("") + "</ul>";
    }
    el.alertArea.innerHTML =
      '<div class="alert alert-' + kind + ' mb-0" role="alert">' +
      escapeHtml(message) + list + "</div>";
  }

  function clearAlert() { el.alertArea.innerHTML = ""; }

  function pageRef(source) {
    if (!source) { return ""; }
    return source.page_number === null || source.page_number === undefined
      ? "" : " &middot; page " + escapeHtml(source.page_number);
  }

  /* ================= file selection ================= */

  function addFiles(fileArray) {
    var rejected = [];

    Array.prototype.forEach.call(fileArray, function (file) {
      var ext = extensionOf(file.name);

      if (cfg.allowedExtensions.indexOf(ext) === -1) {
        rejected.push('"' + file.name + '" is not a supported file type ('
          + cfg.allowedExtensions.map(function (e) { return "." + e; }).join(", ")
          + " only).");
        return;
      }
      if (file.size === 0) {
        rejected.push('"' + file.name + '" is empty (0 bytes).');
        return;
      }
      if (file.size > cfg.maxFileSizeMb * 1024 * 1024) {
        rejected.push('"' + file.name + '" is ' + formatSize(file.size)
          + ", above the " + cfg.maxFileSizeMb + " MB limit.");
        return;
      }
      if (selected.some(function (existing) {
        return existing.name === file.name && existing.size === file.size;
      })) {
        rejected.push('"' + file.name + '" was already added.');
        return;
      }
      if (selected.length >= cfg.maxFiles) {
        rejected.push('"' + file.name + '" was skipped: at most '
          + cfg.maxFiles + " documents can be analysed at once.");
        return;
      }
      selected.push(file);
    });

    if (rejected.length) {
      showAlert("warning", "Some files were not added:", rejected);
    } else {
      clearAlert();
    }
    renderFileList();
  }

  function renderFileList() {
    el.fileListWrap.classList.toggle("d-none", selected.length === 0);
    el.fileCount.textContent = selected.length
      ? "(" + selected.length + ")" : "";

    el.fileList.innerHTML = selected.map(function (file, index) {
      return '<li class="file-row">' +
        "<div>" +
          '<div class="file-name">' + escapeHtml(file.name) + "</div>" +
          '<div class="file-meta">' + extensionOf(file.name).toUpperCase() +
            " &middot; " + formatSize(file.size) + "</div>" +
        "</div>" +
        '<button type="button" class="file-remove" data-index="' + index +
          '" aria-label="Remove ' + escapeHtml(file.name) + '">&times;</button>' +
        "</li>";
    }).join("");

    updateAnalyzeButton();
  }

  function updateAnalyzeButton() {
    el.analyzeBtn.disabled = selected.length < cfg.minFiles || state.busy;

    if (state.busy) { return; }
    if (selected.length === 0) {
      el.statusText.textContent = "";
    } else if (selected.length < cfg.minFiles) {
      el.statusText.textContent = "Add at least " + cfg.minFiles
        + " documents to compare.";
    } else {
      el.statusText.textContent = selected.length + " documents ready.";
    }
  }

  function setBusy(busy, label) {
    state.busy = busy;
    el.analyzeSpinner.classList.toggle("d-none", !busy);
    el.analyzeLabel.textContent = busy
      ? (label || "Working…") : "Analyze documents";
    el.dropzone.style.pointerEvents = busy ? "none" : "";
    el.clearFiles.disabled = busy;
    updateAnalyzeButton();
  }

  /* ================= progress ================= */

  function renderStages(currentStage, message) {
    var stages = cfg.stages && cfg.stages.length
      ? cfg.stages
      : ["embedding", "candidate_selection", "reasoning", "report"];
    var position = stages.indexOf(currentStage);

    el.progressPanel.classList.remove("d-none");
    el.stageList.innerHTML = stages.map(function (stage, index) {
      var cls = "stage-item";
      var mark = "";
      if (position === -1 || index > position) {
        cls += " is-pending";
        mark = "○";
      } else if (index === position) {
        cls += " is-active";
        mark = "●";
      } else {
        cls += " is-done";
        mark = "✓";
      }
      return '<li class="' + cls + '"><span class="stage-mark">' + mark +
        "</span>" + escapeHtml(STAGE_LABELS[stage] || titleCase(stage)) +
        "</li>";
    }).join("");

    el.progressMessage.textContent = message || "";
  }

  function hideProgress() {
    el.progressPanel.classList.add("d-none");
    el.progressMessage.textContent = "";
  }

  /* ================= rendering ================= */

  function renderStats(overview) {
    var tiles = [
      ["Documents analysed", overview.documents_analyzed],
      ["Statements extracted", overview.statements_extracted],
      ["Pairs analysed", overview.candidate_pairs_analyzed],
      ["Contradictions", overview.contradictions_detected, "stat-danger"],
      ["Supporting", overview.supporting_relationships, "stat-success"],
      ["Neutral", overview.neutral_relationships]
    ];

    el.statRow.innerHTML = tiles.map(function (tile) {
      return '<div class="col-6 col-md-4 col-xl-2">' +
        '<div class="stat-card">' +
          '<div class="stat-value ' + (tile[2] || "") + '">' +
            Number(tile[1] || 0).toLocaleString() + "</div>" +
          '<div class="stat-label">' + escapeHtml(tile[0]) + "</div>" +
        "</div></div>";
    }).join("");
  }

  function renderWarnings(report) {
    var warnings = report.warnings || [];
    var unclassified = (report.overview || {}).pairs_unclassified || 0;

    if (!warnings.length && !unclassified) {
      el.warningPanel.classList.add("d-none");
      el.warningPanel.innerHTML = "";
      return;
    }

    var items = warnings.map(function (item) {
      return "<li><strong>" + escapeHtml(titleCase(item.kind)) + ":</strong> " +
        escapeHtml(item.message) + "</li>";
    });
    if (unclassified) {
      items.push("<li><strong>Unclassified:</strong> " + unclassified +
        " pair(s) could not be classified and are not counted in the totals " +
        "above.</li>");
    }

    el.warningPanel.classList.remove("d-none");
    el.warningPanel.innerHTML =
      '<div class="alert alert-warning mb-0"><strong>Analysis notes</strong>' +
      "<ul class=\"mb-0 mt-2\">" + items.join("") + "</ul></div>";
  }

  function conflictCard(item) {
    var relClass = RELATIONSHIP_CLASS[item.relationship] || "pill-neutral";
    var confidence = item.confidence === null || item.confidence === undefined
      ? "n/a" : Number(item.confidence).toFixed(2);

    var category = item.category
      ? '<span class="pill pill-category">' +
        escapeHtml(titleCase(item.category)) + "</span>"
      : "";

    return '<article class="conflict-card ' +
        (item.relationship === "CONTRADICTION" ? "is-conflict" : "") + '">' +
      '<header class="conflict-head">' +
        '<div class="d-flex align-items-center gap-2 flex-wrap">' +
          '<span class="pill ' + relClass + '">' +
            escapeHtml(titleCase(item.relationship)) + "</span>" +
          category +
        "</div>" +
        '<div class="conflict-metrics">' +
          '<span title="Confidence reported by the model, not a calibrated probability">' +
            "confidence " + confidence + "</span>" +
          '<span title="Cosine similarity used only to shortlist this pair">' +
            "similarity " + escapeHtml(item.similarity) + "</span>" +
        "</div>" +
      "</header>" +

      '<div class="conflict-body">' +
        '<div class="statement-side">' +
          '<div class="statement-source">' +
            escapeHtml((item.document_a || {}).document_name) +
            pageRef(item.document_a) +
            ' <span class="mono">' + escapeHtml(item.sentence_id_a) + "</span>" +
          "</div>" +
          '<blockquote class="statement-text">' +
            highlight(item.evidence_a, item.conflicting_span_a) +
          "</blockquote>" +
        "</div>" +
        '<div class="statement-side">' +
          '<div class="statement-source">' +
            escapeHtml((item.document_b || {}).document_name) +
            pageRef(item.document_b) +
            ' <span class="mono">' + escapeHtml(item.sentence_id_b) + "</span>" +
          "</div>" +
          '<blockquote class="statement-text">' +
            highlight(item.evidence_b, item.conflicting_span_b) +
          "</blockquote>" +
        "</div>" +
      "</div>" +

      '<footer class="conflict-explanation">' +
        '<span class="explanation-label">AI explanation</span>' +
        escapeHtml(item.explanation) +
      "</footer>" +
    "</article>";
  }

  function renderConflicts() {
    var relationship = el.relFilter.value;
    var category = el.catFilter.value;
    var minimum = parseFloat(el.confFilter.value);

    var visible = state.results.filter(function (item) {
      if (relationship && item.relationship !== relationship) { return false; }
      if (category && item.category !== category) { return false; }
      if (!isNaN(minimum)) {
        if (item.confidence === null || item.confidence === undefined) {
          return false;
        }
        if (item.confidence < minimum) { return false; }
      }
      return true;
    });

    if (!state.results.length) {
      el.conflictList.innerHTML =
        '<p class="no-results">No statement pairs were classified. See the ' +
        "analysis notes above for the reason.</p>";
    } else if (!visible.length) {
      el.conflictList.innerHTML =
        '<p class="no-results">No pairs match the current filters.</p>';
    } else {
      el.conflictList.innerHTML = visible.map(conflictCard).join("");
    }

    el.resultMeta.textContent = "Showing " + visible.length + " of " +
      state.results.length + " classified pairs.";
  }

  function renderRunDetail(report) {
    var usage = report.usage || {};
    var stages = report.stages || {};
    var candidates = stages.candidate_selection || {};
    var reasoning = stages.reasoning || {};
    var embedding = stages.embedding || {};

    function row(label, value) {
      return "<tr><th scope=\"row\">" + escapeHtml(label) + "</th><td>" +
        escapeHtml(value) + "</td></tr>";
    }

    el.runDetailBody.innerHTML =
      '<div class="row g-4">' +
        '<div class="col-12 col-lg-6"><h3 class="detail-heading">Pipeline</h3>' +
          '<table class="table detail-table mb-0">' +
            row("Embedding model", (report.models || {}).embedding) +
            row("Reasoning model", (report.models || {}).reasoning) +
            row("Statements embedded", embedding.embedded) +
            row("Embedding cache hits", embedding.cache_hits) +
            row("Cross-document pairs possible", candidates.cross_document_pairs) +
            row("Above similarity threshold (" + candidates.threshold + ")",
                candidates.above_threshold) +
            row("Identical pairs skipped", candidates.skipped_identical) +
            row("Shortlisted for analysis", candidates.selected) +
            row("Dropped by pair cap", candidates.dropped_by_cap) +
            row("Reasoning requests", reasoning.requests) +
            row("Pairs classified", reasoning.pairs_classified) +
          "</table></div>" +
        '<div class="col-12 col-lg-6"><h3 class="detail-heading">API usage</h3>' +
          '<table class="table detail-table mb-0">' +
            row("Embedding tokens", usage.embedding_tokens) +
            row("Reasoning input tokens", usage.reasoning_input_tokens) +
            row("Reasoning output tokens", usage.reasoning_output_tokens) +
            row("Total tokens", usage.total_tokens) +
            row("Failed requests", usage.failed_requests) +
            row("Estimated cost (USD)", usage.estimated_cost_usd) +
          "</table>" +
          '<p class="card-sub mt-2 mb-0">' + escapeHtml(usage.cost_basis || "") +
          "</p></div>" +
      "</div>";
  }

  function populateDocFilter() {
    var options = ['<option value="">All documents</option>'];
    state.documents.filter(function (doc) { return doc.status === "success"; })
      .forEach(function (doc) {
        options.push('<option value="' + escapeHtml(doc.document_id) + '">' +
          escapeHtml(doc.document_name) + "</option>");
      });
    el.docFilter.innerHTML = options.join("");
  }

  function renderSentences() {
    var docId = el.docFilter.value;
    var needle = el.textFilter.value.trim().toLowerCase();

    var visible = state.sentences.filter(function (item) {
      if (docId && item.document_id !== docId) { return false; }
      if (needle && item.text.toLowerCase().indexOf(needle) === -1) {
        return false;
      }
      return true;
    });

    if (!visible.length) {
      el.sentencesBody.innerHTML =
        '<tr><td colspan="5" class="text-center text-muted py-4">' +
        "No statements match the current filters.</td></tr>";
    } else {
      el.sentencesBody.innerHTML = visible.map(function (item) {
        return "<tr>" +
          '<td class="mono">' + escapeHtml(item.sentence_id) + "</td>" +
          "<td>" + escapeHtml(item.document_name) + "</td>" +
          "<td>" + (item.page_number === null
            ? '<span class="pill pill-muted">n/a</span>'
            : escapeHtml(item.page_number)) + "</td>" +
          '<td class="sentence-text">' + escapeHtml(item.text) + "</td>" +
          '<td class="text-end">' + escapeHtml(item.word_count) + "</td>" +
          "</tr>";
      }).join("");
    }

    el.sentenceMeta.textContent = "Showing " + visible.length + " of " +
      state.sentences.length + " statements.";
  }

  function renderReport(report) {
    state.results = report.results || [];

    renderStats(report.overview || {});
    renderWarnings(report);
    renderConflicts();
    renderRunDetail(report);
    populateDocFilter();
    renderSentences();

    el.exportJson.href = "/api/export/" + state.analysisId + "?format=json";
    el.exportCsv.href = "/api/export/" + state.analysisId + "?format=csv";
    el.modelFooter.textContent = (report.models || {}).reasoning
      ? "Models: " + report.models.embedding + " · " + report.models.reasoning
      : "";

    resetChat();
    loadChatSuggestions();

    el.emptyState.classList.add("d-none");
    el.results.classList.remove("d-none");
  }

  /* ================= chat ================= */

  function renderChatSuggestions(questions) {
    if (!questions || !questions.length) {
      el.chatSuggestions.innerHTML = "";
      return;
    }
    el.chatSuggestions.innerHTML = questions.map(function (question) {
      return '<button type="button" class="suggestion-chip">' +
        escapeHtml(question) + "</button>";
    }).join("");
  }

  function loadChatSuggestions() {
    if (!state.analysisId) { return; }
    fetch("/api/chat/suggestions/" + state.analysisId)
      .then(readJson)
      .then(function (result) {
        if (result.ok) { renderChatSuggestions(result.body.questions); }
      })
      .catch(function () { /* suggestions are optional */ });
  }

  function chatSourceList(sources) {
    if (!sources || !sources.length) { return ""; }
    return '<div class="chat-sources">' +
      '<span class="chat-sources-label">Sources</span>' +
      sources.map(function (source) {
        var page = source.page_number === null ||
          source.page_number === undefined
          ? "" : " &middot; page " + escapeHtml(source.page_number);
        return '<div class="chat-source">' +
          '<span class="chat-source-ref">' +
            escapeHtml(source.document_name) + page +
            ' <span class="mono">' + escapeHtml(source.sentence_id) +
          "</span></span>" +
          '<span class="chat-source-text">' + escapeHtml(source.text) +
          "</span></div>";
      }).join("") + "</div>";
  }

  function appendChatMessage(role, html, extraClass) {
    var placeholder = el.chatLog.querySelector(".chat-placeholder");
    if (placeholder) { placeholder.remove(); }

    var bubble = document.createElement("div");
    bubble.className = "chat-message chat-" + role +
      (extraClass ? " " + extraClass : "");
    bubble.innerHTML = html;
    el.chatLog.appendChild(bubble);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return bubble;
  }

  function setChatBusy(busy) {
    state.chatBusy = busy;
    el.chatSpinner.classList.toggle("d-none", !busy);
    el.chatSendLabel.textContent = busy ? "Asking" : "Ask";
    el.chatSend.disabled = busy;
    el.chatInput.disabled = busy;
  }

  function askQuestion(question) {
    question = (question || "").trim();
    if (!question || state.chatBusy || !state.analysisId) { return; }

    el.chatError.innerHTML = "";
    el.chatInput.value = "";
    appendChatMessage("user", escapeHtml(question));

    var thinking = appendChatMessage(
      "assistant",
      '<span class="chat-thinking">Searching the documents&hellip;</span>',
      "is-thinking"
    );
    setChatBusy(true);

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_id: state.analysisId,
        question: question,
        history: state.chatHistory.slice(-6)
      })
    })
      .then(readJson)
      .then(function (result) {
        thinking.remove();

        if (!result.ok) {
          appendChatMessage("assistant",
            escapeHtml(result.body.error || "That question could not be " +
              "answered."), "is-error");
          return;
        }

        var body = result.body;
        var grounded = body.answered_from_documents;
        appendChatMessage(
          "assistant",
          '<p class="chat-answer">' + escapeHtml(body.answer) + "</p>" +
            (grounded
              ? chatSourceList(body.sources)
              : '<p class="chat-ungrounded">Not found in these ' +
                "documents.</p>"),
          grounded ? "" : "is-ungrounded"
        );

        state.chatHistory.push({ role: "user", content: question });
        state.chatHistory.push({ role: "assistant", content: body.answer });
      })
      .catch(function () {
        thinking.remove();
        appendChatMessage("assistant",
          "Could not reach the server.", "is-error");
      })
      .then(function () {
        setChatBusy(false);
        el.chatInput.focus();
      });
  }

  function resetChat() {
    state.chatHistory = [];
    el.chatLog.innerHTML =
      '<p class="chat-placeholder">Ask a question, or pick one of the ' +
      "suggestions above.</p>";
    el.chatSuggestions.innerHTML = "";
    el.chatError.innerHTML = "";
  }

  /* ================= network ================= */

  function readJson(response) {
    return response.json().then(function (body) {
      return { ok: response.ok, status: response.status, body: body };
    }).catch(function () {
      return {
        ok: false,
        status: response.status,
        body: {
          error: "The server returned an unreadable response (HTTP " +
            response.status + ")."
        }
      };
    });
  }

  function run() {
    if (state.busy || selected.length < cfg.minFiles) { return; }

    clearAlert();
    state.results = [];
    setBusy(true, "Uploading…");
    el.statusText.textContent = "Extracting text and segmenting statements…";
    hideProgress();

    var form = new FormData();
    selected.forEach(function (file) { form.append("documents", file); });

    fetch("/api/upload", { method: "POST", body: form })
      .then(readJson)
      .then(function (result) {
        if (!result.ok) {
          throw describeFailure(result, "The documents could not be processed.");
        }

        state.analysisId = result.body.analysis_id;
        state.documents = result.body.documents || [];
        state.sentences = result.body.sentences || [];

        var failures = result.body.errors || [];
        if (failures.length) {
          showAlert("warning", failures.length + " document(s) could not be " +
            "processed and were left out of the analysis:",
            failures.map(function (f) {
              return f.document_name + ": " + f.error;
            }));
        }

        setBusy(true, "Analysing…");
        el.statusText.textContent = "";
        renderStages(cfg.stages[0], "Starting analysis");

        return fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ analysis_id: state.analysisId })
        }).then(readJson);
      })
      .then(function (result) {
        if (!result.ok) {
          throw describeFailure(result, "The analysis could not be started.");
        }
        if (result.body.analysis) {        // already analysed
          return result.body.analysis;
        }
        return pollForResults();
      })
      .then(function (report) {
        hideProgress();
        renderReport(report);
        if (!(report.warnings || []).length) {
          showAlert("success", "Analysis complete: " +
            report.overview.contradictions_detected + " contradiction(s) in " +
            report.overview.candidate_pairs_analyzed + " analysed pairs.");
        }
      })
      .catch(function (error) {
        hideProgress();
        if (error && error.__described) {
          showAlert("danger", error.message, error.items);
        } else {
          showAlert("danger", "Could not reach the server. Check that the " +
            "application is still running.");
        }
      })
      .then(function () {
        setBusy(false);
        updateAnalyzeButton();
      });
  }

  function describeFailure(result, fallback) {
    var items = (result.body.errors || []).map(function (item) {
      return item.document_name + ": " + item.error;
    });
    var error = new Error(result.body.error || fallback);
    error.__described = true;
    error.items = items;
    return error;
  }

  function pollForResults() {
    var startedAt = Date.now();

    return new Promise(function (resolve, reject) {
      function tick() {
        fetch("/api/results/" + state.analysisId)
          .then(readJson)
          .then(function (result) {
            if (!result.ok) {
              return reject(describeFailure(
                result, "The analysis status could not be read."
              ));
            }

            var body = result.body;
            if (body.state === "failed") {
              var error = new Error(body.error || "The analysis failed.");
              error.__described = true;
              return reject(error);
            }
            if (body.state === "done" && body.analysis) {
              return resolve(body.analysis);
            }

            renderStages(body.stage, body.message);

            if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
              var timeout = new Error(
                "The analysis is taking longer than expected. It may still " +
                "finish - reload the page and check again."
              );
              timeout.__described = true;
              return reject(timeout);
            }
            state.pollTimer = window.setTimeout(tick, POLL_INTERVAL_MS);
          })
          .catch(reject);
      }
      tick();
    });
  }

  /* ================= events ================= */

  el.dropzone.addEventListener("click", function () { el.fileInput.click(); });

  el.dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      el.fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach(function (name) {
    el.dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      el.dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach(function (name) {
    el.dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      el.dropzone.classList.remove("is-dragover");
    });
  });

  el.dropzone.addEventListener("drop", function (event) {
    if (state.busy) { return; }
    if (event.dataTransfer && event.dataTransfer.files.length) {
      addFiles(event.dataTransfer.files);
    }
  });

  el.fileInput.addEventListener("change", function (event) {
    addFiles(event.target.files);
    event.target.value = "";
  });

  el.fileList.addEventListener("click", function (event) {
    var button = event.target.closest(".file-remove");
    if (button && !state.busy) {
      selected.splice(Number(button.dataset.index), 1);
      clearAlert();
      renderFileList();
    }
  });

  el.clearFiles.addEventListener("click", function () {
    if (state.busy) { return; }
    selected = [];
    clearAlert();
    renderFileList();
  });

  el.analyzeBtn.addEventListener("click", run);

  [el.relFilter, el.catFilter, el.confFilter].forEach(function (control) {
    control.addEventListener("change", renderConflicts);
  });

  el.resetFilters.addEventListener("click", function () {
    el.relFilter.value = "";
    el.catFilter.value = "";
    el.confFilter.value = "";
    renderConflicts();
  });

  el.chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    askQuestion(el.chatInput.value);
  });

  el.chatSuggestions.addEventListener("click", function (event) {
    var chip = event.target.closest(".suggestion-chip");
    if (chip && !state.chatBusy) { askQuestion(chip.textContent); }
  });

  el.docFilter.addEventListener("change", renderSentences);
  el.textFilter.addEventListener("input", renderSentences);

  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) { e.preventDefault(); });

  renderFileList();
})();
