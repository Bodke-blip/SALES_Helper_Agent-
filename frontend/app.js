const composer = document.getElementById("composer");
const query = document.getElementById("query");
const verbose = document.getElementById("verbose");
const llmMode = document.getElementById("llmMode");
const run = document.getElementById("run");
const newChat = document.getElementById("newChat");
const status = document.getElementById("status");
const elapsed = document.getElementById("elapsed");
const emptyState = document.getElementById("emptyState");
const chatList = document.getElementById("chatList");
const chatHistory = document.getElementById("chatHistory");
const fallback = document.getElementById("fallback");
const runtimeLog = document.getElementById("runtimeLog");
const trace = document.getElementById("trace");
const agentTrace = document.getElementById("agentTrace");
const sources = document.getElementById("sources");
const evaluation = document.getElementById("evaluation");
const models = document.getElementById("models");

function createSessionId() {
  return window.crypto && window.crypto.randomUUID
    ? window.crypto.randomUUID()
    : `session_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

let sessionId = localStorage.getItem("predikly_session_id") || createSessionId();
let visibleChatHistory = [];
let chatSessions = [];
localStorage.setItem("predikly_session_id", sessionId);

async function fetchJson(url, options = {}, timeoutMs = 25000) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });

    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    return await response.json();
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function messagePairsFromBackend(messages) {
  const turns = [];

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];

    if (message.role !== "user") {
      continue;
    }

    const nextMessage = messages[index + 1] || {};
    turns.push({
      query: message.content || "",
      answer: nextMessage.role === "assistant" ? nextMessage.content || "" : "",
      retrievalCollection: nextMessage.metadata?.retrieval_collection || ""
    });
  }

  return turns;
}

function pretty(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function formatElapsed(ms) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function scrollThreadToBottom() {
  chatHistory.scrollIntoView({ behavior: "smooth", block: "end" });
}

function collectionLabel(collectionName) {
  if (!collectionName) {
    return "";
  }

  if (collectionName.includes("fallback")) {
    return `Retrieved from fallback collection: ${collectionName}`;
  }

  return `Retrieved from main collection: ${collectionName}`;
}

function isSectionHeading(line) {
  const trimmed = line.trim();

  if (!trimmed || trimmed.length > 90) {
    return false;
  }

  if (trimmed.endsWith(":")) {
    return true;
  }

  return /^(business problem|solution|solution and workflow|workflow|tools|tools and systems used|benefits|benefits and outcomes|source grounding|sources|summary|use cases|recommendations|architecture|context|answer)$/i.test(trimmed);
}

function appendTextBlock(container, tagName, text, className = "") {
  const element = document.createElement(tagName);
  element.textContent = text;

  if (className) {
    element.className = className;
  }

  container.appendChild(element);
  return element;
}

function appendList(container, items, ordered = false) {
  const list = document.createElement(ordered ? "ol" : "ul");

  items.forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = item;
    list.appendChild(listItem);
  });

  container.appendChild(list);
}

function renderFormattedAnswer(container, content) {
  container.textContent = "";
  const lines = String(content || "").split(/\r?\n/);
  let paragraphLines = [];
  let listItems = [];
  let orderedItems = [];

  function flushParagraph() {
    if (!paragraphLines.length) {
      return;
    }

    appendTextBlock(container, "p", paragraphLines.join(" "));
    paragraphLines = [];
  }

  function flushLists() {
    if (listItems.length) {
      appendList(container, listItems, false);
      listItems = [];
    }

    if (orderedItems.length) {
      appendList(container, orderedItems, true);
      orderedItems = [];
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim();

    if (!line) {
      flushParagraph();
      flushLists();
      return;
    }

    const bulletMatch = line.match(/^[-•]\s+(.+)$/);
    const orderedMatch = line.match(/^\d+[.)]\s+(.+)$/);

    if (bulletMatch) {
      flushParagraph();
      orderedItems = [];
      listItems.push(bulletMatch[1].trim());
      return;
    }

    if (orderedMatch) {
      flushParagraph();
      listItems = [];
      orderedItems.push(orderedMatch[1].trim());
      return;
    }

    if (isSectionHeading(line)) {
      flushParagraph();
      flushLists();
      appendTextBlock(container, "h3", line.replace(/:$/, ""));
      return;
    }

    flushLists();
    paragraphLines.push(line);
  });

  flushParagraph();
  flushLists();
}

function createMessage(role, content, { pending = false, retrievalCollection = "" } = {}) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const header = document.createElement("div");
  header.className = "message-header";

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === "user" ? "Y" : "P";

  const label = document.createElement("span");
  label.textContent = role === "user" ? "You" : "Predikly";

  const body = document.createElement("div");
  body.className = "message-body";

  if (pending) {
    const typing = document.createElement("span");
    typing.className = "typing";
    typing.innerHTML = "<span></span><span></span><span></span>";
    body.appendChild(typing);
  } else if (role === "assistant") {
    renderFormattedAnswer(body, content);
  } else {
    body.textContent = content;
  }

  header.append(avatar, label);
  message.append(header, body);

  if (role === "assistant" && retrievalCollection) {
    const source = document.createElement("div");
    source.className = `collection-source${retrievalCollection.includes("fallback") ? " fallback" : ""}`;
    source.textContent = collectionLabel(retrievalCollection);
    message.appendChild(source);
  }

  return message;
}

function renderChatHistory() {
  chatHistory.textContent = "";

  if (visibleChatHistory.length > 0) {
    hideEmptyState();
  } else {
    showEmptyState();
  }

  visibleChatHistory.forEach((turn) => {
    chatHistory.appendChild(createMessage("user", turn.query || ""));
    chatHistory.appendChild(
      createMessage(
        "assistant",
        turn.answer || "",
        { retrievalCollection: turn.retrievalCollection || "" }
      )
    );
  });

  scrollThreadToBottom();
}

function showEmptyState() {
  emptyState.hidden = false;
  emptyState.classList.remove("is-hidden");
  emptyState.classList.add("is-visible");
}

function hideEmptyState() {
  emptyState.classList.remove("is-visible");
  emptyState.classList.add("is-hidden");
  window.setTimeout(() => {
    if (emptyState.classList.contains("is-hidden")) {
      emptyState.hidden = true;
    }
  }, 340);
}

function renderChatList() {
  chatList.textContent = "";

  if (!chatSessions.length) {
    const empty = document.createElement("div");
    empty.className = "chat-list-empty";
    empty.textContent = "No past chats";
    chatList.appendChild(empty);
    return;
  }

  chatSessions.forEach((session) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `chat-list-item${session.id === sessionId ? " active" : ""}`;
    item.textContent = session.title || "New chat";
    item.title = session.title || "New chat";
    item.addEventListener("click", () => loadChatSession(session.id));
    chatList.appendChild(item);
  });
}

async function refreshChatList() {
  try {
    const data = await fetchJson("/chats", {}, 8000);
    chatSessions = Array.isArray(data.sessions) ? data.sessions : [];
    renderChatList();
  } catch {
    chatSessions = [];
    renderChatList();
  }
}

async function loadChatSession(nextSessionId) {
  if (!nextSessionId || nextSessionId === sessionId) {
    return;
  }

  sessionId = nextSessionId;
  localStorage.setItem("predikly_session_id", sessionId);
  status.textContent = "Loading";

  try {
    const data = await fetchJson(`/chat/${encodeURIComponent(sessionId)}`, {}, 8000);
    visibleChatHistory = messagePairsFromBackend(data.messages || []);
  } catch {
    visibleChatHistory = [];
  }

  renderChatHistory();
  renderChatList();
  resetDebugPanels();
  status.textContent = "Ready";
  query.focus();
}

function appendPendingTurn(submittedQuery) {
  hideEmptyState();
  chatHistory.appendChild(createMessage("user", submittedQuery));
  const pendingMessage = createMessage("assistant", "", { pending: true });
  chatHistory.appendChild(pendingMessage);
  scrollThreadToBottom();
  return pendingMessage;
}

function updatePendingTurn(pendingMessage, finalAnswer, retrievalCollection = "") {
  const body = pendingMessage.querySelector(".message-body");
  renderFormattedAnswer(body, finalAnswer);

  if (retrievalCollection) {
    const existingSource = pendingMessage.querySelector(".collection-source");

    if (existingSource) {
      existingSource.textContent = collectionLabel(retrievalCollection);
      existingSource.classList.toggle("fallback", retrievalCollection.includes("fallback"));
    } else {
      const source = document.createElement("div");
      source.className = `collection-source${retrievalCollection.includes("fallback") ? " fallback" : ""}`;
      source.textContent = collectionLabel(retrievalCollection);
      pendingMessage.appendChild(source);
    }
  }

  scrollThreadToBottom();
}

function selectedLlmPreferences() {
  const mode = llmMode.value;
  return {
    label: {
      gemini_ollama: "Gemini with Ollama fallback",
      gemini_only: "Gemini only",
      ollama_only: "Ollama only"
    }[mode] || "Gemini with Ollama fallback",
    useGemini: mode !== "ollama_only",
    useLocal: mode !== "gemini_only"
  };
}

function setDebugPanels(data) {
  fallback.textContent = data.fallback_status || "not_used";
  trace.textContent = data.trace_id || "-";
  agentTrace.textContent = pretty(data.agent_trace || []);
  sources.textContent = pretty(data.sources || []);
  evaluation.textContent = pretty(data.evaluation || []);
  models.textContent = pretty(data.llm_models || {});
}

function resetDebugPanels() {
  fallback.textContent = "not_started";
  elapsed.textContent = "0.0s";
  runtimeLog.textContent = "No query run yet.";
  trace.textContent = "-";
  agentTrace.textContent = "[]";
  sources.textContent = "[]";
  evaluation.textContent = "[]";
  models.textContent = "{}";
}

function logRuntime(message) {
  const timestamp = new Date().toLocaleTimeString();
  const line = `[${timestamp}] ${message}`;
  runtimeLog.textContent = runtimeLog.textContent === "No query run yet."
    ? line
    : `${runtimeLog.textContent}\n${line}`;
}

function resizeComposer() {
  query.style.height = "auto";
  query.style.height = `${Math.min(query.scrollHeight, 190)}px`;
}

async function submitQuery() {
  const submittedQuery = query.value.trim();

  if (!submittedQuery) {
    return;
  }

  const startedAt = performance.now();
  let timerId = null;
  const pendingMessage = appendPendingTurn(submittedQuery);
  const llmPreferences = selectedLlmPreferences();

  run.disabled = true;
  newChat.disabled = true;
  query.value = "";
  resizeComposer();
  status.textContent = "Thinking";
  elapsed.textContent = "0.0s";
  runtimeLog.textContent = "";
  logRuntime(`Started query (retrieval on, ${llmPreferences.label}).`);

  timerId = window.setInterval(() => {
    elapsed.textContent = formatElapsed(performance.now() - startedAt);
  }, 100);

  try {
    const data = await fetchJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: submittedQuery,
        session_id: sessionId,
        use_gemini_llm: llmPreferences.useGemini,
        use_local_llm: llmPreferences.useLocal,
        verbose: verbose.checked
      })
    }, 30000);
    const finalAnswer = data.answer || data.message || "";
    const retrievalCollection = data.retrieval_collection || "";

    updatePendingTurn(pendingMessage, finalAnswer, retrievalCollection);
    visibleChatHistory.push({ query: submittedQuery, answer: finalAnswer, retrievalCollection });
    setDebugPanels(data);
    await refreshChatList();
    status.textContent = "Ready";
    elapsed.textContent = formatElapsed(performance.now() - startedAt);
    logRuntime(`Completed in ${elapsed.textContent}. Cache: ${data.cache_status || "unknown"}. Trace: ${data.trace_id || "-"}.`);
    (data.workflow_timings || []).forEach((item) => {
      const seconds = Number(item.duration_seconds ?? 0).toFixed(3);
      logRuntime(`${item.step}: ${seconds}s (${item.status || "completed"})`);
    });
  } catch (error) {
    const message = error.name === "AbortError"
      ? "The request timed out. Please try again with a more specific case-study question."
      : String(error);
    updatePendingTurn(pendingMessage, message);
    fallback.textContent = "error";
    status.textContent = "Error";
    elapsed.textContent = formatElapsed(performance.now() - startedAt);
    logRuntime(`Failed after ${elapsed.textContent}: ${String(error)}`);
  } finally {
    if (timerId) window.clearInterval(timerId);
    run.disabled = false;
    newChat.disabled = false;
    query.focus();
  }
}

async function startNewChat() {
  sessionId = createSessionId();
  visibleChatHistory = [];
  localStorage.setItem("predikly_session_id", sessionId);
  renderChatHistory();
  renderChatList();

  query.value = "";
  resizeComposer();
  resetDebugPanels();
  status.textContent = "Ready";
  query.focus();
}

async function initializeChatUi() {
  status.textContent = "Loading";

  try {
    await refreshChatList();

    if (chatSessions.some((session) => session.id === sessionId)) {
      const data = await fetchJson(`/chat/${encodeURIComponent(sessionId)}`, {}, 8000);
      visibleChatHistory = messagePairsFromBackend(data.messages || []);
    } else {
      visibleChatHistory = [];
    }
  } catch {
    visibleChatHistory = [];
    chatSessions = [];
    renderChatList();
  } finally {
    status.textContent = "Ready";
  }

  renderChatHistory();
  renderChatList();
  resizeComposer();
}

initializeChatUi();

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuery();
});

query.addEventListener("input", resizeComposer);
query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuery();
  }
});

newChat.addEventListener("click", startNewChat);
