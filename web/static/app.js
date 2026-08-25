/* Aster & Row — AI Support Agent frontend.
   All answers, citations, order data, conflict/handoff states and evaluation
   results come from the real backend API (/api/chat, /api/eval). Nothing here
   is mocked or hardcoded. */
"use strict";

const $ = (sel) => document.querySelector(sel);

const chatLog = $("#chat-log");
const composer = $("#composer");
const input = $("#chat-input");
const sendBtn = $("#send-btn");
const newChatBtn = $("#new-chat");
const devToggle = $("#dev-toggle");
const devPanel = $("#dev-panel");
const statusDot = $("#status-dot");
const statusLabel = $("#status-label");

let sessionId = "web-" + Math.random().toString(36).slice(2, 10);
let busy = false;
let devMode = false;
let typingRow = null;

/* ---------------- helpers ---------------- */

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderRichText(container, text) {
  // Minimal renderer for the backend's plain-text answers:
  // paragraphs on blank lines, "- " bullet lines, **bold** emphasis.
  container.innerHTML = "";
  const blocks = String(text).split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").filter((l) => l.trim().length > 0);
    if (!lines.length) continue;
    if (lines.every((l) => /^\s*[-•]\s+/.test(l))) {
      const ul = document.createElement("ul");
      for (const line of lines) {
        const li = document.createElement("li");
        li.innerHTML = inline(escapeHtml(line.replace(/^\s*[-•]\s+/, "")));
        ul.appendChild(li);
      }
      container.appendChild(ul);
    } else {
      const p = document.createElement("p");
      p.innerHTML = inline(escapeHtml(lines.join(" ")));
      container.appendChild(p);
    }
  }
}

function inline(escaped) {
  return escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function hideWelcome() {
  const welcome = $("#welcome");
  if (welcome) welcome.remove();
}

/* ---------------- message rendering ---------------- */

function addUserMessage(text) {
  hideWelcome();
  const tpl = $("#tpl-user-msg");
  const node = tpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".bubble-user").textContent = text;
  chatLog.appendChild(node);
  scrollToBottom();
}

function addBanner(parent, cls, icon, title, bodyText) {
  const el = document.createElement("div");
  el.className = "banner " + cls;
  el.setAttribute("role", "note");
  const i = document.createElement("span");
  i.className = "banner-icon";
  i.setAttribute("aria-hidden", "true");
  i.textContent = icon;
  const content = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = title;
  content.appendChild(strong);
  if (bodyText) {
    const p = document.createElement("p");
    p.textContent = bodyText;
    content.appendChild(p);
  }
  el.append(i, content);
  parent.appendChild(el);
}

function addSources(parent, sources) {
  if (!sources || !sources.length) return;
  const details = document.createElement("details");
  details.className = "sources";
  const summary = document.createElement("summary");
  summary.textContent =
    "Sources" + (sources.length > 1 ? " (" + sources.length + ")" : "");
  details.appendChild(summary);
  const list = document.createElement("div");
  list.className = "source-list";
  for (const s of sources) {
    const item = document.createElement("div");
    item.className = "source-item";
    const file = document.createElement("div");
    file.className = "source-file";
    file.textContent = s.file;
    const heading = document.createElement("div");
    heading.className = "source-heading";
    heading.textContent = s.heading || "";
    const docId = document.createElement("span");
    docId.className = "source-docid";
    docId.textContent = s.document_id || "";
    item.append(file, heading, docId);
    list.appendChild(item);
  }
  details.appendChild(list);
  parent.appendChild(details);
}

const ORDER_FIELD_LABELS = [
  ["status", "Status"],
  ["carrier", "Carrier"],
  ["estimated_delivery", "Estimated delivery"],
  ["membership_tier", "Membership"],
];
// Rendered verbatim from the backend's sanitized tool result. Any field not
// listed here (and not present in the payload) is never displayed.

function addOrderCard(parent, order) {
  if (!order || !order.found) return;
  const card = document.createElement("div");
  card.className = "order-card";
  const h = document.createElement("h4");
  h.textContent = "Order " + (order.order_id || "");
  card.appendChild(h);
  const dl = document.createElement("dl");
  dl.className = "order-fields";

  for (const [key, label] of ORDER_FIELD_LABELS) {
    const value = order[key];
    if (value === null || value === undefined || value === "") continue;
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    if (key === "status") {
      const pill = document.createElement("span");
      pill.className = "order-status-pill";
      pill.textContent = value;
      dd.appendChild(pill);
    } else {
      dd.textContent = value;
    }
    dl.append(dt, dd);
  }
  if (dl.children.length === 0) {
    // Found but no displayable fields — the answer text carries the info.
    return;
  }
  card.appendChild(dl);
  if (Array.isArray(order.items) && order.items.length) {
    const meta = document.createElement("p");
    meta.style.cssText = "margin:8px 0 0;font-size:.78rem;color:var(--ink-soft)";
    meta.textContent =
      order.items.length +
      " item" +
      (order.items.length > 1 ? "s" : "") +
      " on this order";
    card.appendChild(meta);
  }
  parent.appendChild(card);
}

function addDevTrace(parent, debug) {
  if (!devMode || !debug) return;
  const details = document.createElement("details");
  details.className = "devtrace";
  const summary = document.createElement("summary");
  summary.textContent = "Developer trace (sanitized)";
  details.appendChild(summary);
  const body = document.createElement("div");
  body.className = "devtrace-body";

  const section = (title, obj) => {
    if (obj === undefined || obj === null) return;
    if (Array.isArray(obj) && obj.length === 0) return;
    if (typeof obj === "object" && Object.keys(obj).length === 0) return;
    const wrap = document.createElement("section");
    const h = document.createElement("h5");
    h.textContent = title;
    const pre = document.createElement("pre");
    pre.textContent =
      typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    wrap.append(h, pre);
    body.appendChild(wrap);
  };

  section("Decision", debug.decision);
  section("Resolved query", debug.resolved_query);
  section("Context notes", debug.context_notes);
  section("Intent", debug.intent);
  section("Retrieval candidates", debug.retrieval && debug.retrieval.candidates);
  section(
    "Insufficiency",
    debug.retrieval && (debug.retrieval.insufficient || debug.retrieval.insufficiency_reason)
      ? {
          insufficient: debug.retrieval.insufficient,
          reason: debug.retrieval.insufficiency_reason,
        }
      : null
  );
  section("Conflicts", debug.conflicts);
  section("Tool calls", debug.tool_calls);
  section("Evidence coverage", debug.evidence_coverage);
  section("Fallbacks", debug.fallbacks);
  section("Errors", debug.errors);
  section("LLM phrasing used", debug.used_llm);

  details.appendChild(body);
  parent.appendChild(details);
}

function addAgentMessage(payload) {
  const tpl = $("#tpl-agent-msg");
  const node = tpl.content.firstElementChild.cloneNode(true);
  const bubble = node.querySelector(".bubble-agent");
  const extras = node.querySelector(".msg-extras");

  renderRichText(bubble, payload.answer);

  if (payload.conflict_detected) {
    addBanner(
      extras,
      "banner-conflict",
      "\u26A0",
      "Conflicting information found",
      "The supplied Aster & Row sources contain conflicting information. " +
        "Human confirmation is recommended."
    );
  }

  addOrderCard(extras, payload.order);
  addSources(extras, payload.sources);

  if (payload.handoff) {
    addBanner(
      extras,
      "banner-handoff",
      "\u{1F465}",
      "Human assistance recommended",
      "I don't want to guess or give you incorrect information. " +
        "A support specialist should confirm this." +
        (payload.reason ? " (" + payload.reason + ")" : "")
    );
  }

  addDevTrace(extras, payload.debug);
  chatLog.appendChild(node);
  scrollToBottom();
}

function addErrorBubble(message) {
  const tpl = $("#tpl-agent-msg");
  const node = tpl.content.firstElementChild.cloneNode(true);
  const bubble = node.querySelector(".bubble-agent");
  const extras = node.querySelector(".msg-extras");
  renderRichText(
    bubble,
    "Something went wrong.\n\nI couldn't complete that request safely. Please try again."
  );
  addBanner(extras, "banner-error", "\u26A0", "Request failed", message);
  chatLog.appendChild(node);
  scrollToBottom();
}

/* ---------------- loading state ---------------- */

function showTyping() {
  hideWelcome();
  typingRow = document.createElement("article");
  typingRow.className = "msg msg-agent";
  typingRow.setAttribute("aria-label", "Agent is responding");
  const avatar = document.createElement("div");
  avatar.className = "agent-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "A";
  const bubble = document.createElement("div");
  bubble.className = "typing-bubble";
  bubble.innerHTML =
    'Aster &amp; Row AI is checking the information<span class="dots">' +
    "<i></i><i></i><i></i></span>";
  typingRow.append(avatar, bubble);
  chatLog.appendChild(typingRow);
  scrollToBottom();
}

function removeTyping() {
  if (typingRow) {
    typingRow.remove();
    typingRow = null;
  }
}

/* ---------------- API ---------------- */

async function apiPost(path, body, timeoutMs = 60000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new Error("Server responded with status " + res.status);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function sendMessage(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  addUserMessage(text.trim());
  input.value = "";
  autoResize();
  showTyping();
  try {
    const payload = await apiPost("/api/chat", {
      message: text,
      session_id: sessionId,
      debug: devMode,
    });
    removeTyping();
    if (!payload || typeof payload.answer !== "string") {
      addErrorBubble("The server returned a malformed response.");
      return;
    }
    sessionId = payload.session_id || sessionId;
    updateRuntime();
    addAgentMessage(payload);
  } catch (err) {
    removeTyping();
    const detail =
      err.name === "AbortError"
        ? "The request timed out. Please try again."
        : err.message === "Failed to fetch"
          ? "Could not reach the support service. Check your connection and try again."
          : err.message || "Unexpected error.";
    addErrorBubble(detail);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

/* ---------------- session reset ---------------- */

async function newChat() {
  try {
    await apiPost("/api/session/reset", { session_id: sessionId }, 10000);
  } catch (_) {
    /* reset is best-effort; local state is cleared regardless */
  }
  sessionId = "web-" + Math.random().toString(36).slice(2, 10);
  chatLog.querySelectorAll(".msg").forEach((el) => el.remove());
  restoreWelcome();
  updateRuntime();
  input.focus();
}

function restoreWelcome() {
  if ($("#welcome")) return;
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.id = "welcome";
  welcome.innerHTML =
    '<div class="welcome-mark" aria-hidden="true">A</div>' +
    "<h1>How can we help?</h1>" +
    "<p class=\"welcome-sub\">Ask about returns, shipping, products, or an order.<br/>" +
    "Answers come from Aster&nbsp;&amp;&nbsp;Row's own documentation — with sources.</p>" +
    '<div class="suggestions" role="list" aria-label="Suggested questions"></div>';
  const suggestions = welcome.querySelector(".suggestions");
  for (const q of [
    "What is your return policy?",
    "Do you ship to Canada?",
    "Where is ORD-1007?",
    "Can I return a damaged final-sale item?",
  ]) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion";
    btn.setAttribute("role", "listitem");
    btn.textContent = q;
    btn.addEventListener("click", () => sendMessage(q));
    suggestions.appendChild(btn);
  }
  chatLog.prepend(welcome);
}

/* ---------------- developer panel ---------------- */

function updateRuntime() {
  const target = $("#dev-runtime");
  if (!target) return;
  target.innerHTML = "";
  const rows = [
    ["Session ID", sessionId],
    ["Profile", window.__health ? window.__health.profile : "…"],
    ["LLM phrasing", window.__health ? (window.__health.llm_phrasing_enabled ? "enabled" : "off (deterministic)") : "…"],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    row.append(dt, dd);
    target.appendChild(row);
  }
}

async function runEvaluation() {
  const out = $("#eval-output");
  out.hidden = false;
  out.innerHTML = "<em>Running the real evaluation suite…</em>";
  try {
    const res = await fetch("/api/eval");
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();
    const s = data.summary;
    let html =
      "<table><thead><tr><th>Category</th><th class='num'>Passed</th>" +
      "<th class='num'>Total</th><th class='num'>Score</th></tr></thead><tbody>";
    for (const [name, c] of Object.entries(s.categories)) {
      html +=
        "<tr><td>" + escapeHtml(name) + "</td><td class='num'>" + c.passed +
        "</td><td class='num'>" + c.total + "</td><td class='num'>" + c.pct +
        "%</td></tr>";
    }
    html += "</tbody></table>";
    html +=
      "<div class='eval-overall'>Overall: " + s.overall.passed + "/" +
      s.overall.total + " (" + s.overall.pct + "%)</div>";
    const failed = data.cases.filter((c) => !c.passed);
    if (failed.length) {
      html +=
        "<div class='eval-faildetails'>Failing cases: " +
        failed.map((c) => escapeHtml(c.id)).join(", ") + "</div>";
    }
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML =
      "<span style='color:var(--error-ink)'>Evaluation failed to run: " +
      escapeHtml(err.message || "unknown error") + "</span>";
  }
}

devToggle.addEventListener("click", () => {
  devMode = !devMode;
  devToggle.setAttribute("aria-pressed", String(devMode));
  devPanel.hidden = !devMode;
  updateRuntime();
});

$("#eval-run").addEventListener("click", runEvaluation);

/* ---------------- health / status ---------------- */

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    window.__health = await res.json();
    statusDot.className = "status-dot online";
    statusLabel.textContent = "Online";
  } catch (_) {
    statusDot.className = "status-dot offline";
    statusLabel.textContent = "Offline";
  }
  updateRuntime();
}

/* ---------------- composer wiring ---------------- */

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", autoResize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(input.value);
  }
});

newChatBtn.addEventListener("click", newChat);

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(btn.textContent));
});

checkHealth();
updateRuntime();
input.focus();

/* ---------------- demo autoplay (?demo=1&steps=N) ----------------
   Drives the REAL backend through the required demo flows so the README
   GIF can be captured from the actual application. Nothing is mocked:
   every turn goes through sendMessage() → POST /api/chat. */

const urlParams = new URLSearchParams(location.search);
const DEMO_MODE = urlParams.get("demo");
const DEMO_STEPS = parseInt(urlParams.get("steps") || "99", 10);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function openAllSources() {
  document.querySelectorAll("details.sources").forEach((d) => { d.open = true; });
}

async function playDemo() {
  window.__demoDone = false;
  const script = [
    "What is your return policy?",                    // A: KB + citation
    "Where is ORD-1007?",                             // B: real order tool
    "When will it arrive?",                           // C: multi-turn context
    "Do you offer a lifetime warranty on your bags?", // D: grounded safe answer
    "Can I put the entire Breeze Tumbler in the dishwasher?", // E: conflict
  ];
  let done = 0;
  for (const q of script) {
    if (done >= DEMO_STEPS) { window.__demoDone = true; return; }
    await sendMessage(q);
    await sleep(500);
    openAllSources();
    done += 1;
  }
  window.__demoDone = true;
}

if (DEMO_MODE === "eval") {
  window.addEventListener("load", async () => {
    window.__demoDone = false;
    devMode = true;
    devToggle.setAttribute("aria-pressed", "true");
    devPanel.hidden = false;
    updateRuntime();
    await sleep(300);
    await runEvaluation();
    window.__demoDone = true;
  });
} else if (DEMO_MODE) {
  window.addEventListener("load", () => {
    setTimeout(playDemo, 400);
  });
}
