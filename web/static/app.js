const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const selector = document.querySelector("#agent-select");
const sendButton = document.querySelector("#send-button");
const newChatButton = document.querySelector("#new-chat");
const status = document.querySelector("#connection-status");
let sessionId = null;
let composing = false;
let compositionJustEnded = false;

function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function inlineMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function markdown(value) {
  const codeBlocks = [];
  let text = escapeHtml(value).replace(/```([\s\S]*?)```/g, (_, code) => {
    const trimmed = code.trim();
    const id = codeBlocks.push(`<div class="code-block"><button class="copy-code" type="button" data-copy="${encodeURIComponent(trimmed)}">Copy code</button><pre><code>${trimmed}</code></pre></div>`) - 1;
    return `@@CODE${id}@@`;
  });
  const blocks = [];
  for (const rawBlock of text.split(/\n{2,}/)) {
    const block = rawBlock.trim();
    if (!block) continue;
    if (/^@@CODE\d+@@$/.test(block)) { blocks.push(block); continue; }
    if (/^---+$/.test(block)) { blocks.push("<hr>"); continue; }
    const heading = block.match(/^(#{1,3})\s+(.+)$/);
    if (heading) { blocks.push(`<h${heading[1].length}>${inlineMarkdown(heading[2])}</h${heading[1].length}>`); continue; }
    if (block.split("\n").every((line) => /^>\s?/.test(line))) {
      blocks.push(`<blockquote>${inlineMarkdown(block.replace(/^>\s?/gm, "").replaceAll("\n", "<br>"))}</blockquote>`);
      continue;
    }
    const lines = block.split("\n");
    if (lines.every((line) => /^[-*]\s+/.test(line))) {
      blocks.push(`<ul>${lines.map((line) => `<li>${inlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`).join("")}</ul>`);
      continue;
    }
    if (lines.every((line) => /^\d+\.\s+/.test(line))) {
      blocks.push(`<ol>${lines.map((line) => `<li>${inlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`).join("")}</ol>`);
      continue;
    }
    blocks.push(`<p>${inlineMarkdown(block.replaceAll("\n", "<br>"))}</p>`);
  }
  text = blocks.join("");
  return text.replace(/@@CODE(\d+)@@/g, (_, id) => codeBlocks[Number(id)]);
}

function addMessage(role, content, label, activity) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const copyButton = role === "assistant" ? `<button class="copy-response" type="button" aria-label="Copy response">Copy</button>` : "";
  article.innerHTML = `<div class="message-header"><div class="message-label">${escapeHtml(label)}</div>${copyButton}</div><div class="markdown">${markdown(content)}</div>`;
  if (activity) article.append(activityElement(activity));
  messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  article.querySelector(".copy-response")?.addEventListener("click", (event) => copyText(content, event.currentTarget, "Copied"));
  article.querySelectorAll(".copy-code").forEach((button) => button.addEventListener("click", (event) => copyText(decodeURIComponent(event.currentTarget.dataset.copy), event.currentTarget, "Copied")));
}

async function copyText(value, button, successLabel) {
  const original = button.textContent;
  try {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(value);
      } catch (error) {
        legacyCopy(value);
      }
    } else {
      legacyCopy(value);
    }
    button.textContent = successLabel;
  } catch (error) {
    button.textContent = "Copy failed";
  }
  setTimeout(() => { button.textContent = original; }, 1600);
}

function legacyCopy(value) {
  const temporaryInput = document.createElement("textarea");
  temporaryInput.value = value;
  temporaryInput.setAttribute("readonly", "");
  temporaryInput.style.position = "fixed";
  temporaryInput.style.opacity = "0";
  document.body.append(temporaryInput);
  temporaryInput.select();
  const copied = document.execCommand("copy");
  temporaryInput.remove();
  if (!copied) throw new Error("clipboard copy was rejected");
}

function activityElement(activity) {
  const details = document.createElement("details");
  const tools = activity.tools.map((tool) => `<div class="${tool.success ? "tool-ok" : "tool-failed"}">${tool.success ? "✓" : "!"} ${escapeHtml(tool.name)} (${tool.duration_ms} ms)</div>`).join("") || "No tools called";
  const usage = activity.usage || {};
  const rows = [
    ["Selected", activity.selected_agent], ["Route", activity.direct ? `Direct ${activity.routed_agent}` : `Main → ${activity.routed_agent}`],
    ["Summary", activity.route_summary], ["Time", `${(activity.duration_ms / 1000).toFixed(1)} sec`],
    ["Input tokens", usage.prompt_tokens ?? "N/A"], ["Output tokens", usage.completion_tokens ?? "N/A"],
    ["Speed", activity.tokens_per_second ? `${activity.tokens_per_second} tok/s` : "N/A"], ["Tools", tools],
  ];
  details.innerHTML = `<summary>Agent Activity</summary><div class="activity-grid">${rows.map(([key, value]) => `<strong>${escapeHtml(key)}</strong><span>${value}</span>`).join("")}</div>`;
  return details;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

async function newSession() {
  const payload = await request("/api/new-session", { method: "POST" });
  sessionId = payload.session_id;
  messages.innerHTML = "";
  addMessage("assistant", "New short-term session created. Long-term memory is not changed.", "Main / Secretary");
  input.focus();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  addMessage("user", message, "You");
  input.value = "";
  sendButton.disabled = true;
  try {
    const payload = await request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, selected_agent: selector.value, session_id: sessionId }) });
    sessionId = payload.session_id;
    addMessage("assistant", payload.content, payload.activity.routed_agent, payload.activity);
  } catch (error) {
    addMessage("assistant", `Request failed: ${error.message}`, "Runtime");
  } finally { sendButton.disabled = false; input.focus(); }
});

input.addEventListener("compositionstart", () => {
  composing = true;
});

input.addEventListener("compositionend", () => {
  composing = false;
  compositionJustEnded = true;
  setTimeout(() => { compositionJustEnded = false; }, 0);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !composing && !compositionJustEnded && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    form.requestSubmit();
  }
});
newChatButton.addEventListener("click", newSession);

async function initialize() {
  try {
    const [agentPayload, health] = await Promise.all([request("/api/agents"), request("/health")]);
    selector.innerHTML = agentPayload.agents.map((agent) => `<option value="${agent.id}">${agent.label}</option>`).join("");
    status.textContent = `Connected: ${health.model}`;
    status.className = "status online";
    await newSession();
  } catch (error) { status.textContent = "Backend unavailable"; status.className = "status offline"; }
}
initialize();