const messages = document.querySelector("#messages");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const selector = document.querySelector("#agent-select");
const sendButton = document.querySelector("#send-button");
const attachButton = document.querySelector("#attach-button");
const fileInput = document.querySelector("#file-input");
const attachmentsElement = document.querySelector("#attachments");
const newChatButton = document.querySelector("#new-chat");
const logoutButton = document.querySelector("#logout");
const accountName = document.querySelector("#account-name");
const status = document.querySelector("#connection-status");
const sidebar = document.querySelector("#project-sidebar");
const sidebarToggle = document.querySelector("#sidebar-toggle");
const sidebarScrim = document.querySelector("#sidebar-scrim");
const projectStorageStatus = document.querySelector("#project-storage-status");
const projectListElement = document.querySelector("#project-list");
const generalChatButton = document.querySelector("#general-chat");
const generalNewChatButton = document.querySelector("#general-new-chat");
const newProjectButton = document.querySelector("#new-project");
const newProjectDialog = document.querySelector("#new-project-dialog");
const newProjectForm = document.querySelector("#new-project-form");
const newProjectName = document.querySelector("#new-project-name");
const newProjectDescription = document.querySelector("#new-project-description");
const newProjectError = document.querySelector("#new-project-error");
const cancelProjectButton = document.querySelector("#cancel-project");
const projectHeader = document.querySelector("#project-header");
const projectNameElement = document.querySelector("#project-name");
const projectDescriptionElement = document.querySelector("#project-description");
const projectNewChatButton = document.querySelector("#project-new-chat");
const projectHeaderActions = document.createElement("div");
projectHeaderActions.style.display = "flex";
projectHeaderActions.style.alignItems = "center";
projectHeaderActions.style.gap = "8px";
const deleteProjectButton = document.createElement("button");
deleteProjectButton.id = "delete-project";
deleteProjectButton.className = "secondary";
deleteProjectButton.type = "button";
deleteProjectButton.textContent = "Delete Project";
projectNewChatButton.replaceWith(projectHeaderActions);
projectHeaderActions.append(projectNewChatButton, deleteProjectButton);
const workspaceTabs = document.querySelector("#workspace-tabs");
const conversationContext = document.querySelector("#conversation-context");
const projectFileInput = document.querySelector("#project-file-input");
const fileSearch = document.querySelector("#file-search");
const projectFilesElement = document.querySelector("#project-files");
const projectMemoryElement = document.querySelector("#project-memory");
const projectActivityElement = document.querySelector("#project-activity");
let sessionId = null;
let continuationImageId = null;
let projects = [];
let currentProject = null;
let currentConversation = null;
let currentView = "chat";
let canUseProjects = false;
let composing = false;
let sending = false;
let uploading = false;
let attachments = [];
let selectedAssistantText = "";
let idleTimeoutMs = 15 * 60 * 1000;
let idleTimer;

const selectionCopyButton = document.createElement("button");
selectionCopyButton.className = "selection-copy";
selectionCopyButton.type = "button";
selectionCopyButton.textContent = "Copy selection";
selectionCopyButton.hidden = true;
document.body.append(selectionCopyButton);

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
  article.innerHTML = `<div class="message-header"><div class="message-label">${escapeHtml(label)}</div></div><div class="markdown">${markdown(content)}</div>`;
  if (activity) article.append(activityElement(activity));
  messages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  article.querySelectorAll(".copy-code").forEach((button) => button.addEventListener("click", (event) => copyText(decodeURIComponent(event.currentTarget.dataset.copy), event.currentTarget, "Copied")));
  return article;
}

function addGeneratedImages(article, images) {
  for (const image of images || []) {
    const link = document.createElement("a");
    link.className = "generated-image";
    link.href = image.data_url;
    link.download = image.filename;
    link.innerHTML = `<img src="${image.data_url}" alt="Generated image"><span>Download PNG</span>`;
    article.append(link);
  }
}

function addMessageAttachments(article, items) {
  if (!items.length) return;
  const container = document.createElement("div");
  container.className = "message-attachments";
  container.innerHTML = items.map((attachment) => (
    `<figure class="message-attachment${attachment.thumbnail_data_url ? " image" : " file"}">` +
    `${attachment.thumbnail_data_url ? `<img src="${escapeHtml(attachment.thumbnail_data_url)}" alt="Attached image ${escapeHtml(attachment.filename)}">` : ""}` +
    `<figcaption>${escapeHtml(attachment.filename)}${attachment.truncated ? " (trimmed)" : ""}</figcaption>` +
    `</figure>`
  )).join("");
  article.append(container);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
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

function hideSelectionCopyButton() {
  selectedAssistantText = "";
  selectionCopyButton.hidden = true;
}

function updateSelectionCopyButton() {
  const selection = document.getSelection();
  const text = selection?.toString().trim() || "";
  if (!selection?.rangeCount || !text) {
    hideSelectionCopyButton();
    return;
  }
  const node = selection.getRangeAt(0).commonAncestorContainer;
  const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
  if (!element?.closest(".message.assistant .markdown")) {
    hideSelectionCopyButton();
    return;
  }
  const rect = selection.getRangeAt(0).getBoundingClientRect();
  selectedAssistantText = text;
  selectionCopyButton.style.top = `${Math.max(8, rect.top - 38)}px`;
  selectionCopyButton.style.left = `${Math.min(window.innerWidth - 132, Math.max(8, rect.left))}px`;
  selectionCopyButton.hidden = false;
}

function activityElement(activity) {
  const details = document.createElement("details");
  const tools = activity.tools.map((tool) => toolActivity(tool)).join("") || "No tools called";
  const usage = activity.usage || {};
  const wholeUsage = activity.whole_request_usage || {};
  const finalCall = activity.final_call || {};
  const research = activity.research || {};
  const researchRounds = (research.rounds || []).map((round) => {
    const academic = round.academic_intelligence || {};
    const sourceStatus = Object.entries(academic.source_status || {})
      .map(([source, status]) => `${source}: ${status}`).join(", ");
    const publicationCandidates = Object.entries(academic.publication_candidates || {})
      .map(([source, count]) => `${source}: ${count}`).join(", ");
    const academicDetail = sourceStatus
      ? `<br>Academic sources: ${escapeHtml(sourceStatus)}<br>` +
        `Providers called: ${escapeHtml((academic.providers_called || []).join(", ") || "none")}; ` +
        `Publication candidates: ${escapeHtml(publicationCandidates || "none")}; ` +
        `identity sources: ${escapeHtml((academic.identity_sources || []).join(", ") || "none")}; ` +
        `identity confidence: ${escapeHtml(academic.identity_confidence || "UNKNOWN")}; ` +
        `coverage conflicts: ${escapeHtml(String(academic.coverage_conflicts ?? 0))}; ` +
        `merged corpus: ${escapeHtml(String(academic.merged_verified_corpus ?? 0))}; ` +
        `representative papers: ${escapeHtml(String(academic.representative_papers ?? 0))}`
      : "";
    return `<div>Round ${escapeHtml(String(round.round))}: ${escapeHtml((round.queries || []).join(" | "))}<br>` +
      `Tools: ${escapeHtml((round.tools || []).join(", ") || "none")}; sources: ${escapeHtml(String(round.sources_fetched ?? 0))}; ` +
      `entity: ${escapeHtml(round.entity_confidence || "UNKNOWN")}; gap: ${round.ready_to_answer ? "ready" : "follow-up"}${academicDetail}</div>`;
  }).join("") || "N/A";
  const llmCalls = (activity.llm_calls || []).map((call) => (
    `<div>#${escapeHtml(String(call.call_id))} ${escapeHtml(call.purpose)}: ${formatMs(call.total_llm_latency_ms)}${call.decode_tokens_per_second ? `, ${escapeHtml(String(call.decode_tokens_per_second))} tok/s decode` : ""}</div>`
  )).join("") || "No model calls recorded";
  const stages = (activity.stages || []).map((stage) => (
    `<div>${escapeHtml(stage.name)}: ${formatMs(stage.duration_ms)}</div>`
  )).join("") || "No timed stages";
  const rows = [
    ["Selected", activity.selected_agent], ["Route", activity.direct ? `Direct ${activity.routed_agent}` : `Main → ${activity.routed_agent}`],
    ["Summary", activity.route_summary], ["Time", `${(activity.duration_ms / 1000).toFixed(1)} sec`],
    ["End-to-end rate", activity.end_to_end_tokens_per_second ? `${activity.end_to_end_tokens_per_second} tok/s` : "N/A"],
    ["Final synthesis input", usage.prompt_tokens ?? "N/A"], ["Final synthesis output", usage.completion_tokens ?? "N/A"],
    ["Final synthesis TTFT", formatMs(finalCall.ttft_ms)], ["Final synthesis decode", formatMs(finalCall.generation_time_ms)],
    ["Final synthesis decode speed", finalCall.decode_tokens_per_second ? `${finalCall.decode_tokens_per_second} tok/s` : "N/A"],
    ["Research mode", research.mode || "N/A"], ["Research state", research.state || "N/A"],
    ["State history", (research.state_history || []).join(" → ") || "N/A"],
    ["Research rounds", activity.research_rounds || "N/A"], ["Round detail", researchRounds],
    ["Entity confidence", research.entity_confidence || "N/A"], ["Gap status", research.gap_status || "N/A"],
    ["Final synthesis executed", research.final_synthesis_executed ? "YES" : "NO"],
    ["Termination", research.termination_reason || "N/A"],
    ["LLM calls", wholeUsage.llm_call_count ?? 0], ["Whole LLM input", wholeUsage.input_tokens ?? 0],
    ["Whole LLM output", wholeUsage.output_tokens ?? 0], ["Stages", stages], ["LLM timing", llmCalls], ["Tools", tools],
  ];
  details.innerHTML = `<summary>Agent Activity</summary><div class="activity-grid">${rows.map(([key, value]) => `<strong>${escapeHtml(key)}</strong><span>${value}</span>`).join("")}</div>`;
  return details;
}

function formatMs(value) {
  return typeof value === "number" ? `${(value / 1000).toFixed(1)} sec` : "N/A";
}

function toolActivity(tool) {
  const details = tool.details || {};
  const reason = details.failure_reason ? `: ${escapeHtml(details.failure_reason)}` : "";
  const requests = (details.requests || []).map((request) => (
    `<div>${request.success ? "✓" : "!"} ${escapeHtml(request.path || request.query || request.operation || "request")} (${formatMs(request.duration_ms)})${request.attempt ? ` attempt ${escapeHtml(String(request.attempt))}` : ""}${request.failure_reason ? `: ${escapeHtml(request.failure_reason)}` : ""}</div>`
  )).join("");
  const fetches = (details.fetches || []).map((fetch) => (
    `<div>${fetch.success ? "✓" : "!"} ${escapeHtml(fetch.url)} (${formatMs(fetch.total_fetch_time_ms)}, ${fetch.text_length ?? 0} chars${fetch.connect_time_ms === null ? ", connect N/A" : ""})${fetch.failure_reason ? `: ${escapeHtml(fetch.failure_reason)}` : ""}</div>`
  )).join("");
  return `<div class="${tool.success ? "tool-ok" : "tool-failed"}">${tool.success ? "✓" : "!"} ${escapeHtml(tool.name)} (${formatMs(tool.duration_ms)})${reason}${requests}${fetches}</div>`;
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (response.status === 401) {
    window.location.replace("/login");
    throw new Error("login required");
  }
  if (!response.ok) {
    const error = new Error(payload.detail || "Request failed");
    error.status = response.status;
    throw error;
  }
  return payload;
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarScrim.hidden = true;
}

function projectRequestBody(value) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) };
}

function renderProjectNavigation() {
  generalChatButton.classList.toggle("active", !currentProject);
  projectListElement.innerHTML = projects.map((project) => {
    const active = currentProject?.id === project.id;
    const conversations = active ? (currentProject.conversations || []).map((conversation) => (
      `<button class="nav-item${currentConversation?.id === conversation.id ? " active" : ""}" type="button" data-conversation-id="${escapeHtml(conversation.id)}">${escapeHtml(conversation.title)}</button>`
    )).join("") : "";
    return `<div><button class="nav-item project-nav${active ? " active" : ""}" type="button" data-project-id="${escapeHtml(project.id)}">${active ? "▾" : "▸"} ${escapeHtml(project.name)}</button>${active ? `<div class="conversation-list">${conversations}</div>` : ""}</div>`;
  }).join("") || `<p class="empty-state">No projects yet.</p>`;
  projectListElement.querySelectorAll("[data-project-id]").forEach((button) => button.addEventListener("click", () => openProject(button.dataset.projectId)));
  projectListElement.querySelectorAll("[data-conversation-id]").forEach((button) => button.addEventListener("click", () => openConversation(button.dataset.conversationId)));
}

async function loadProjects() {
  if (!canUseProjects) return;
  const payload = await request("/api/projects");
  projects = payload.projects;
  const storage = payload.storage;
  projectStorageStatus.textContent = storage.online
    ? `ONLINE · ${formatBytes(storage.total_bytes)} total · ${formatBytes(storage.free_bytes)} free`
    : "STORAGE OFFLINE";
  projectStorageStatus.className = `storage-state ${storage.online ? "online" : "offline"}`;
  newProjectButton.disabled = !storage.online;
  renderProjectNavigation();
}

async function openProject(projectId, conversationId = null) {
  currentProject = await request(`/api/projects/${encodeURIComponent(projectId)}`);
  projects = projects.map((project) => project.id === currentProject.id ? currentProject : project);
  projectHeader.hidden = false;
  workspaceTabs.hidden = false;
  projectNameElement.textContent = currentProject.name;
  projectDescriptionElement.textContent = currentProject.description || "No description";
  let conversation = currentProject.conversations.find((item) => item.id === conversationId) || currentProject.conversations[0];
  if (!conversation) {
    conversation = await request(`/api/projects/${encodeURIComponent(projectId)}/conversations`, projectRequestBody({ title: "Overall research" }));
    currentProject.conversations.unshift(conversation);
  }
  currentConversation = conversation;
  continuationImageId = null;
  sessionId = null;
  renderProjectNavigation();
  await openConversation(conversation.id);
  setView("chat");
  closeSidebar();
}

async function openConversation(conversationId) {
  if (!currentProject) return;
  currentConversation = currentProject.conversations.find((item) => item.id === conversationId);
  if (!currentConversation) return;
  const payload = await request(`/api/projects/${encodeURIComponent(currentProject.id)}/conversations/${encodeURIComponent(conversationId)}/messages`);
  messages.innerHTML = "";
  for (const message of payload.messages) {
    addMessage(message.role, message.content, message.role === "user" ? "You" : "main");
  }
  if (!payload.messages.length) addMessage("assistant", "This conversation shares the project's files, summary, and durable memory.", "Project Workspace");
  conversationContext.hidden = false;
  conversationContext.textContent = `Project: ${currentProject.name} · Conversation: ${currentConversation.title}`;
  sessionId = null;
  continuationImageId = null;
  renderProjectNavigation();
  closeSidebar();
}

async function createProjectConversation() {
  if (!currentProject) return;
  const conversation = await request(
    `/api/projects/${encodeURIComponent(currentProject.id)}/conversations`,
    projectRequestBody({ title: `Conversation ${currentProject.conversations.length + 1}` }),
  );
  currentProject.conversations.unshift(conversation);
  await openConversation(conversation.id);
  setView("chat");
}

async function deleteCurrentProject() {
  if (!currentProject) return;
  const projectId = currentProject.id;
  const projectName = currentProject.name;
  const confirmation = window.prompt(
    `Delete "${projectName}" permanently? Type the project name to confirm.`,
  );
  if (confirmation !== projectName) return;
  deleteProjectButton.disabled = true;
  try {
    await request(`/api/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" });
    projects = projects.filter((project) => project.id !== projectId);
    await openGeneralChat();
    await loadProjects();
  } catch (error) {
    window.alert(`Could not delete project: ${error.message}`);
  } finally {
    deleteProjectButton.disabled = false;
  }
}

async function openGeneralChat() {
  currentProject = null;
  currentConversation = null;
  projectHeader.hidden = true;
  workspaceTabs.hidden = true;
  conversationContext.hidden = true;
  renderProjectNavigation();
  setView("chat");
  await newSession();
  closeSidebar();
}

function setView(view) {
  currentView = view;
  document.querySelectorAll(".workspace-view").forEach((element) => { element.hidden = element.id !== `${view}-view`; });
  workspaceTabs.querySelectorAll("button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  if (view === "files") loadProjectFiles();
  if (view === "memory") loadProjectMemory();
  if (view === "activity") loadProjectActivity();
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / (1024 ** index)).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function formatDate(value) {
  return new Date(value).toLocaleString();
}

async function loadProjectFiles() {
  if (!currentProject) return;
  const payload = await request(`/api/projects/${encodeURIComponent(currentProject.id)}/files`);
  const query = fileSearch.value.trim().toLowerCase();
  const files = payload.files.filter((file) => !query || file.original_name.toLowerCase().includes(query));
  projectFilesElement.innerHTML = files.map((file) => (
    `<div class="data-row" data-file-id="${escapeHtml(file.id)}"><div class="data-row-main"><strong>${escapeHtml(file.original_name)}</strong><span>${escapeHtml(file.mime_type)} · ${formatBytes(file.size)} · ${escapeHtml(file.index_status)}${file.artifact_id ? " · artifact" : ""}</span></div><span class="data-meta">${escapeHtml(formatDate(file.created_at))}</span><div class="data-actions"><a href="/api/projects/${encodeURIComponent(currentProject.id)}/files/${encodeURIComponent(file.id)}">Download</a><button class="danger" type="button" data-delete-file>Delete</button></div></div>`
  )).join("") || `<p class="empty-state">No matching files.</p>`;
  projectFilesElement.querySelectorAll("[data-delete-file]").forEach((button) => button.addEventListener("click", async () => {
    const row = button.closest("[data-file-id]");
    if (button.dataset.confirm !== "true") {
      button.dataset.confirm = "true";
      button.textContent = "Confirm delete";
      return;
    }
    await request(`/api/projects/${encodeURIComponent(currentProject.id)}/files/${encodeURIComponent(row.dataset.fileId)}`, { method: "DELETE" });
    await loadProjectFiles();
  }));
}

async function uploadProjectFiles(files) {
  if (!currentProject || !currentConversation) return;
  for (const file of files) {
    const body = new FormData();
    body.append("file", file);
    body.append("project_id", currentProject.id);
    body.append("conversation_id", currentConversation.id);
    const uploaded = await request("/api/upload", { method: "POST", body });
    await request(`/api/upload/${encodeURIComponent(uploaded.attachment_id)}`, { method: "DELETE" });
  }
  projectFileInput.value = "";
  await loadProjectFiles();
}

async function loadProjectMemory() {
  if (!currentProject) return;
  const payload = await request(`/api/projects/${encodeURIComponent(currentProject.id)}/memories`);
  const groups = Object.groupBy ? Object.groupBy(payload.memories, (memory) => memory.type) : payload.memories.reduce((result, memory) => { (result[memory.type] ||= []).push(memory); return result; }, {});
  const groupHtml = Object.entries(groups).map(([type, items]) => `<section class="memory-group"><h3>${escapeHtml(type.toUpperCase())}</h3>${items.map((memory) => (
    `<div class="data-row memory-row${memory.active ? "" : " inactive"}" data-memory-id="${escapeHtml(memory.id)}"><div class="data-row-main"><strong>${escapeHtml(memory.content)}</strong><span>${escapeHtml(memory.confidence)} · updated ${escapeHtml(formatDate(memory.updated_at))}${memory.active ? "" : " · superseded/inactive"}</span></div><div class="data-actions"><button type="button" data-edit-memory>Edit</button><button class="danger" type="button" data-delete-memory>Delete</button></div></div>`
  )).join("")}</section>`).join("");
  projectMemoryElement.innerHTML = `<section class="memory-summary"><h3>PROJECT SUMMARY</h3><p>${escapeHtml(payload.summary || "No summary yet.")}</p></section>${groupHtml || `<p class="empty-state">No durable memories yet.</p>`}`;
  projectMemoryElement.querySelectorAll("[data-edit-memory]").forEach((button) => button.addEventListener("click", async () => {
    const row = button.closest("[data-memory-id]");
    const contentElement = row.querySelector("strong");
    if (button.dataset.editing !== "true") {
      button.dataset.editing = "true";
      button.textContent = "Save";
      contentElement.contentEditable = "true";
      contentElement.focus();
      return;
    }
    const content = contentElement.textContent.trim();
    if (!content) return;
    await request(`/api/projects/${encodeURIComponent(currentProject.id)}/memories/${encodeURIComponent(row.dataset.memoryId)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content, active: true }) });
    await loadProjectMemory();
  }));
  projectMemoryElement.querySelectorAll("[data-delete-memory]").forEach((button) => button.addEventListener("click", async () => {
    const row = button.closest("[data-memory-id]");
    if (button.dataset.confirm !== "true") {
      button.dataset.confirm = "true";
      button.textContent = "Confirm delete";
      return;
    }
    await request(`/api/projects/${encodeURIComponent(currentProject.id)}/memories/${encodeURIComponent(row.dataset.memoryId)}`, { method: "DELETE" });
    await loadProjectMemory();
  }));
}

async function loadProjectActivity() {
  if (!currentProject) return;
  const payload = await request(`/api/projects/${encodeURIComponent(currentProject.id)}/activity`);
  projectActivityElement.innerHTML = payload.events.map((event) => (
    `<div class="data-row"><div class="data-row-main"><strong>${escapeHtml(event.event_type.replaceAll("_", " "))}</strong><span>${escapeHtml(event.actor)}</span></div><span class="data-meta">${escapeHtml(formatDate(event.created_at))}</span><span></span></div>`
  )).join("") || `<p class="empty-state">No activity yet.</p>`;
}

function renderAttachments() {
  attachmentsElement.innerHTML = attachments.map((attachment) => (
    `<span class="attachment-chip" data-id="${escapeHtml(attachment.attachment_id)}">` +
    `${attachment.thumbnail_data_url ? `<img src="${escapeHtml(attachment.thumbnail_data_url)}" alt="Preview of ${escapeHtml(attachment.filename)}">` : ""}` +
    `<span>${escapeHtml(attachment.filename)}${attachment.truncated ? " (trimmed)" : ""}</span>` +
    `<button type="button" title="Remove attachment" aria-label="Remove ${escapeHtml(attachment.filename)}">×</button></span>`
  )).join("");
  attachmentsElement.hidden = attachments.length === 0;
  attachmentsElement.querySelectorAll("button").forEach((button) => button.addEventListener("click", async () => {
    const chip = button.closest(".attachment-chip");
    const attachmentId = chip.dataset.id;
    await request(`/api/upload/${encodeURIComponent(attachmentId)}`, { method: "DELETE" });
    attachments = attachments.filter((attachment) => attachment.attachment_id !== attachmentId);
    renderAttachments();
  }));
}

async function uploadFiles(files) {
  const availableSlots = 3 - attachments.length;
  if (files.length > availableSlots) {
    addMessage("assistant", "You can attach up to three files per message.", "Upload");
    return;
  }
  const videoExtensions = ["mp4", "mov", "webm", "mkv", "avi"];
  const imageExtensions = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
  const oversized = files.find((file) => {
    const extension = file.name.split(".").pop().toLowerCase();
    const limit = videoExtensions.includes(extension) ? 100 : imageExtensions.includes(extension) ? 20 : 10;
    return file.size > limit * 1024 * 1024;
  });
  if (oversized) {
    addMessage("assistant", `${oversized.name} exceeds the upload limit for its file type.`, "Upload");
    return;
  }
  uploading = true;
  attachButton.disabled = true;
  sendButton.disabled = true;
  try {
    for (const file of files) {
      const body = new FormData();
      body.append("file", file);
      if (currentProject && currentConversation) {
        body.append("project_id", currentProject.id);
        body.append("conversation_id", currentConversation.id);
      }
      const attachment = await request("/api/upload", { method: "POST", body });
      attachments.push(attachment);
      renderAttachments();
    }
  } catch (error) {
    addMessage("assistant", `Upload failed: ${error.message}`, "Upload");
  } finally {
    fileInput.value = "";
    attachButton.disabled = false;
    sendButton.disabled = false;
    uploading = false;
    input.focus();
  }
}

async function logout() {
  clearTimeout(idleTimer);
  await fetch("/api/logout", { method: "POST" });
  window.location.replace("/login");
}

function resetIdleTimer() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(logout, idleTimeoutMs);
}

async function newSession() {
  if (currentProject) {
    await createProjectConversation();
    return;
  }
  const payload = await request("/api/new-session", { method: "POST" });
  sessionId = payload.session_id;
  const discardedAttachmentIds = attachments.map((attachment) => attachment.attachment_id);
  if (continuationImageId) discardedAttachmentIds.push(continuationImageId);
  await Promise.all(discardedAttachmentIds.map((attachmentId) => request(`/api/upload/${encodeURIComponent(attachmentId)}`, { method: "DELETE" })));
  attachments = [];
  continuationImageId = null;
  renderAttachments();
  messages.innerHTML = "";
  addMessage("assistant", "New short-term session created. Long-term memory is not changed.", "Main / Secretary");
  input.focus();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (sending || uploading) return;
  const typedMessage = input.value.trim();
  if (!typedMessage && attachments.length === 0) return;
  const message = typedMessage || "Analyze the attached document(s).";
  const submittedAttachments = [...attachments];
  const submittedAttachmentIds = new Set(submittedAttachments.map((attachment) => attachment.attachment_id));
  attachments = attachments.filter((attachment) => !submittedAttachmentIds.has(attachment.attachment_id));
  renderAttachments();
  sending = true;
  const userArticle = addMessage("user", message, "You");
  addMessageAttachments(userArticle, submittedAttachments);
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  try {
    const payload = await request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, selected_agent: selector.value, session_id: sessionId, attachment_ids: submittedAttachments.map((attachment) => attachment.attachment_id), continuation_image_id: continuationImageId, project_id: currentProject?.id || null, conversation_id: currentConversation?.id || null }) });
    sessionId = payload.session_id;
    if (payload.continuation_image_id) continuationImageId = payload.continuation_image_id;
    const article = addMessage("assistant", payload.content, payload.activity.routed_agent, payload.activity);
    addGeneratedImages(article, payload.generated_images);
  } catch (error) {
    if (error.status === 404 && (submittedAttachments.length || continuationImageId)) {
      continuationImageId = null;
      addMessage("assistant", "The attachment is no longer available. Please attach the file again.", "Runtime");
    } else {
      attachments = [...submittedAttachments, ...attachments];
      renderAttachments();
      addMessage("assistant", `Request failed: ${error.message}`, "Runtime");
    }
  } finally {
    input.value = "";
    input.disabled = false;
    sendButton.disabled = false;
    sending = false;
    input.focus();
  }
});

input.addEventListener("compositionstart", () => {
  composing = true;
});

input.addEventListener("compositionend", () => {
  composing = false;
});

input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  if (composing || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  form.requestSubmit();
});
newChatButton.addEventListener("click", newSession);
generalNewChatButton.addEventListener("click", openGeneralChat);
generalChatButton.addEventListener("click", openGeneralChat);
projectNewChatButton.addEventListener("click", createProjectConversation);
deleteProjectButton.addEventListener("click", deleteCurrentProject);
workspaceTabs.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
sidebarToggle.addEventListener("click", () => { sidebar.classList.add("open"); sidebarScrim.hidden = false; });
sidebarScrim.addEventListener("click", closeSidebar);
newProjectButton.addEventListener("click", () => { newProjectError.textContent = ""; newProjectDialog.showModal(); newProjectName.focus(); });
cancelProjectButton.addEventListener("click", () => newProjectDialog.close());
newProjectForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const project = await request("/api/projects", projectRequestBody({ name: newProjectName.value.trim(), description: newProjectDescription.value.trim() }));
    projects.unshift(project);
    newProjectDialog.close();
    newProjectForm.reset();
    await openProject(project.id);
  } catch (error) {
    newProjectError.textContent = error.message;
  }
});
projectFileInput.addEventListener("change", () => uploadProjectFiles([...projectFileInput.files]).catch((error) => { projectFilesElement.innerHTML = `<p class="empty-state">Upload failed: ${escapeHtml(error.message)}</p>`; }));
fileSearch.addEventListener("input", loadProjectFiles);
attachButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => uploadFiles([...fileInput.files]));
logoutButton.addEventListener("click", logout);
for (const eventName of ["pointerdown", "keydown", "input", "scroll", "touchstart"]) {
  document.addEventListener(eventName, resetIdleTimer, { passive: true });
}
document.addEventListener("selectionchange", updateSelectionCopyButton);
document.addEventListener("scroll", hideSelectionCopyButton, true);
selectionCopyButton.addEventListener("mousedown", (event) => event.preventDefault());
selectionCopyButton.addEventListener("click", () => copyText(selectedAssistantText, selectionCopyButton, "Copied"));

async function initialize() {
  try {
    const [agentPayload, health, account] = await Promise.all([request("/api/agents"), request("/health"), request("/api/me")]);
    selector.innerHTML = agentPayload.agents.map((agent) => `<option value="${agent.id}">${agent.label}</option>`).join("");
    accountName.textContent = `${account.username} (${account.role})`;
    attachButton.hidden = !account.can_upload;
    fileInput.disabled = !account.can_upload;
    canUseProjects = account.can_use_projects;
    newProjectButton.hidden = !canUseProjects;
    if (canUseProjects) await loadProjects();
    else {
      projectStorageStatus.textContent = "Unavailable for this account";
      projectStorageStatus.className = "storage-state offline";
    }
    idleTimeoutMs = account.session_idle_timeout_seconds * 1000;
    resetIdleTimer();
    status.textContent = "Connected";
    status.className = "status online";
    await newSession();
  } catch (error) { status.textContent = "Backend unavailable"; status.className = "status offline"; }
}
renderAttachments();
resetIdleTimer();
initialize();