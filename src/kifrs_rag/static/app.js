const form = document.querySelector("#query-form");
const question = document.querySelector("#question");
const button = document.querySelector("#submit-button");
const resultPanel = document.querySelector("#result");
const answerText = document.querySelector("#answer-text");
const answerStatus = document.querySelector("#answer-status");
const systemStatus = document.querySelector("#system-status");
const citations = document.querySelector("#citations");
const trace = document.querySelector("#trace-id");
const modeChip = document.querySelector("#mode-chip");
const chunkCount = document.querySelector("#chunk-count");
const vectorSize = document.querySelector("#vector-size");
const threshold = document.querySelector("#threshold");
const modeDescription = document.querySelector("#mode-description");

fetch("/v1/meta")
  .then((response) => response.json())
  .then((meta) => {
    modeChip.textContent = meta.mode === "demo" ? `Demo · ${meta.standards} Topics` : `${meta.standards} Standards`;
    chunkCount.textContent = Number(meta.chunks).toLocaleString();
    vectorSize.textContent = meta.vector;
    threshold.textContent = meta.threshold;
    modeDescription.textContent = meta.mode === "demo"
      ? "합성 문단으로 검색·재순위화·인용 검증을 체험하는 공개 데모입니다."
      : "근거가 충분하지 않으면 답변을 생성하지 않습니다.";
  })
  .catch(() => {});

document.querySelectorAll(".example-question").forEach((example) => {
  example.addEventListener("click", () => {
    question.value = example.textContent;
    question.focus();
  });
});

function addCitation(item) {
  const card = document.createElement("article");
  card.className = "citation";
  const head = document.createElement("div");
  head.className = "citation-head";
  const identity = document.createElement("span");
  identity.textContent = `${item.standard_id} · 문단 ${item.paragraph_id}`;
  const score = document.createElement("span");
  score.className = "citation-score";
  score.textContent = Number(item.score).toFixed(3);
  const quote = document.createElement("blockquote");
  quote.textContent = item.quote;
  head.append(identity, score);
  card.append(head, quote);
  citations.append(card);
}

function showMessage(status, message, payload = {}) {
  resultPanel.hidden = false;
  answerStatus.textContent = status;
  answerText.textContent = message;
  trace.textContent = payload.trace_id ? `Trace · ${payload.trace_id}` : "";
  citations.replaceChildren();
  (payload.citations || []).forEach(addCitation);
  if (!payload.citations?.length) {
    const note = document.createElement("article");
    note.className = "citation error";
    note.textContent = "표시할 수 있는 검증된 근거가 없습니다.";
    citations.append(note);
  }
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.textContent = "검색 중…";
  systemStatus.textContent = "근거 검색 중";
  try {
    const response = await fetch("/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.value.trim() }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "요청을 처리하지 못했습니다.");
    if (payload.status === "answered") {
      showMessage("인용 검증 완료", payload.answer, payload);
    } else if (payload.status === "insufficient_evidence") {
      showMessage("근거 부족", "질문에 답할 수 있는 충분한 기준서 근거를 찾지 못했습니다.", payload);
    } else if (payload.status === "clarification_required") {
      showMessage("추가 정보 필요", payload.answer, payload);
    } else if (payload.status === "budget_exceeded") {
      showMessage("API 예산 한도", "설정된 OpenAI API 사용 한도에 도달했습니다.", payload);
    } else {
      showMessage("검증 실패", "생성된 답변의 근거를 검증하지 못해 반환하지 않았습니다.", payload);
    }
  } catch (error) {
    showMessage("요청 오류", error.message);
  } finally {
    button.disabled = false;
    button.textContent = "기준서 검색";
    systemStatus.textContent = "근거 검색 준비";
  }
});
