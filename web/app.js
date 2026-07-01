const form = document.querySelector("#predictForm");
const resultGrid = document.querySelector("#resultGrid");
const statusPill = document.querySelector("#statusPill");
const submitButton = document.querySelector("#submitButton");
const sampleButton = document.querySelector("#sampleButton");

const targetLabels = {
  category: "カテゴリ",
  priority: "優先度",
  department: "担当部署",
};

const samples = [
  {
    text: "出勤打刻が全社で利用できず業務が止まっています",
    impact_scope: "全社",
    requester_role: "管理者",
    channel: "Slack",
  },
  {
    text: "請求書の一括出力が失敗して月末処理が完了しません",
    impact_scope: "全社",
    requester_role: "経理担当",
    channel: "メール",
  },
  {
    text: "VPN接続後に社内システムへアクセスできません",
    impact_scope: "個人",
    requester_role: "社員",
    channel: "問い合わせフォーム",
  },
  {
    text: "API連携で取引先データが欠落しています",
    impact_scope: "複数部署",
    requester_role: "管理者",
    channel: "メール",
  },
];

let sampleIndex = 0;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = getPayload();
  await predict(payload);
});

sampleButton.addEventListener("click", () => {
  sampleIndex = (sampleIndex + 1) % samples.length;
  setPayload(samples[sampleIndex]);
});

function getPayload() {
  const formData = new FormData(form);
  return {
    text: formData.get("text"),
    impact_scope: formData.get("impact_scope"),
    requester_role: formData.get("requester_role"),
    channel: formData.get("channel"),
  };
}

function setPayload(payload) {
  form.elements.text.value = payload.text;
  form.elements.impact_scope.value = payload.impact_scope;
  form.elements.requester_role.value = payload.requester_role;
  form.elements.channel.value = payload.channel;
}

async function predict(payload) {
  setStatus("Predicting", "is-busy");
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error);
    }

    renderPredictions(body.predictions);
    setStatus("Done", "is-done");
  } catch (error) {
    renderError(error.message);
    setStatus("Error", "is-error");
  } finally {
    submitButton.disabled = false;
  }
}

function renderPredictions(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0) {
    throw new Error("Prediction response is empty.");
  }

  resultGrid.replaceChildren(
    ...predictions.map((prediction) => createPredictionCard(prediction)),
  );
}

function createPredictionCard(prediction) {
  if (!(prediction.target in targetLabels)) {
    throw new Error(`Unknown prediction target: ${prediction.target}`);
  }
  if (!Array.isArray(prediction.ranking) || prediction.ranking.length === 0) {
    throw new Error(`Ranking is empty for target: ${prediction.target}`);
  }

  const card = document.createElement("article");
  card.className = "prediction-card";
  card.dataset.target = prediction.target;

  const top = document.createElement("div");
  top.className = "prediction-top";

  const labelGroup = document.createElement("div");
  const targetName = document.createElement("span");
  targetName.className = "target-name";
  targetName.textContent = targetLabels[prediction.target];

  const predictedLabel = document.createElement("strong");
  predictedLabel.className = "predicted-label";
  predictedLabel.textContent = prediction.label;
  labelGroup.append(targetName, predictedLabel);

  const confidence = document.createElement("span");
  confidence.className = "confidence";
  confidence.textContent = formatPercent(prediction.confidence);
  top.append(labelGroup, confidence);

  const rankingList = document.createElement("div");
  rankingList.className = "ranking-list";
  prediction.ranking.forEach((rank) => {
    rankingList.append(createRankingRow(rank));
  });

  card.append(top, rankingList);
  return card;
}

function createRankingRow(rank) {
  const row = document.createElement("div");
  row.className = "ranking-row";

  const label = document.createElement("span");
  label.className = "ranking-label";
  label.textContent = rank.label;

  const track = document.createElement("span");
  track.className = "bar-track";

  const fill = document.createElement("span");
  fill.className = "bar-fill";
  fill.style.width = formatPercent(rank.confidence);
  track.append(fill);

  const value = document.createElement("span");
  value.className = "ranking-value";
  value.textContent = formatPercent(rank.confidence);

  row.append(label, track, value);
  return row;
}

function renderError(message) {
  const card = document.createElement("article");
  card.className = "error-card";
  const title = document.createElement("strong");
  title.textContent = "Error";
  const detail = document.createElement("span");
  detail.textContent = message;
  card.append(title, detail);
  resultGrid.replaceChildren(card);
}

function setStatus(text, className) {
  statusPill.className = `status-pill ${className}`;
  statusPill.textContent = text;
}

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}
