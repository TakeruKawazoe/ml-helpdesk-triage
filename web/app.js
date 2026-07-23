const form = document.querySelector("#predictForm");
const resultGrid = document.querySelector("#resultGrid");
const historyList = document.querySelector("#historyList");
const statusPill = document.querySelector("#statusPill");
const submitButton = document.querySelector("#submitButton");
const sampleButton = document.querySelector("#sampleButton");
const refreshHistoryButton = document.querySelector("#refreshHistoryButton");

const targetLabels = {
  category: "カテゴリ",
  priority: "優先度",
  department: "担当部署",
};

const correctionFields = [
  {
    name: "corrected_category",
    label: "正解カテゴリ",
    target: "category",
    options: ["勤怠", "請求", "権限", "システム障害", "ネットワーク", "アカウント", "データ連携", "端末"],
  },
  {
    name: "corrected_priority",
    label: "正解優先度",
    target: "priority",
    options: ["High", "Middle", "Low"],
  },
  {
    name: "corrected_department",
    label: "正解担当部署",
    target: "department",
    options: ["総務", "経理", "情シス", "開発", "インフラ"],
  },
];

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
let retrainingPollTimer = null;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await predict(getPayload());
});

sampleButton.addEventListener("click", () => {
  sampleIndex = (sampleIndex + 1) % samples.length;
  setPayload(samples[sampleIndex]);
});

refreshHistoryButton.addEventListener("click", async () => {
  await loadHistory();
});

loadHistory();
loadRetrainingStatus();

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

    renderPredictions(
      body.predictions,
      body.prediction_id,
      body.notion_sync,
      body.slack_notification,
    );
    await loadHistory();
    setStatus("Done", "is-done");
  } catch (error) {
    renderError(error.message);
    setStatus("Error", "is-error");
  } finally {
    submitButton.disabled = false;
  }
}

async function saveFeedback(predictionId, formElement) {
  const formData = new FormData(formElement);
  const payload = {
    prediction_id: predictionId,
    corrected_category: formData.get("corrected_category"),
    corrected_priority: formData.get("corrected_priority"),
    corrected_department: formData.get("corrected_department"),
    note: formData.get("note"),
  };

  setStatus("Saving", "is-busy");
  const saveButton = formElement.querySelector("button[type='submit']");
  saveButton.disabled = true;

  try {
    const response = await fetch("/api/feedback", {
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
    formElement.dataset.saved = "true";
    updateNotionSyncBanner(body.notion_sync);
    updateRetrainingBanner(body.retraining);
    await loadHistory();
    setStatus("Saved", "is-done");
  } catch (error) {
    renderError(error.message);
    setStatus("Error", "is-error");
  } finally {
    saveButton.disabled = false;
  }
}

async function loadRetrainingStatus() {
  try {
    const response = await fetch("/api/retraining");
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error);
    }
    updateRetrainingBanner(body.retraining);
  } catch (error) {
    updateRetrainingBanner({
      status: "failed",
      pending_feedback_count: 0,
      threshold: 0,
      message: "再学習状態を取得できませんでした。",
      error: error.message,
    });
  }
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=10");
  const body = await response.json();
  if (!response.ok) {
    renderHistoryError(body.error);
    return;
  }
  renderHistory(body.items);
}

function renderPredictions(predictions, predictionId, notionSync, slackNotification) {
  if (!predictionId) {
    throw new Error("prediction_id is empty.");
  }
  if (!Array.isArray(predictions) || predictions.length === 0) {
    throw new Error("Prediction response is empty.");
  }

  resultGrid.replaceChildren(
    ...predictions.map((prediction) => createPredictionCard(prediction)),
    createNotionSyncBanner(notionSync),
    createSlackNotificationBanner(slackNotification),
    createFeedbackForm(predictions, predictionId),
  );
}

function createNotionSyncBanner(notionSync) {
  const banner = document.createElement("div");
  banner.className = "notion-sync-banner";

  const status = notionSync?.status || "disabled";
  const messages = {
    synced: "Notionに登録しました",
    disabled: "Notion連携は未設定です",
    failed: "Notionへの登録に失敗しました",
    not_linked: "この履歴に対応するNotion行がありません",
  };
  banner.classList.add(`is-${status}`);

  const title = document.createElement("strong");
  title.textContent = messages[status] || `Notion連携: ${status}`;
  banner.append(title);

  if (notionSync?.error) {
    const detail = document.createElement("span");
    detail.textContent = notionSync.error;
    banner.append(detail);
  }
  return banner;
}

function updateNotionSyncBanner(notionSync) {
  const currentBanner = resultGrid.querySelector(".notion-sync-banner");
  const nextBanner = createNotionSyncBanner(notionSync);
  if (currentBanner) {
    currentBanner.replaceWith(nextBanner);
    return;
  }
  resultGrid.prepend(nextBanner);
}

function createSlackNotificationBanner(slackNotification) {
  const banner = document.createElement("div");
  banner.className = "slack-notification-banner";

  const status = slackNotification?.status || "disabled";
  const messages = {
    sent: "Slackへ担当者メンション付きで通知しました",
    skipped: "優先度LowのためSlack通知対象外です",
    disabled: "Slack通知は未設定です",
    failed: "Slack通知に失敗しました",
  };
  banner.classList.add(`is-${status}`);

  const title = document.createElement("strong");
  title.textContent = messages[status] || `Slack通知: ${status}`;
  banner.append(title);

  if (slackNotification?.error) {
    const detail = document.createElement("span");
    detail.textContent = slackNotification.error;
    banner.append(detail);
  }
  return banner;
}

function createRetrainingBanner(retraining) {
  const banner = document.createElement("div");
  banner.className = "retraining-banner";

  const status = retraining.status;
  banner.classList.add(`is-${status}`);

  const title = document.createElement("strong");
  const labels = {
    waiting: `モデル再学習待機中 ${retraining.pending_feedback_count}/${retraining.threshold}件`,
    running: "フィードバックを使って候補モデルを学習中",
    promoted: "再学習モデルを採用しました",
    rejected: "精度を維持するため現行モデルを継続します",
    failed: "モデル再学習に失敗しました",
  };
  title.textContent = labels[status] || `モデル再学習: ${status}`;
  banner.append(title);

  const detail = document.createElement("span");
  detail.textContent = retraining.error || retraining.message;
  banner.append(detail);
  return banner;
}

function updateRetrainingBanner(retraining) {
  const currentBanner = resultGrid.querySelector(".retraining-banner");
  const nextBanner = createRetrainingBanner(retraining);
  if (currentBanner) {
    currentBanner.replaceWith(nextBanner);
  } else {
    resultGrid.append(nextBanner);
  }

  if (retrainingPollTimer) {
    window.clearTimeout(retrainingPollTimer);
    retrainingPollTimer = null;
  }
  if (retraining.status === "running") {
    retrainingPollTimer = window.setTimeout(loadRetrainingStatus, 5000);
  }
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

function createFeedbackForm(predictions, predictionId) {
  const predictedValues = Object.fromEntries(
    predictions.map((prediction) => [prediction.target, prediction.label]),
  );
  const formElement = document.createElement("form");
  formElement.className = "feedback-card";
  formElement.addEventListener("submit", async (event) => {
    event.preventDefault();
    await saveFeedback(predictionId, formElement);
  });

  const title = document.createElement("div");
  title.className = "feedback-title";
  const eyebrow = document.createElement("span");
  eyebrow.className = "target-name";
  eyebrow.textContent = "Feedback";
  const heading = document.createElement("strong");
  heading.textContent = "修正フィードバック";
  title.append(eyebrow, heading);

  const grid = document.createElement("div");
  grid.className = "correction-grid";
  correctionFields.forEach((field) => {
    grid.append(createCorrectionSelect(field, predictedValues[field.target]));
  });

  const noteGroup = document.createElement("label");
  noteGroup.className = "note-group";
  noteGroup.textContent = "メモ";
  const note = document.createElement("textarea");
  note.name = "note";
  note.rows = 3;
  note.placeholder = "修正理由";
  noteGroup.append(note);

  const buttonRow = document.createElement("div");
  buttonRow.className = "button-row feedback-actions";
  const saveButton = document.createElement("button");
  saveButton.type = "submit";
  saveButton.textContent = "修正を保存";
  buttonRow.append(saveButton);

  formElement.append(title, grid, noteGroup, buttonRow);
  return formElement;
}

function createCorrectionSelect(field, selectedValue) {
  const group = document.createElement("label");
  group.className = "field-group";
  group.textContent = field.label;

  const select = document.createElement("select");
  select.name = field.name;
  field.options.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selectedValue;
    select.append(option);
  });

  group.append(select);
  return group;
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

function renderHistory(items) {
  if (!Array.isArray(items) || items.length === 0) {
    const card = document.createElement("article");
    card.className = "empty-state";
    const title = document.createElement("strong");
    title.textContent = "No records";
    const detail = document.createElement("span");
    detail.textContent = "Waiting";
    card.append(title, detail);
    historyList.replaceChildren(card);
    return;
  }

  historyList.replaceChildren(...items.map((item) => createHistoryCard(item)));
}

function createHistoryCard(item) {
  const card = document.createElement("article");
  card.className = "history-card";
  if (item.feedback_saved_at) {
    card.classList.add("has-feedback");
  }

  const top = document.createElement("div");
  top.className = "history-top";
  const text = document.createElement("strong");
  text.textContent = item.text;
  const time = document.createElement("span");
  time.textContent = formatDateTime(item.created_at);
  top.append(text, time);

  const meta = document.createElement("div");
  meta.className = "history-meta";
  meta.append(
    createBadge(`影響範囲: ${item.impact_scope}`),
    createBadge(`依頼者: ${item.requester_role}`),
    createBadge(`経路: ${item.channel}`),
  );
  if (item.notion_sync_status) {
    meta.append(createBadge(notionHistoryLabel(item.notion_sync_status)));
  }
  if (item.slack_notification_status) {
    meta.append(createBadge(slackHistoryLabel(item.slack_notification_status)));
  }

  const predicted = document.createElement("div");
  predicted.className = "history-labels";
  predicted.append(
    createLabelPair("予測カテゴリ", item.predicted_category),
    createLabelPair("予測優先度", item.predicted_priority),
    createLabelPair("予測担当部署", item.predicted_department),
  );

  card.append(top, meta, predicted);

  if (item.feedback_saved_at) {
    const corrected = document.createElement("div");
    corrected.className = "history-labels corrected-labels";
    corrected.append(
      createLabelPair("正解カテゴリ", item.corrected_category),
      createLabelPair("正解優先度", item.corrected_priority),
      createLabelPair("正解担当部署", item.corrected_department),
    );
    card.append(corrected);

    if (item.note) {
      const note = document.createElement("p");
      note.className = "history-note";
      const noteLabel = document.createElement("span");
      noteLabel.textContent = "メモ";
      const noteText = document.createElement("strong");
      noteText.textContent = item.note;
      note.append(noteLabel, noteText);
      card.append(note);
    }
  }

  return card;
}

function notionHistoryLabel(status) {
  const labels = {
    synced: "Notion: 同期済み",
    disabled: "Notion: 未設定",
    failed: "Notion: 同期失敗",
    not_linked: "Notion: 未連携",
  };
  return labels[status] || `Notion: ${status}`;
}

function slackHistoryLabel(status) {
  const labels = {
    sent: "Slack: 通知済み",
    skipped: "Slack: 対象外",
    disabled: "Slack: 未設定",
    failed: "Slack: 通知失敗",
  };
  return labels[status] || `Slack: ${status}`;
}

function createBadge(text) {
  const badge = document.createElement("span");
  badge.className = "meta-badge";
  badge.textContent = text;
  return badge;
}

function createLabelPair(label, value) {
  const pair = document.createElement("span");
  pair.className = "label-pair";
  const labelElement = document.createElement("span");
  labelElement.textContent = label;
  const valueElement = document.createElement("strong");
  valueElement.textContent = value || "-";
  pair.append(labelElement, valueElement);
  return pair;
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

function renderHistoryError(message) {
  const card = document.createElement("article");
  card.className = "error-card";
  const title = document.createElement("strong");
  title.textContent = "History Error";
  const detail = document.createElement("span");
  detail.textContent = message;
  card.append(title, detail);
  historyList.replaceChildren(card);
}

function setStatus(text, className) {
  statusPill.className = `status-pill ${className}`;
  statusPill.textContent = text;
}

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
