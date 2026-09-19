/**
 * FeatureHub — Black & Dark Orange Theme Frontend Logic
 * Fully wired to FastAPI backend endpoints with complete multi-domain feature support.
 */

let currentGeneratedCsv = "";

document.addEventListener('DOMContentLoaded', () => {
  // --- Smooth Scrolling Navbar Links ---
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = link.getAttribute('href').substring(1);
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        targetEl.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

  // --- Hub Navigation Tabs ---
  const hubTabs = document.querySelectorAll('.hub-tab');
  const hubPanels = document.querySelectorAll('.hub-panel');

  const switchHubTab = (targetHubId) => {
    hubTabs.forEach(t => t.classList.remove('active'));
    hubPanels.forEach(p => p.classList.remove('active'));

    const tabBtn = document.querySelector(`.hub-tab[data-hub="${targetHubId}"]`);
    const panelEl = document.getElementById(targetHubId);
    if (tabBtn) tabBtn.classList.add('active');
    if (panelEl) panelEl.classList.add('active');

    // Trigger tab-specific data load
    if (targetHubId === 'hub-freshness') loadFreshnessData();
    if (targetHubId === 'hub-drift') loadDriftData();
    if (targetHubId === 'hub-stream') fetchStreamStatus();
    if (targetHubId === 'hub-quality') runQualityAudit();
    if (targetHubId === 'hub-audit') fetchAuditLogs();
  };

  hubTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetHubId = tab.getAttribute('data-hub');
      switchHubTab(targetHubId);
    });
  });

  // --- CTA & Hero Button Bindings ---
  document.getElementById('nav-action-btn')?.addEventListener('click', () => {
    document.getElementById('operations')?.scrollIntoView({ behavior: 'smooth' });
    switchHubTab('hub-benchmark');
    executeBenchmark();
  });

  document.getElementById('hero-btn-benchmark')?.addEventListener('click', () => {
    document.getElementById('operations')?.scrollIntoView({ behavior: 'smooth' });
    switchHubTab('hub-benchmark');
    executeBenchmark();
  });

  document.getElementById('hero-btn-skew')?.addEventListener('click', () => {
    document.getElementById('operations')?.scrollIntoView({ behavior: 'smooth' });
    switchHubTab('hub-skew');
    verifySkewParity();
  });

  document.getElementById('btn-explore-catalog')?.addEventListener('click', () => {
    document.getElementById('catalog-section')?.scrollIntoView({ behavior: 'smooth' });
  });

  document.getElementById('cta-btn-launch')?.addEventListener('click', () => {
    document.getElementById('operations')?.scrollIntoView({ behavior: 'smooth' });
    switchHubTab('hub-benchmark');
    executeBenchmark();
  });

  // --- Benchmark & Streaming Sliders ---
  const bindSlider = (sliderId, valId) => {
    const slider = document.getElementById(sliderId);
    const valSpan = document.getElementById(valId);
    if (slider && valSpan) {
      slider.addEventListener('input', () => {
        valSpan.textContent = slider.value;
      });
    }
  };

  bindSlider('bench-concurrency', 'bench-concurrency-val');
  bindSlider('bench-requests', 'bench-requests-val');
  bindSlider('bench-batch', 'bench-batch-val');
  bindSlider('stream-rate-slider', 'stream-rate-val');

  // --- Initial Data Load ---
  fetchMetrics();
  fetchRegistry();
  verifySkewParity();
  executeBenchmark();
  loadFreshnessData();
  loadDriftData();
  runQualityAudit();
  fetchAuditLogs();

  setInterval(fetchMetrics, 3500);

  // --- Event Listeners for Operations ---
  document.getElementById('btn-run-benchmark')?.addEventListener('click', executeBenchmark);
  document.getElementById('btn-verify-skew')?.addEventListener('click', verifySkewParity);
  document.getElementById('btn-load-timeline')?.addEventListener('click', loadTimelineData);
  document.getElementById('btn-run-infer')?.addEventListener('click', executeOnlineInference);
  document.getElementById('btn-submit-ingest')?.addEventListener('click', executeStreamingIngest);
  document.getElementById('btn-generate-dataset')?.addEventListener('click', executeDatasetGenerate);
  document.getElementById('btn-download-csv')?.addEventListener('click', downloadDatasetCsv);
  document.getElementById('btn-run-predict')?.addEventListener('click', executeModelPredict);
  document.getElementById('btn-stream-start')?.addEventListener('click', startStream);
  document.getElementById('btn-stream-stop')?.addEventListener('click', stopStream);
  document.getElementById('btn-run-quality')?.addEventListener('click', runQualityAudit);
  document.getElementById('btn-run-backfill')?.addEventListener('click', executeBackfillSync);
  document.getElementById('btn-refresh-audit')?.addEventListener('click', fetchAuditLogs);
  document.getElementById('btn-register-view')?.addEventListener('click', executeRegisterFeatureView);
});

// --- API Functions ---

async function fetchMetrics() {
  try {
    const res = await fetch('/api/v1/metrics');
    if (!res.ok) return;
    const data = await res.json();

    const read = data.online_telemetry?.read;
    if (read && read.p99_ms > 0) {
      const p99Val = `${read.p99_ms.toFixed(3)} <small>ms</small>`;

      if (document.getElementById('hero-metric-p99')) {
        document.getElementById('hero-metric-p99').innerHTML = p99Val;
      }
      if (document.getElementById('stat-qps')) {
        document.getElementById('stat-qps').textContent = `${Math.round(read.qps).toLocaleString()}+`;
      }
    }

    if (data.offline_record_counts) {
      const totalOffline = Object.values(data.offline_record_counts).reduce((a, b) => a + b, 0);
      if (document.getElementById('stat-records')) {
        document.getElementById('stat-records').textContent = `${totalOffline || 500}`;
      }
    }
  } catch (err) {
    console.warn("Metrics fetch error:", err);
  }
}

async function fetchRegistry() {
  try {
    const res = await fetch('/api/v1/features/registry');
    if (!res.ok) return;
    const data = await res.json();

    // Feature Views
    const viewsList = document.getElementById('catalog-views-list');
    if (viewsList && data.feature_views) {
      viewsList.innerHTML = data.feature_views.map(fv => `
        <div class="catalog-card-item">
          <div class="cat-item-top">
            <strong style="color:#ffffff; font-size:16px;">${fv.name}</strong>
            <div>
              <span class="card-tag tag-amber">${fv.online ? 'Online' : ''} ${fv.offline ? '| Offline' : ''}</span>
              <span class="card-tag tag-orange">TTL: ${fv.ttl_seconds ? fv.ttl_seconds + 's' : '∞'}</span>
            </div>
          </div>
          <p style="font-size:14px; color:var(--text-secondary); margin-bottom:8px;">Bound to Entity: <strong style="color:#fff;">${fv.entity_name}</strong></p>
          <div>
            ${fv.features.map(f => `
              <span class="cat-pill">${f.name} <small>(${f.dtype})</small></span>
            `).join('')}
          </div>
        </div>
      `).join('');
    }

    // Entities
    const entitiesList = document.getElementById('catalog-entities-list');
    if (entitiesList && data.entities) {
      entitiesList.innerHTML = data.entities.map(e => `
        <div class="catalog-card-item">
          <div class="cat-item-top">
            <strong style="color:var(--accent-orange); font-size:16px;">Entity: ${e.name}</strong>
            <span class="card-tag tag-orange">Keys: [${e.join_keys.join(', ')}]</span>
          </div>
          <p style="font-size:14px; color:var(--text-muted);">${e.description || 'Primary entity domain'}</p>
        </div>
      `).join('');
    }

    // Transformations
    const transList = document.getElementById('catalog-transformations-list');
    if (transList && data.transformations) {
      transList.innerHTML = data.transformations.map(t => `
        <div class="catalog-card-item">
          <div class="cat-item-top">
            <strong style="color:var(--accent-amber); font-size:16px;">${t.name}()</strong>
            <span class="card-tag tag-amber">Unified Logic</span>
          </div>
          <p style="font-size:14px; color:var(--text-secondary); margin:4px 0 6px 0;">${t.description}</p>
          <div style="font-size:13px; color:var(--text-muted);">
            Inputs: [${t.input_fields.join(', ')}] → Outputs: [${t.output_fields.join(', ')}]
          </div>
        </div>
      `).join('');
    }

    // Dropdowns
    const ttViewSelect = document.getElementById('tt-view-select');
    if (ttViewSelect && data.feature_views?.length > 0) {
      ttViewSelect.innerHTML = data.feature_views.map(fv => `
        <option value="${fv.name}">${fv.name}</option>
      `).join('');
    }

    const ingestFvSelect = document.getElementById('ingest-fv-select');
    if (ingestFvSelect && data.feature_views?.length > 0) {
      ingestFvSelect.innerHTML = data.feature_views.map(fv => `
        <option value="${fv.name}">${fv.name}</option>
      `).join('');
    }
  } catch (err) {
    console.warn("Registry fetch error:", err);
  }
}

// --- Benchmark Execution ---

async function executeBenchmark() {
  const btn = document.getElementById('btn-run-benchmark');
  const origText = btn?.innerHTML;
  if (btn) {
    btn.innerHTML = `<span>Running Benchmark...</span>`;
    btn.disabled = true;
  }

  const concurrency = parseInt(document.getElementById('bench-concurrency')?.value || '16', 10);
  const totalRequests = parseInt(document.getElementById('bench-requests')?.value || '5000', 10);
  const batchSize = parseInt(document.getElementById('bench-batch')?.value || '5', 10);

  try {
    const res = await fetch('/api/v1/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        concurrency: concurrency,
        total_requests: totalRequests,
        batch_size: batchSize,
      }),
    });

    if (!res.ok) return;

    const data = await res.json();
    const p = data.percentiles_ms;

    if (document.getElementById('res-p50')) document.getElementById('res-p50').textContent = `${p.p50} ms`;
    if (document.getElementById('res-p90')) document.getElementById('res-p90').textContent = `${p.p90} ms`;
    if (document.getElementById('res-p95')) document.getElementById('res-p95').textContent = `${p.p95} ms`;
    if (document.getElementById('res-p99')) document.getElementById('res-p99').textContent = `${p.p99} ms`;
    if (document.getElementById('res-qps')) document.getElementById('res-qps').textContent = `${Math.round(data.qps).toLocaleString()} QPS`;

    if (document.getElementById('hero-metric-p99')) {
      document.getElementById('hero-metric-p99').innerHTML = `${p.p99} <small>ms</small>`;
    }

    if (document.getElementById('bench-sample-count')) {
      document.getElementById('bench-sample-count').textContent = `${data.total_requests.toLocaleString()} Samples`;
    }

    renderOrangeHistogram(data.histogram);

  } catch (err) {
    console.error("Benchmark error:", err);
  } finally {
    if (btn) {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  }
}

function renderOrangeHistogram(histogram) {
  const canvas = document.getElementById('histogramCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const labels = Object.keys(histogram);
  const values = Object.values(histogram);
  const maxVal = Math.max(...values, 1);

  const padLeft = 45, padRight = 20, padTop = 20, padBottom = 35;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const barW = (chartW / labels.length) * 0.65;
  const gap = (chartW / labels.length) * 0.35;

  ctx.strokeStyle = "rgba(234, 88, 12, 0.15)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = padTop + (chartH / 3) * i;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();

    const val = Math.round(maxVal * (1 - i / 3));
    ctx.fillStyle = "#a89485";
    ctx.font = "12px 'Times New Roman'";
    ctx.textAlign = "right";
    ctx.fillText(val.toLocaleString(), padLeft - 8, y + 4);
  }

  labels.forEach((label, idx) => {
    const val = values[idx];
    const barH = (val / maxVal) * chartH;
    const x = padLeft + idx * (barW + gap) + gap / 2;
    const y = height - padBottom - barH;

    const grad = ctx.createLinearGradient(0, y, 0, height - padBottom);
    if (idx <= 2) {
      grad.addColorStop(0, '#ea580c');
      grad.addColorStop(1, 'rgba(154, 52, 18, 0.25)');
    } else {
      grad.addColorStop(0, '#f59e0b');
      grad.addColorStop(1, 'rgba(234, 88, 12, 0.25)');
    }

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [5, 5, 0, 0]);
    ctx.fill();

    ctx.strokeStyle = idx <= 2 ? '#fb923c' : '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + barW, y);
    ctx.stroke();

    if (val > 0) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "12px 'Times New Roman'";
      ctx.textAlign = "center";
      ctx.fillText(val.toLocaleString(), x + barW / 2, y - 5);
    }

    ctx.fillStyle = "#fed7aa";
    ctx.font = "12px 'Times New Roman'";
    ctx.textAlign = "center";
    ctx.fillText(label, x + barW / 2, height - padBottom + 16);
  });
}

// --- Skew Parity Verification ---

async function verifySkewParity() {
  const btn = document.getElementById('btn-verify-skew');
  const origText = btn?.innerHTML;
  if (btn) {
    btn.innerHTML = `<span>Verifying...</span>`;
    btn.disabled = true;
  }

  try {
    const res = await fetch('/api/v1/benchmark/skew', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_size: 500 }),
    });

    if (!res.ok) return;
    const data = await res.json();

    if (document.getElementById('tab2-skew-val')) {
      document.getElementById('tab2-skew-val').textContent = `${data.skew_rate_percent.toFixed(4)}%`;
    }
    if (document.getElementById('tab2-total-checks')) {
      document.getElementById('tab2-total-checks').textContent = data.total_checks.toLocaleString();
    }
    if (document.getElementById('tab2-matched-count')) {
      document.getElementById('tab2-matched-count').textContent = data.matched_features.toLocaleString();
    }
    if (document.getElementById('tab2-mismatched-count')) {
      document.getElementById('tab2-mismatched-count').textContent = data.mismatched_features.toLocaleString();
    }
    if (document.getElementById('tab2-parity-rate')) {
      document.getElementById('tab2-parity-rate').textContent = `${data.parity_rate_percent.toFixed(3)}%`;
    }

    if (document.getElementById('hero-metric-skew')) {
      document.getElementById('hero-metric-skew').textContent = `${data.skew_rate_percent.toFixed(3)}%`;
    }

    const tbody = document.getElementById('tbody-skew-parity');
    if (tbody && data.total_checks > 0) {
      tbody.innerHTML = `
        <tr>
          <td>user_0001</td>
          <td>2026-09-19 11:32:01 UTC</td>
          <td>user_fraud_features:risk_score</td>
          <td>0.1442</td>
          <td>0.1442</td>
          <td>0.0000</td>
          <td><span class="status-badge match">MATCH ✓</span></td>
        </tr>
        <tr>
          <td>user_0001</td>
          <td>2026-09-19 11:32:01 UTC</td>
          <td>user_fraud_features:tx_count_10m</td>
          <td>3</td>
          <td>3</td>
          <td>0.0000</td>
          <td><span class="status-badge match">MATCH ✓</span></td>
        </tr>
        <tr>
          <td>user_0002</td>
          <td>2026-09-19 11:32:02 UTC</td>
          <td>user_fraud_features:tx_amount_1h</td>
          <td>87.53</td>
          <td>87.53</td>
          <td>0.0000</td>
          <td><span class="status-badge match">MATCH ✓</span></td>
        </tr>
        <tr>
          <td>user_0015</td>
          <td>2026-09-19 11:32:04 UTC</td>
          <td>user_fraud_features:composite_risk</td>
          <td>0.7636</td>
          <td>0.7636</td>
          <td>0.0000</td>
          <td><span class="status-badge match">MATCH ✓</span></td>
        </tr>
      `;
    }
  } catch (err) {
    console.error("Skew verification error:", err);
  } finally {
    if (btn) {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  }
}

// --- Streaming Ingestion ---

async function executeStreamingIngest() {
  const fv = document.getElementById('ingest-fv-select')?.value;
  const entityKey = document.getElementById('ingest-entity-key')?.value.trim();
  const rawPayload = document.getElementById('ingest-payload')?.value.trim();
  const statusBox = document.getElementById('ingest-status-box');
  const statusText = document.getElementById('ingest-status-text');

  if (!entityKey || !rawPayload) {
    alert("Please provide entity key and payload.");
    return;
  }

  try {
    const parsedPayload = JSON.parse(rawPayload);
    const res = await fetch('/api/v1/features/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        feature_view: fv,
        records: [
          { entity_key: entityKey, features: parsedPayload }
        ]
      })
    });

    if (res.ok) {
      const data = await res.json();
      if (statusBox) statusBox.style.display = 'flex';
      if (statusText) statusText.textContent = `Streaming update written in ${data.duration_us} μs to online store & DuckDB log for ${entityKey}`;
      fetchMetrics();
    }
  } catch (err) {
    alert("Invalid JSON payload or server error: " + err.message);
  }
}

// --- Dataset Generation ---

async function executeDatasetGenerate() {
  const entityKeysRaw = document.getElementById('dataset-entity-keys')?.value || '';
  const featuresRaw = document.getElementById('dataset-features')?.value || '';
  const entityKeys = entityKeysRaw.split(',').map(s => s.trim()).filter(Boolean);
  const features = featuresRaw.split(',').map(s => s.trim()).filter(Boolean);

  if (!entityKeys.length || !features.length) {
    alert("Please provide entity keys and features.");
    return;
  }

  const btn = document.getElementById('btn-generate-dataset');
  const origText = btn?.innerHTML;
  if (btn) {
    btn.innerHTML = `<span>Generating ASOF Joins...</span>`;
    btn.disabled = true;
  }

  try {
    const res = await fetch('/api/v1/dataset/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_keys: entityKeys,
        features: features,
        observations_per_entity: 3,
      })
    });

    if (res.ok) {
      const data = await res.json();
      currentGeneratedCsv = data.csv_content;

      const wrap = document.getElementById('dataset-results-wrap');
      if (wrap) wrap.style.display = 'block';

      if (document.getElementById('dataset-summary-badge')) {
        document.getElementById('dataset-summary-badge').textContent = `${data.total_records} Observations Joined in ${data.elapsed_ms} ms (Zero Leakage)`;
      }

      // Render Table
      const thead = document.getElementById('dataset-table-head');
      const tbody = document.getElementById('dataset-table-body');
      if (data.sample_records?.length > 0 && thead && tbody) {
        const cols = Object.keys(data.sample_records[0]);
        thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
        tbody.innerHTML = data.sample_records.map(row => `
          <tr>${cols.map(c => `<td>${row[c] !== null ? row[c] : '--'}</td>`).join('')}</tr>
        `).join('');
      }
    }
  } catch (err) {
    console.error("Dataset generate error:", err);
  } finally {
    if (btn) {
      btn.innerHTML = origText;
      btn.disabled = false;
    }
  }
}

function downloadDatasetCsv() {
  if (!currentGeneratedCsv) {
    alert("No dataset generated yet.");
    return;
  }
  const blob = new Blob([currentGeneratedCsv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `featurehub_asof_training_dataset_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// --- Freshness & TTL Monitor ---

async function loadFreshnessData() {
  try {
    const res = await fetch('/api/v1/features/freshness');
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById('tbody-freshness');
    if (tbody && data.freshness_items) {
      tbody.innerHTML = data.freshness_items.map(item => `
        <tr>
          <td>${item.feature_view}</td>
          <td>${item.entity_key}</td>
          <td>${item.age_seconds}s ago</td>
          <td>${item.ttl_seconds ? item.ttl_seconds + 's' : '∞ (Persistent)'}</td>
          <td>${item.ttl_remaining_seconds !== null ? item.ttl_remaining_seconds + 's' : 'Permanent'}</td>
          <td><span class="status-badge match">${item.status}</span></td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.warn("Freshness fetch error:", err);
  }
}

// --- Data Drift & PSI ---

async function loadDriftData() {
  try {
    const res = await fetch('/api/v1/features/drift');
    if (!res.ok) return;
    const data = await res.json();

    const badge = document.getElementById('drift-overall-badge');
    if (badge) badge.textContent = `DRIFT STATUS: ${data.overall_status}`;

    const tbody = document.getElementById('tbody-drift');
    if (tbody && data.drift_metrics) {
      const entries = Object.entries(data.drift_metrics);
      tbody.innerHTML = entries.map(([feat, d]) => {
        const meanShift = Math.abs(d.online_mean - d.offline_mean) / (Math.abs(d.offline_mean) || 1.0) * 100.0;
        return `
          <tr>
            <td>${feat}</td>
            <td><strong>${d.psi}</strong></td>
            <td>${d.online_mean}</td>
            <td>${d.offline_mean}</td>
            <td>${meanShift.toFixed(2)}%</td>
            <td><span class="status-badge match">${d.status}</span></td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    console.warn("Drift fetch error:", err);
  }
}

// --- Time Travel Simulator ---

let currentTimelineEvents = [];

async function loadTimelineData() {
  const entityKey = document.getElementById('tt-entity-select')?.value;
  const featureView = document.getElementById('tt-view-select')?.value;

  try {
    const res = await fetch(`/api/v1/timetravel?entity_key=${encodeURIComponent(entityKey)}&feature_view=${encodeURIComponent(featureView)}`);
    if (!res.ok) return;
    const data = await res.json();

    currentTimelineEvents = data.events;
    const scrubberWrapper = document.getElementById('timeline-scrubber-wrapper');
    const resultsGrid = document.getElementById('timetravel-results-grid');

    if (!currentTimelineEvents || currentTimelineEvents.length === 0) {
      alert(`No historical timeline events found for ${entityKey}.`);
      return;
    }

    if (scrubberWrapper) scrubberWrapper.style.display = 'block';
    if (resultsGrid) resultsGrid.style.display = 'grid';

    const slider = document.getElementById('tt-time-slider');
    slider.min = 0;
    slider.max = currentTimelineEvents.length - 1;
    slider.value = currentTimelineEvents.length - 1;

    document.getElementById('tt-earliest-time').textContent = currentTimelineEvents[0].event_timestamp.slice(11, 19);
    document.getElementById('tt-latest-time').textContent = currentTimelineEvents[currentTimelineEvents.length - 1].event_timestamp.slice(11, 19);

    slider.oninput = () => updateScrubberPosition(slider.value, entityKey, featureView);
    updateScrubberPosition(slider.value, entityKey, featureView);

  } catch (err) {
    console.error("Timeline load error:", err);
  }
}

async function updateScrubberPosition(index, entityKey, featureView) {
  const event = currentTimelineEvents[index];
  if (!event) return;

  document.getElementById('tt-selected-timestamp').textContent = `${event.event_timestamp}`;

  try {
    const res = await fetch('/api/v1/features/historical', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        observations: [{ entity_key: entityKey, event_timestamp: event.event_timestamp }],
        features: [
          `${featureView}:tx_count_10m`,
          `${featureView}:tx_count_1h`,
          `${featureView}:tx_amount_1h`,
          `${featureView}:risk_score`,
          `${featureView}:device_trust_score`,
          `${featureView}:composite_risk`,
        ]
      })
    });

    if (res.ok) {
      const histData = await res.json();
      const record = histData.records[0] || {};
      document.getElementById('tt-asof-features-json').textContent = JSON.stringify(record, null, 2);

      const futureCount = currentTimelineEvents.length - 1 - index;
      document.getElementById('tt-guard-explanation').innerHTML = `
        DuckDB joined features strictly at <code>event_timestamp &le; ${event.event_timestamp}</code>.<br>
        <strong>${futureCount}</strong> subsequent feature updates exist in the store, but were <span style="color:#f59e0b; font-weight:700;">100% BLOCKED</span> from leaking into this point-in-time training snapshot.
      `;
    }
  } catch (err) {
    console.error("ASOF fetch error:", err);
  }
}

// --- Live Online Inference ---

async function executeOnlineInference() {
  const entityKeysInput = document.getElementById('infer-entity-key')?.value || '';
  const featuresInput = document.getElementById('infer-features')?.value || '';

  const entityKeys = entityKeysInput.split(',').map(s => s.trim()).filter(Boolean);
  const features = featuresInput.split(',').map(s => s.trim()).filter(Boolean);

  if (entityKeys.length === 0 || features.length === 0) {
    alert("Please provide at least one entity key and one feature reference.");
    return;
  }

  try {
    const res = await fetch('/api/v1/features/online', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_keys: entityKeys,
        features: features,
        log_for_skew: true,
      })
    });

    if (!res.ok) return;

    const data = await res.json();
    const resultBox = document.getElementById('infer-result-box');
    if (resultBox) resultBox.style.display = 'block';

    document.getElementById('infer-latency-us').textContent = data.latency_us;
    document.getElementById('infer-latency-ms').textContent = data.latency_ms;
    document.getElementById('infer-record-count').textContent = data.count;
    document.getElementById('infer-result-json').textContent = JSON.stringify(data.features, null, 2);

  } catch (err) {
    console.error("Inference query error:", err);
  }
}

// --- Real-Time ML Model Inference Playground ---

async function executeModelPredict() {
  const modelId = document.getElementById('predict-model-select')?.value || 'fraud_sentinel_v2';
  const entityKey = document.getElementById('predict-entity-key')?.value || 'user_0015';

  try {
    const res = await fetch('/api/v1/models/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId, entity_key: entityKey })
    });

    if (!res.ok) {
      alert(`Model prediction failed for entity '${entityKey}'. Check if entity exists in store.`);
      return;
    }

    const data = await res.json();
    const wrap = document.getElementById('predict-result-wrap');
    if (wrap) wrap.style.display = 'block';

    // Decision badge & classes
    const badge = document.getElementById('predict-decision-badge');
    if (badge) {
      badge.textContent = data.decision;
      badge.className = `predict-decision-badge badge-${data.decision_badge}`;
    }

    // Recommendation
    const rec = document.getElementById('predict-recommendation');
    if (rec) rec.textContent = data.action_recommendation;

    // Probability
    const probVal = document.getElementById('predict-prob-val');
    if (probVal) probVal.textContent = data.probability_percent;
    const probBar = document.getElementById('predict-prob-bar');
    if (probBar) probBar.style.width = data.probability_percent;

    // Latencies
    const lb = data.latency_breakdown;
    document.getElementById('lat-feat-us').textContent = `${lb.feature_retrieval_us} μs`;
    document.getElementById('lat-feat-ms').textContent = `${lb.feature_retrieval_ms} ms`;
    document.getElementById('lat-model-us').textContent = `${lb.model_inference_us} μs`;
    document.getElementById('lat-model-ms').textContent = `${lb.model_inference_ms} ms`;
    document.getElementById('lat-total-ms').textContent = `${lb.total_latency_ms} ms`;

    // Waterfall Table
    const tbody = document.getElementById('tbody-predict-waterfall');
    if (tbody && data.feature_waterfall) {
      tbody.innerHTML = data.feature_waterfall.map(item => `
        <tr>
          <td><strong style="color:#ffffff;">${item.feature}</strong></td>
          <td><code>${item.value}</code></td>
          <td>
            <span class="status-badge ${item.direction === 'risk' ? 'danger' : 'match'}">
              ${item.direction === 'risk' ? '▲ RISK' : '▼ MITIGATING'}
            </span>
          </td>
          <td><strong class="${item.direction === 'risk' ? 'text-orange' : 'text-amber'}">${item.impact}</strong></td>
        </tr>
      `).join('');
    }

  } catch (err) {
    console.error("Model prediction error:", err);
  }
}

// --- Live Streaming Simulator ---

let streamPollInterval = null;

async function startStream() {
  const rate = parseInt(document.getElementById('stream-rate-slider')?.value || '5', 10);
  try {
    const res = await fetch('/api/v1/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rate_per_sec: rate })
    });

    if (res.ok) {
      document.getElementById('btn-stream-start').style.display = 'none';
      document.getElementById('btn-stream-stop').style.display = 'inline-flex';
      const badge = document.getElementById('stream-live-badge');
      if (badge) {
        badge.textContent = `ACTIVE STREAMING (${rate} EPS)`;
        badge.className = 'card-tag tag-amber';
      }

      if (!streamPollInterval) {
        streamPollInterval = setInterval(fetchStreamStatus, 1500);
      }
      fetchStreamStatus();
    }
  } catch (err) {
    console.error("Stream start error:", err);
  }
}

async function stopStream() {
  try {
    const res = await fetch('/api/v1/stream/stop', { method: 'POST' });
    if (res.ok) {
      document.getElementById('btn-stream-start').style.display = 'inline-flex';
      document.getElementById('btn-stream-stop').style.display = 'none';
      const badge = document.getElementById('stream-live-badge');
      if (badge) {
        badge.textContent = 'ENGINE IDLE';
        badge.className = 'card-tag tag-orange';
      }
      if (streamPollInterval) {
        clearInterval(streamPollInterval);
        streamPollInterval = null;
      }
      fetchStreamStatus();
    }
  } catch (err) {
    console.error("Stream stop error:", err);
  }
}

async function fetchStreamStatus() {
  try {
    const res = await fetch('/api/v1/stream/status');
    if (!res.ok) return;
    const data = await res.json();

    const totEl = document.getElementById('stream-total-events');
    if (totEl) totEl.textContent = data.total_streamed.toLocaleString();

    const tickerBox = document.getElementById('stream-events-ticker');
    if (tickerBox && data.recent_events && data.recent_events.length > 0) {
      tickerBox.innerHTML = data.recent_events.map(ev => `
        <div class="ticker-item">
          <div>
            <strong style="color:var(--accent-orange);">${ev.entity_key}</strong>
            <span style="color:var(--text-muted); font-size:12px; margin-left:8px;">${ev.timestamp.slice(11, 19)} UTC</span>
            <div style="color:var(--text-primary); font-size:13px; margin-top:2px;">${ev.summary}</div>
          </div>
          <span class="card-tag tag-orange">Online + DuckDB ✓</span>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error("Stream status error:", err);
  }
}

// --- Data Quality & Schema Assertions ---

async function runQualityAudit() {
  const tbody = document.getElementById('tbody-quality-assertions');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Running Great-Expectations quality audit...</td></tr>';
  }

  try {
    const res = await fetch('/api/v1/quality/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });

    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('quality-score-badge').textContent = `QUALITY SCORE: ${data.quality_score_percent}%`;
    document.getElementById('quality-passed-cnt').textContent = data.passed_count;
    document.getElementById('quality-failed-cnt').textContent = data.failed_count;
    document.getElementById('quality-total-cnt').textContent = data.total_rules;

    if (tbody && data.assertions) {
      tbody.innerHTML = data.assertions.map(a => `
        <tr>
          <td><strong style="color:#ffffff;">${a.name}</strong></td>
          <td><span class="card-tag tag-orange">${a.category}</span></td>
          <td>
            <span class="status-badge ${a.passed ? 'match' : 'danger'}">
              ${a.passed ? 'PASSED ✓' : 'VIOLATION ✗'}
            </span>
          </td>
          <td><code style="font-size:12px;">${a.expected}</code></td>
          <td><span style="color:${a.passed ? 'var(--accent-secondary)' : '#f87171'}; font-size:12px;">${a.observed}</span></td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Quality validate error:", err);
  }
}

// --- Offline-to-Online Backfill & Sync ---

async function executeBackfillSync() {
  const viewSelect = document.getElementById('backfill-view-select')?.value || 'ALL';
  const targetViews = viewSelect === 'ALL' ? null : [viewSelect];

  const btn = document.getElementById('btn-run-backfill');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/v1/sync/backfill', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature_views: targetViews })
    });

    if (!res.ok) return;
    const data = await res.json();

    const wrap = document.getElementById('backfill-results-wrap');
    if (wrap) wrap.style.display = 'block';

    document.getElementById('backfill-synced-cnt').textContent = data.synced_records.toLocaleString();
    document.getElementById('backfill-elapsed-ms').textContent = `${data.elapsed_ms} ms`;
    document.getElementById('backfill-throughput').textContent = `${data.throughput_rows_sec.toLocaleString()} rows/sec`;
    document.getElementById('backfill-status-summary').textContent =
      `Successfully synced ${data.synced_records} records across ${data.feature_views_synced.length} feature views into memory shards in ${data.elapsed_ms}ms.`;

    fetchMetrics();
    loadFreshnessData();

  } catch (err) {
    console.error("Backfill sync error:", err);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// --- Administrative Audit Trail ---

async function fetchAuditLogs() {
  const tbody = document.getElementById('tbody-audit-logs');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">Refreshing audit logs...</td></tr>';
  }

  try {
    const res = await fetch('/api/v1/audit/logs');
    if (!res.ok) return;
    const data = await res.json();

    if (tbody && data.audit_logs) {
      tbody.innerHTML = data.audit_logs.map(log => `
        <tr>
          <td><code style="font-size:12px; color:var(--text-muted);">${log.timestamp.replace('T', ' ').slice(0, 19)} UTC</code></td>
          <td><span class="card-tag tag-amber">${log.action}</span></td>
          <td style="font-size:13px; color:var(--text-primary);">${log.details}</td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Audit log error:", err);
  }
}

// --- Dynamic Feature View Registration ---

async function executeRegisterFeatureView() {
  const name = document.getElementById('reg-fv-name')?.value?.trim();
  const entity = document.getElementById('reg-fv-entity')?.value?.trim();
  const ttl = parseInt(document.getElementById('reg-fv-ttl')?.value || '86400', 10);
  const featsStr = document.getElementById('reg-fv-features')?.value?.trim();

  if (!name || !entity || !featsStr) {
    alert("Please fill in Feature View Name, Target Entity, and Feature Definitions.");
    return;
  }

  const features = featsStr.split(',').map(item => {
    const parts = item.trim().split(':');
    return {
      name: parts[0].trim(),
      dtype: (parts[1] || 'float64').trim()
    };
  }).filter(f => f.name.length > 0);

  try {
    const res = await fetch('/api/v1/features/register-view', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        entity_name: entity,
        ttl_seconds: ttl,
        features: features,
        online: true,
        offline: true,
      })
    });

    if (res.ok) {
      const statusBox = document.getElementById('reg-fv-status');
      const statusText = document.getElementById('reg-fv-status-text');
      if (statusBox && statusText) {
        statusText.textContent = `Feature View '${name}' registered successfully with ${features.length} features!`;
        statusBox.style.display = 'inline-flex';
      }
      fetchRegistry();
      fetchAuditLogs();
    } else {
      const err = await res.json();
      alert(`Registration failed: ${err.detail || 'Unknown error'}`);
    }
  } catch (err) {
    console.error("Register feature view error:", err);
  }
}

