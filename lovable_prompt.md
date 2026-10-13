Lovable Prompt — “VAYU: Predictive Maintenance Suite (CMAPSS backend)”

Goal
Build a production-looking web app that sits on top of an existing FastAPI backend at http://127.0.0.1:8000. Do not create any backend inside Lovable. Use 4 REST endpoints only:

GET /predict_global_live (poll every 2s)

GET /alerts (poll every 2s)

GET /explain_global_live (poll every 3–5s)

GET /metrics (poll every 10s)

The backend already streams a simulated engine (CMAPSS) window loop. This Lovable app must implement all 10 functional areas below with a clean, dark theme, responsive layout, and professional visual polish.

Data Sources / Polling
Create 4 REST data sources bound to the above endpoints with the polling intervals specified. Handle loading, empty, and error states gracefully (skeletons, toasts, and inline “retry” buttons).

Page 1 — Fleet & Engine Health Dashboard (Home)

Header KPIs (top bar):

Large title: GLOBAL50_MIXED • Fleet Ops

Subcaption: fd_hint={{pred_live.fd_hint}} • window={{pred_live.window}} • last idx={{pred_live.idx}} • latency={{metrics_live.last_latency_ms}}ms.

KPI tiles:

Fleet Size: hardcode 22 (demo).

Needs Attention: compute count(alerts_live where acknowledged==false and severity in ["warning","critical"]) (cap at last 20).

Avg RUL: compute avg(pred_live.trace) rounded.

Open Work Orders: local demo store (start at 3; editable in Planner page).

Triage Strip (blinking cards):

Show up to 4 cards if pred_live.predicted_rul < 40 or recent alerts contain “Vibration spike”/“Temperature rising”.

Each card: a small sparkline (use pred_live.trace), label = “Engine FD={{pred_live.fd_hint}}”.

Fleet Table (demo):

One row for “FD003 / Engine-01 (live)” bound to your live data plus 4–5 static rows for demo (FD001/FD002/FD004).

Columns: Engine ID, Status (Good/Warning/Critical), RUL (value; add P50/P10 demo band from pred_live.predicted_rul±{5,10}), Last Updated (from pred_live.ts), Drift flag (if alerts include “Temperature rising” or “Vibration spike”, mark Drift=Yes), Explain button (navigates to Page 4 with query param engine=Engine-01).

Live Update: All tiles auto-refresh via polling. No manual refresh needed.

Page 2 — Engine Detail & Live Telemetry

Header: Engine ID (Engine-01) • fd_hint, window.

Current Prediction Panel:

Big number: {{pred_live.predicted_rul.toFixed(1)}} cycles.

Confidence band: show P50 (itself), P10/P90 computed as ±(0.2 * predicted_rul) for demo.

Rolling Telemetry:

Three charts (area or line):

“Temperature (proxy) • normalized”

“Vibration (proxy) • normalized”

“Pressure (proxy) • normalized”

For demo, derive small client-side time-series that update on each pred_live poll:

Temp increases slowly if any “Temperature rising” alert is present; else flat.

Vibration increases if “Vibration spike” alert is present; else slight noise.

Pressure stays flat.

Each chart keeps last ~60 points; drop oldest.

Event Timeline:

Show last 20 alerts with time, title, and message. Color-code severity.

Page 3 — Alerts Center

Priority Lanes:

🔴 Critical, ⚠️ Warning, 🟡 Scheduled, 🟢 Info.

Pipe alerts_live items into “Warning” lane (Critical lane remains demo-empty unless a message contains “critical”).

Each card: title — message, time, and an “Acknowledge” toggle (client-only state since backend has GET only).

Filter/search box.

Rules Panel (read-only demo):

Show the composite rules used:

“RUL < 40 cycles → Warning”

“Temperature rising (Δ>+0.1 normalized over last 10 points) → Warning”

“Vibration spike (Δ>+0.2 normalized over last 10 points) → Warning”

Page 4 — Predict & Explain (XAI)

Live XAI Bars (auto, no button):

Fetch /explain_global_live every 3–5s.

Build a horizontal bar chart: labels = features[], values = heatmap[]. Sort desc and show top 8 bars. Show pred at top (current RUL).

“XAI in words” paragraph: “Top drivers now: {top3 features}. Model: {metrics_live.model}. Confidence: (demo) computed from variance of trace.”

What-if Simulator (client-side):

Three numeric inputs (sliders): Δtemp, Δvib, Δpress in [-0.5, +0.5] (normalized units).

Find indices of your chosen proxy features in features[] (e.g., pick any 3 actual sensors and label them Temp/Vib/Press locally).

Let w_temp, w_vib, w_press be their weights from heatmap[].

Compute adjusted RUL (client side):
adjusted_rul = max(0, pred_live.predicted_rul - (w_temp*Δtemp + w_vib*Δvib + w_press*Δpress) * K) with K=10.

Show “Adjusted RUL: X.Y cycles”.

Page 5 — Maintenance Planner

Recommendations (data-driven text):

If pred_live.predicted_rul < 40 and latest alerts include “Vibration spike” → “Inspect Fan module — due in {Math.max(5, round(pred*0.6))} cycles. Reason: Low RUL + rising vibration.”

Else if includes “Temperature rising” → “Check bleed air cooling path next stop.”

Else: “No immediate action.”

Work-order Draft (client-side form):

Asset: Engine-01; Symptoms: last alert; Evidence: link to XAI page.

Save as client-only item; increase “Open Work Orders” KPI on save.

Calendar (demo-only):

Show “Next planned inspection” in N cycles converted to a tentative date.

Page 6 — Historical Explorer

Compare two engines (Engine-01 live vs demo Engine-02):

Two aligned charts: RUL over (synthetic) cycles; vibration trend.

Engine-02 uses a static demo series.

Crosshair + tooltip.

Page 7 — Synthetic Data Lab

Comparison (demo only):

Three bars: Real-only vs Real+Synthetic vs Pretrain-Synth→Finetune-Real (RMSE numbers: 22.1 / 18.6 / 17.4).

A small paragraph: “Synthetic improves coverage for under-represented regimes; reduces RMSE by ~21%.”

Page 8 — Model Ops & Governance

Experiment Registry (demo table):

Columns: run_id, params summary, metric (RMSE), data hash.

Rows: 5 demo entries.

Model Registry panel:

Current: {{metrics_live.model}} (Stage: Prod).

Latency: {{metrics_live.last_latency_ms}}ms.

Failure Rate (demo): 0.0%.

Drift Monitoring (demo chart):

PSI/KS values vs time (static series).

Page 9 — Team Communication Hub

Threaded comments (client only):

Per alert or per engine; supports @mentions (plain text).

Assignments: multi-select “Nishtha / Jubi / Isha”.

Status tags: Open, In Progress, Resolved (client-state).

Page 10 — Reports & Admin

Reports:

Button “Download weekly PDF” (form a simple HTML → PDF in client; include live KPIs + last 10 alerts + top XAI features).

Admin & Settings (client-state):

Roles: Admin, Ops Engineer, Viewer (affects visibility of Planner actions).

Data source: Fixed to “Live Sim (CMAPSS)”.

Thresholds: RUL warning=40 (editable); cycle length=1; window=from backend (read-only).

Non-Functional Requirements (apply globally)

Performance: Keep charts lightweight; do not keep more than last 100 points in client memory. Prefer canvas/virtual lists for long alert lists.

Reliability: If an endpoint fails, show a non-blocking toast and retain last good data (graceful degrade). Auto-retry next poll.

Scalability (demo claim): Stateless UI; polling sources can be switched to WS/SSE later.

Security: No secrets in client. Do not store PII. (All demo.)

Observability: Footer shows: model={{metrics_live.model}} • trace points={{metrics_live.rul_trace_len}} • recent alerts={{metrics_live.alerts}}.

Accessibility: Keyboard nav, ARIA labels for charts and buttons, high-contrast palette, color-blind friendly series.

Error States: For each panel, add skeleton + empty-state text (“No alerts yet”, “No XAI yet — waiting for first window”, etc.).

Visual & UX Notes

Dark theme; subtle glass cards; rounded corners; soft shadows; consistent spacing.

Avoid jitter in charts; animate line updates lightly.

Time formatting from unix seconds → local time.

Bindings Summary

pred_live → big RUL, sparkline, header info, what-if base.

alerts_live → Alerts Center, Triage Strip, timeline.

xai_live → XAI bars + “XAI in words”; what-if weights.

metrics_live → footer KPIs, header latency, model name.