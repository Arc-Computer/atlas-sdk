"""Inline HTML template for the telemetry dashboard."""

from __future__ import annotations

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Atlas Telemetry Dashboard</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 1rem 1.5rem; background: #1e293b; font-size: 1.25rem; font-weight: 600; }
    .layout { display: flex; height: calc(100vh - 64px); }
    nav { width: 320px; border-right: 1px solid #334155; overflow-y: auto; padding: 1rem; background: #111827; }
    nav h2 { margin-top: 0; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }
    nav ul { list-style: none; padding: 0; margin: 0; }
    nav li { margin-bottom: 0.5rem; }
    nav button { width: 100%; padding: 0.5rem 0.75rem; border: 1px solid #334155; border-radius: 0.5rem; background: #1e293b; color: inherit; cursor: pointer; text-align: left; }
    nav button:hover, nav button.active { background: #2563eb; border-color: #2563eb; color: #f8fafc; }
    main { flex: 1; display: flex; flex-direction: column; overflow-y: auto; padding: 1.5rem; gap: 1.5rem; }
    section { background: #111827; border: 1px solid #334155; border-radius: 0.75rem; padding: 1rem 1.25rem; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.4); }
    section h2 { margin-top: 0; font-size: 1.1rem; color: #cbd5f5; }
    .session-meta { display: grid; gap: 0.5rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
    .badge { display: inline-flex; align-items: center; padding: 0.25rem 0.5rem; border-radius: 9999px; background: #1e40af; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .timeline { display: flex; flex-direction: column; gap: 0.75rem; }
    .card { border: 1px solid #334155; border-radius: 0.75rem; padding: 0.75rem 1rem; background: #0f172a; }
    .card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #f8fafc; }
    pre { margin: 0.5rem 0 0 0; padding: 0.75rem; border-radius: 0.5rem; background: #020617; overflow-x: auto; font-size: 0.85rem; }
    .events { max-height: 240px; overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; }
    .events .event { border: 1px solid #334155; border-radius: 0.75rem; padding: 0.75rem; background: #0f172a; font-size: 0.85rem; }
    .muted { color: #94a3b8; font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>Atlas Telemetry Dashboard</header>
  <div class=\"layout\">
    <nav>
      <h2>Sessions</h2>
      <ul id=\"session-list\"><li class=\"muted\">Loading sessions...</li></ul>
    </nav>
    <main>
      <section id=\"session-summary\">
        <h2>Session Overview</h2>
        <p class=\"muted\">Select a session to see stored data.</p>
      </section>
      <section>
        <h2>Step Timeline</h2>
        <div id=\"step-timeline\" class=\"timeline\"><div class=\"muted\">No session selected.</div></div>
      </section>
      <section>
        <h2>Stored Events</h2>
        <div id=\"stored-events\" class=\"events\"><div class=\"muted\">No session selected.</div></div>
      </section>
      <section>
        <h2>Live Telemetry</h2>
        <div id=\"live-events\" class=\"events\"><div class=\"muted\">Awaiting live events...</div></div>
      </section>
    </main>
  </div>
  <script>
    const state = { selectedSession: null };

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      return await response.json();
    }

    function renderSessions(sessions) {
      const list = document.getElementById('session-list');
      list.innerHTML = '';
      if (!sessions.length) {
        list.innerHTML = '<li class="muted">No sessions found.</li>';
        return;
      }
      for (const session of sessions) {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.textContent = `${session.id} · ${session.task}`;
        if (session.id === state.selectedSession) {
          button.classList.add('active');
        }
        button.onclick = () => selectSession(session.id);
        item.appendChild(button);
        list.appendChild(item);
      }
    }

    function formatDate(value) {
      if (!value) {
        return '—';
      }
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return value;
      }
      return date.toLocaleString();
    }

    function renderSessionSummary(session) {
      const container = document.getElementById('session-summary');
      const badge = `<span class="badge">${session.status || 'unknown'}</span>`;
      container.innerHTML = `
        <h2>Session ${session.id}</h2>
        <div class="session-meta">
          <div><strong>Task</strong><br/>${session.task}</div>
          <div><strong>Created</strong><br/>${formatDate(session.created_at)}</div>
          <div><strong>Completed</strong><br/>${formatDate(session.completed_at)}</div>
          <div><strong>Final Answer</strong><br/>${session.final_answer || '—'}</div>
          <div><strong>Status</strong><br/>${badge}</div>
        </div>
        <details style="margin-top:1rem;">
          <summary>Plan JSON</summary>
          <pre>${JSON.stringify(session.plan || {}, null, 2)}</pre>
        </details>
      `;
    }

    function renderTimeline(steps) {
      const container = document.getElementById('step-timeline');
      container.innerHTML = '';
      if (!steps.length) {
        container.innerHTML = '<div class="muted">No step data recorded.</div>';
        return;
      }
      for (const step of steps) {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <h3>Step ${step.step_id}</h3>
          <div class="muted">Attempts: ${step.attempts || 0}</div>
          <details open>
            <summary>Evaluation</summary>
            <pre>${JSON.stringify(step.evaluation || {}, null, 2)}</pre>
          </details>
          <details>
            <summary>Trace</summary>
            <pre>${step.trace || ''}</pre>
          </details>
          <details>
            <summary>Guidance Notes</summary>
            <pre>${JSON.stringify(step.guidance_notes || [], null, 2)}</pre>
          </details>
          <details>
            <summary>Attempt Details</summary>
            <pre>${JSON.stringify(step.attempt_details || [], null, 2)}</pre>
          </details>
        `;
        container.appendChild(card);
      }
    }

    function renderStoredEvents(events) {
      const container = document.getElementById('stored-events');
      container.innerHTML = '';
      if (!events.length) {
        container.innerHTML = '<div class="muted">No stored events.</div>';
        return;
      }
      for (const entry of events) {
        const div = document.createElement('div');
        div.className = 'event';
        div.innerHTML = `<div class="muted">${formatDate(entry.created_at)}</div><pre>${JSON.stringify(entry.event, null, 2)}</pre>`;
        container.appendChild(div);
      }
    }

    function appendLiveEvent(payload) {
      const container = document.getElementById('live-events');
      if (container.dataset.initialized !== 'true') {
        container.innerHTML = '';
        container.dataset.initialized = 'true';
      }
      const entry = document.createElement('div');
      entry.className = 'event';
      const summaryParts = [payload.type];
      const data = payload.data || {};
      if (data.payload && data.payload.event_type) {
        summaryParts.push(data.payload.event_type);
      }
      if (data.payload && data.payload.name) {
        summaryParts.push(data.payload.name);
      }
      if (payload.type === 'session-started' && data.session_id) {
        summaryParts.push(`#${data.session_id}`);
      }
      if (payload.type === 'session-completed' && data.session_id) {
        summaryParts.push(`#${data.session_id}`);
        summaryParts.push(data.status || '');
      }
      entry.innerHTML = `<div class="muted">${summaryParts.filter(Boolean).join(' · ')}</div><pre>${JSON.stringify(payload.data || {}, null, 2)}</pre>`;
      container.prepend(entry);
      while (container.childElementCount > 50) {
        container.removeChild(container.lastChild);
      }
    }

    async function selectSession(sessionId) {
      state.selectedSession = sessionId;
      await Promise.all([
        fetchJson(`/api/sessions/${sessionId}`).then((data) => renderSessionSummary(data.session)),
        fetchJson(`/api/sessions/${sessionId}/steps`).then((data) => renderTimeline(data.steps)),
        fetchJson(`/api/sessions/${sessionId}/events`).then((data) => renderStoredEvents(data.events)),
      ]);
      const buttons = document.querySelectorAll('nav button');
      buttons.forEach((button) => {
        button.classList.toggle('active', button.textContent.startsWith(`${sessionId}`));
      });
    }

    async function refreshSessions() {
      try {
        const data = await fetchJson('/api/sessions');
        renderSessions(data.sessions);
      } catch (error) {
        console.error('Failed to refresh sessions', error);
      }
    }

    function connectWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const url = `${protocol}://${window.location.host}/ws/events`;
      const socket = new WebSocket(url);
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          appendLiveEvent(payload);
          if (payload.type === 'session-started' && state.selectedSession === null && payload.data && payload.data.session_id) {
            state.selectedSession = payload.data.session_id;
            selectSession(payload.data.session_id);
          }
        } catch (error) {
          console.error('Invalid event payload', error);
        }
      };
      socket.onclose = () => {
        setTimeout(connectWebSocket, 2000);
      };
      socket.onerror = () => {
        socket.close();
      };
    }

    document.addEventListener('DOMContentLoaded', () => {
      refreshSessions();
      connectWebSocket();
      setInterval(refreshSessions, 15000);
    });
  </script>
</body>
</html>
"""
