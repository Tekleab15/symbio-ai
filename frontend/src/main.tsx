import React, { useEffect, useMemo, useState } from 'react';
import ReactDOM from 'react-dom/client';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

type DemoCase = {
  case_id: string;
  title: string;
  crop: string;
  location: string;
  symptoms: string[];
  urgency: string;
  acreage: number;
};

type EventEnvelope = {
  event_id: string;
  agent: string;
  role: string;
  task_state: string;
  finding: string;
  next_agent?: string | null;
  risk_level: string;
  confidence?: number;
  requires_human_review: boolean;
  payload: any;
  band_delivery?: any;
};

type RunResult = {
  case: any;
  events: EventEnvelope[];
  band_transcript: any[];
};

function App() {
  const [demos, setDemos] = useState<DemoCase[]>([]);
  const [selected, setSelected] = useState('cassava-low-confidence');
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/demo-cases`)
      .then((r) => r.json())
      .then(setDemos)
      .catch((e) => setError(String(e)));
  }, []);

  async function runDemo(caseId = selected) {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/demo/${caseId}/run`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setResult(data);
      setSelected(caseId);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  const metrics = useMemo(() => {
    const events = result?.events || [];
    const rules = events.find((e) => e.agent === 'Rule_Compliance_Agent')?.payload?.triggered_rules || [];
    return {
      handoffs: events.filter((e) => e.next_agent).length,
      rules: rules.length,
      human: result?.case?.requires_human_review ? 'Required' : 'Not required',
      risk: result?.case?.risk_level || 'unknown',
    };
  }, [result]);

  return (
    <main>
      <section className="hero">
        <div>
          <p className="eyebrow">Band of Agents Hackathon MVP</p>
          <h1>Symbio.AI Biosecurity Command</h1>
          <p className="subtitle">
            A Band-powered neuro-symbolic command room where crop vision, agronomy, symbolic safety, open-source auditing, and human review agents coordinate risky biosecurity decisions.
          </p>
          <div className="actions">
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {demos.map((demo) => (
                <option key={demo.case_id} value={demo.case_id}>{demo.title}</option>
              ))}
            </select>
            <button onClick={() => runDemo()} disabled={loading}>{loading ? 'Running mesh...' : 'Run Band Mesh'}</button>
          </div>
          {error && <p className="error">{error}</p>}
        </div>
        <div className="heroCard">
          <h3>Winning edge</h3>
          <p>Neural agents propose. Symbolic agents constrain. Band records every handoff, veto, and escalation.</p>
        </div>
      </section>

      <section className="metrics">
        <Metric label="Band handoffs" value={metrics.handoffs} />
        <Metric label="Rules triggered" value={metrics.rules} />
        <Metric label="Human review" value={metrics.human} />
        <Metric label="Risk" value={metrics.risk} />
      </section>

      {result ? (
        <div className="grid">
          <section className="panel">
            <h2>Agent timeline</h2>
            <Timeline events={result.events} />
          </section>
          <section className="panel">
            <h2>Audit packet</h2>
            <Report report={result.case.final_report} caseId={result.case.case_id} />
          </section>
          <section className="panel wide">
            <h2>Band-style transcript</h2>
            <Transcript records={result.band_transcript} />
          </section>
        </div>
      ) : (
        <section className="empty">
          <h2>Run a demo case to generate the Band mesh transcript.</h2>
          <div className="demoCards">
            {demos.map((demo) => (
              <button className="demoCard" key={demo.case_id} onClick={() => runDemo(demo.case_id)}>
                <strong>{demo.title}</strong>
                <span>{demo.crop} - {demo.location}</span>
              </button>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

function Metric({ label, value }: { label: string; value: any }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Timeline({ events }: { events: EventEnvelope[] }) {
  return (
    <div className="timeline">
      {events.map((event) => (
        <article className={`event ${event.risk_level}`} key={event.event_id}>
          <div className="dot" />
          <div>
            <div className="eventHead">
              <strong>{event.agent}</strong>
              <span>{event.task_state}</span>
            </div>
            <p>{event.finding}</p>
            {event.next_agent && <small>Handoff: @{event.next_agent}</small>}
            <details>
              <summary>Payload</summary>
              <pre>{JSON.stringify(event.payload, null, 2)}</pre>
            </details>
          </div>
        </article>
      ))}
    </div>
  );
}

function Report({ report, caseId }: { report: any; caseId: string }) {
  if (!report) return <p>No report generated yet.</p>;
  return (
    <div className="report">
      <p className="summary">{report.executive_summary}</p>
      <div className="chips">
        <span>Review: {report.human_review_status}</span>
        <span>Blocked: {(report.blocked_actions || []).length}</span>
        <span>Rules: {(report.triggered_rules || []).length}</span>
      </div>
      <h3>Allowed next steps</h3>
      <ul>{(report.recommended_actions || []).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <h3>Blocked actions</h3>
      <ul>{(report.blocked_actions || ['None']).map((x: string) => <li key={x}>{x}</li>)}</ul>
      <a className="reportLink" href={`${API_BASE}/api/cases/${caseId}/report.html`} target="_blank">Open HTML audit report</a>
    </div>
  );
}

function Transcript({ records }: { records: any[] }) {
  return (
    <div className="transcript">
      {records.map((record, idx) => (
        <article key={`${record.created_at}-${idx}`}>
          <div><strong>{record.sender}</strong> <span>mentions {(record.mentions || []).join(', ') || 'none'}</span></div>
          <pre>{record.content}</pre>
        </article>
      ))}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(<App />);