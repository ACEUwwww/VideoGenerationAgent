import type { AgentEvent } from "../types";

interface Props {
  events: AgentEvent[];
}

const typeClass: Record<string, string> = {
  Thought: "thought",
  Action: "action",
  Observation: "observation",
  Decision: "decision",
};

export function AgentLogPanel({ events }: Props) {
  return (
    <section className="panel log-panel">
      <h2>Agent Logs</h2>
      {events.length === 0 && <p className="muted">等待 Thought / Action / Observation / Decision...</p>}
      <div className="logs">
        {events.map((event, index) => (
          <article className={`log-item ${typeClass[event.type] ?? ""}`} key={`${event.ts}-${index}`}>
            <strong>{event.type}</strong>
            <span>{new Date(event.ts).toLocaleTimeString()}</span>
            <p>{event.message}</p>
            {Object.keys(event.data ?? {}).length > 0 && <pre>{JSON.stringify(event.data, null, 2)}</pre>}
          </article>
        ))}
      </div>
    </section>
  );
}
