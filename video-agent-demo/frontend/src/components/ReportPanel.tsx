import type { EpisodeReport, KeyframeMemory } from "../types";

interface Props {
  report?: EpisodeReport;
  memory?: KeyframeMemory;
  usage?: { calls?: number; prompt_tokens?: number; completion_tokens?: number };
}

export function ReportPanel({ report, memory, usage }: Props) {
  const dims = report?.dimensions ?? {};
  const entries = memory?.entries ?? [];

  return (
    <section className="panel">
      <h2>
        Video-Bench 多维评估 {report?.score ? <span className="score-big">{report.score.toFixed(2)}</span> : null}
      </h2>
      {Object.keys(dims).length === 0 ? (
        <p className="muted">片段生成后按 7 个维度打分（借鉴 Video-Bench 维度体系，本地像素测量实现）。</p>
      ) : (
        <ul className="bar-list">
          {Object.entries(dims).map(([name, entry]) => (
            <li key={name} title={entry.description}>
              <span className="bar-label">{name.replace(/_/g, " ")}</span>
              {entry.score === null ? (
                <>
                  <span className="bar-track bar-na" title="景别与锚点不同，整帧结构不可比">
                    <span className="bar-na-label">不适用</span>
                  </span>
                  <span className="bar-value muted">—</span>
                  <span className="muted small">跨景别跳过</span>
                </>
              ) : (
                <>
                  <span className="bar-track">
                    <span
                      className={`bar-fill ${entry.score >= 0.62 ? "ok" : "warn"}`}
                      style={{ width: `${entry.score * 100}%` }}
                    />
                  </span>
                  <span className="bar-value">{entry.score.toFixed(2)}</span>
                  <span className="muted small">最差 {entry.worst?.toFixed(2)}</span>
                </>
              )}
            </li>
          ))}
        </ul>
      )}

      {entries.length > 0 && (
        <>
          <h3>关键帧记忆</h3>
          <p className="muted small">首镜关键帧作为全片视觉锚点，后续镜头与锚点的漂移量（越低越好）。</p>
          <table className="memory-table">
            <thead>
              <tr>
                <th>镜头</th>
                <th>关键帧</th>
                <th>结构漂移</th>
                <th>色彩漂移</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => {
                const drift = entry.drift_vs_anchor as { color: number; structure: number } | null;
                return (
                  <tr key={entry.shot_id}>
                    <td>{entry.shot_id}</td>
                    <td className="mono small">{entry.path}</td>
                    <td>{drift ? (1 - drift.structure).toFixed(3) : "锚点"}</td>
                    <td>{drift ? (1 - drift.color).toFixed(3) : "锚点"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {usage?.calls ? (
        <p className="muted small">
          LLM 调用 {usage.calls} 次 · 输入 {usage.prompt_tokens} tokens · 输出 {usage.completion_tokens} tokens
        </p>
      ) : null}
    </section>
  );
}
