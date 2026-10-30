import { assetUrl } from "../api";
import type { Shot } from "../types";

interface Props {
  projectId: string;
  shot: Shot;
  active: boolean;
  onSelect: () => void;
}

export function ShotCard({ projectId, shot, active, onSelect }: Props) {
  const dims = shot.dims ?? {};
  return (
    <article className={`shot-card ${active ? "active" : ""}`} onClick={onSelect}>
      <div className="shot-media">
        {shot.image_path ? (
          <img src={assetUrl(projectId, shot.image_path)} alt={shot.id} />
        ) : (
          <div className="placeholder">待生成</div>
        )}
        <span className="duration">{shot.duration_s?.toFixed(1)}s</span>
        {shot.multidim_score ? <span className="score-badge">{shot.multidim_score.toFixed(2)}</span> : null}
      </div>
      <div className="shot-body">
        <header>
          <strong>{shot.id}</strong>
          <span className="tag">{shot.shot_size}</span>
          <span className="status-dot" data-status={shot.status} />
        </header>
        <p>{shot.description}</p>
        {shot.dialogue && (
          <p className="dialogue">
            「{shot.dialogue}」{shot.emotion && <span className="muted"> · {shot.emotion}</span>}
          </p>
        )}
        {Object.keys(dims).length > 0 && (
          <ul className="dim-list">
            {Object.entries(dims).map(([name, score]) => (
              <li key={name}>
                <span>{name.replace(/_/g, " ")}</span>
                <span className={score >= 0.62 ? "ok" : "warn"}>{(score * 100).toFixed(0)}</span>
              </li>
            ))}
          </ul>
        )}
        {shot.drift_vs_anchor && (
          <small className="muted">
            与首镜锚点漂移：结构 {(1 - shot.drift_vs_anchor.structure).toFixed(2)} · 色彩{" "}
            {(1 - shot.drift_vs_anchor.color).toFixed(2)}
          </small>
        )}
      </div>
    </article>
  );
}
