import { assetUrl } from "../api";
import type { Shot } from "../types";

interface Props {
  projectId: string;
  shot: Shot | null;
}

export function MediaPreview({ projectId, shot }: Props) {
  return (
    <section className="panel preview-panel">
      <h2>镜头预览</h2>
      {!shot ? (
        <p className="muted">暂无媒体结果。</p>
      ) : (
        <div className="phone-frame">
          {shot.video_path ? (
            <video
              key={shot.video_path}
              src={assetUrl(projectId, shot.video_path)}
              controls
              playsInline
              loop
              muted
            />
          ) : shot.image_path ? (
            <img src={assetUrl(projectId, shot.image_path)} alt={shot.id} />
          ) : (
            <div className="placeholder">待生成</div>
          )}
          {shot.dialogue && <p className="overlay-subtitle">{shot.dialogue}</p>}
        </div>
      )}
      {shot && (
        <dl className="preview-meta">
          <dt>景别 / 运镜</dt>
          <dd>{shot.shot_size} · {shot.camera}</dd>
          <dt>场景</dt>
          <dd>{shot.scene}</dd>
          <dt>关键帧提示词</dt>
          <dd className="mono">{shot.image_prompt || "-"}</dd>
          <dt>运动提示词</dt>
          <dd className="mono">{shot.motion_prompt || "-"}</dd>
        </dl>
      )}
    </section>
  );
}
