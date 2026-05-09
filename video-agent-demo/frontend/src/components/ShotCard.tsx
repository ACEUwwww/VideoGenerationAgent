import { assetUrl } from "../api";
import type { Shot } from "../types";

interface Props {
  projectId: string;
  shot: Shot;
}

export function ShotCard({ projectId, shot }: Props) {
  return (
    <article className="shot-card">
      <div className="shot-header">
        <strong>{shot.title}</strong>
        <span>{shot.status}</span>
      </div>
      <p>{shot.description}</p>
      <dl>
        <dt>Image</dt>
        <dd>{shot.image_path || "-"}</dd>
        <dt>Video</dt>
        <dd>{shot.video_path || "-"}</dd>
        <dt>Score</dt>
        <dd>
          image {shot.image_score || 0} / video {shot.video_score || 0}
        </dd>
      </dl>
      {shot.image_path && <img src={assetUrl(projectId, shot.image_path)} alt={shot.title} />}
      {shot.feedback && <small>{shot.feedback}</small>}
    </article>
  );
}
