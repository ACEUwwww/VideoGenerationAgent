import { assetUrl } from "../api";
import type { Project } from "../types";

interface Props {
  project: Project | null;
}

export function MediaPreview({ project }: Props) {
  const state = project?.state;
  const currentShot = state?.shots[state.current_shot_index] ?? state?.shots.at(-1);

  return (
    <section className="panel">
      <h2>Current Preview</h2>
      {!project || !currentShot ? (
        <p className="muted">暂无媒体结果。</p>
      ) : (
        <>
          <h3>{currentShot.title}</h3>
          {currentShot.image_path && <img src={assetUrl(project.project_id, currentShot.image_path)} alt={currentShot.title} />}
          {currentShot.video_path && (
            <div className="mock-video">
              Mock video file: <a href={assetUrl(project.project_id, currentShot.video_path)}>{currentShot.video_path}</a>
            </div>
          )}
        </>
      )}
    </section>
  );
}
