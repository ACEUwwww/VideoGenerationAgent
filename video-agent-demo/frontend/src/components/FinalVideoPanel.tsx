import { assetUrl } from "../api";
import type { Project } from "../types";

interface Props {
  project: Project | null;
}

export function FinalVideoPanel({ project }: Props) {
  const finalPath = project?.state.final_video_path;

  return (
    <section className="panel">
      <h2>Final Video</h2>
      {!project || !finalPath ? (
        <p className="muted">最终视频将在合成后显示。</p>
      ) : (
        <div className="final-video">
          <p>Mock final video: {finalPath}</p>
          <a href={assetUrl(project.project_id, finalPath)} target="_blank">
            打开生成文件
          </a>
        </div>
      )}
    </section>
  );
}
