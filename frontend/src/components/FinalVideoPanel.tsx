import { assetUrl } from "../api";
import type { Project } from "../types";

interface Props {
  project: Project | null;
}

export function FinalVideoPanel({ project }: Props) {
  const state = project?.state;
  const finalPath = state?.final_video_path;
  const cover = state?.cover_image_path;

  return (
    <section className="panel">
      <h2>成片</h2>
      {!project || !finalPath ? (
        <p className="muted">所有镜头完成后合成竖屏成片（含字幕烧录）。</p>
      ) : (
        <div className="final-grid">
          <div className="phone-frame">
            <video key={finalPath} src={assetUrl(project.project_id, finalPath)} controls playsInline loop />
          </div>
          <div>
            {cover && <img className="cover" src={assetUrl(project.project_id, cover)} alt="cover" />}
            <p className="muted mono">{finalPath}</p>
            <p className="muted">
              播放器可直接拖动进度条：后端以 Range 请求提供 mp4。
            </p>
            <a href={assetUrl(project.project_id, finalPath)} target="_blank" rel="noreferrer">
              下载成片
            </a>
          </div>
        </div>
      )}
    </section>
  );
}
