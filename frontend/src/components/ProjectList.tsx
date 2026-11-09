import type { ProjectSummary } from "../api";

interface Props {
  projects: ProjectSummary[];
  currentId: string;
  onOpen: (projectId: string) => void;
}

const statusLabel: Record<string, string> = {
  created: "待开拍",
  running: "生成中",
  completed: "已完成",
  failed: "失败",
};

export function ProjectList({ projects, currentId, onOpen }: Props) {
  if (projects.length === 0) {
    return (
      <section className="panel">
        <h2>项目</h2>
        <p className="muted">还没有项目，提交上面的创意开始第一部。</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>
        项目 <span className="muted">({projects.length})</span>
      </h2>
      <ul className="project-list">
        {projects.map((project) => (
          <li key={project.project_id}>
            <button
              type="button"
              className={`project-row ${project.project_id === currentId ? "current" : ""}`}
              onClick={() => onOpen(project.project_id)}
            >
              <span className={`status-chip ${project.status}`}>{statusLabel[project.status] ?? project.status}</span>
              <span className="project-text">
                <strong>{project.logline || project.user_prompt}</strong>
                <small className="muted mono">{project.project_id}</small>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
