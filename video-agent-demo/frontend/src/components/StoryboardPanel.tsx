import type { Shot } from "../types";
import { ShotCard } from "./ShotCard";

interface Props {
  projectId: string;
  shots: Shot[];
}

export function StoryboardPanel({ projectId, shots }: Props) {
  return (
    <section className="panel">
      <h2>Storyboard</h2>
      <div className="shot-grid">
        {shots.length === 0 && <p className="muted">等待 Agent 规划分镜...</p>}
        {shots.map((shot) => (
          <ShotCard key={shot.id} projectId={projectId} shot={shot} />
        ))}
      </div>
    </section>
  );
}
