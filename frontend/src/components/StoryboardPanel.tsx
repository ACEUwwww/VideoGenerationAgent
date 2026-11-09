import type { Shot } from "../types";
import { ShotCard } from "./ShotCard";

interface Props {
  projectId: string;
  shots: Shot[];
  selected: string;
  onSelect: (id: string) => void;
}

export function StoryboardPanel({ projectId, shots, selected, onSelect }: Props) {
  return (
    <section className="panel">
      <h2>分镜表 <span className="muted">({shots.length})</span></h2>
      <div className="shot-grid">
        {shots.length === 0 && <p className="muted">等待 Agent 规划分镜...</p>}
        {shots.map((shot) => (
          <ShotCard
            key={shot.id}
            projectId={projectId}
            shot={shot}
            active={shot.id === selected}
            onSelect={() => onSelect(shot.id)}
          />
        ))}
      </div>
    </section>
  );
}
