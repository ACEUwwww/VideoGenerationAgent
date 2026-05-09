import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createProject, getEvents, getProject, startAgent } from "./api";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { FinalVideoPanel } from "./components/FinalVideoPanel";
import { MediaPreview } from "./components/MediaPreview";
import { PromptForm } from "./components/PromptForm";
import { StoryboardPanel } from "./components/StoryboardPanel";
import type { AgentEvent, Project, ProjectCreateInput } from "./types";
import "./style.css";

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [error, setError] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const eventOffset = useRef(0);

  const running = project?.state.status === "running" || isStarting;

  async function handleStart(input: ProjectCreateInput) {
    setError("");
    setIsStarting(true);
    setEvents([]);
    eventOffset.current = 0;
    try {
      const created = await createProject(input);
      setProject(created);
      await startAgent(created.project_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "启动失败");
    } finally {
      setIsStarting(false);
    }
  }

  useEffect(() => {
    if (!project?.project_id) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const [nextProject, nextEvents] = await Promise.all([
          getProject(project.project_id),
          getEvents(project.project_id, eventOffset.current),
        ]);
        setProject(nextProject);
        eventOffset.current = nextEvents.offset;
        if (nextEvents.events.length > 0) {
          setEvents((current) => [...current, ...nextEvents.events]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "刷新失败");
      }
    }, 900);

    return () => window.clearInterval(timer);
  }, [project?.project_id]);

  return (
    <main className="app">
      <header>
        <div>
          <p className="eyebrow">LangGraph + MCP-like Tools</p>
          <h1>Lightweight Video Generation Agent Demo</h1>
        </div>
        <span className={`status ${project?.state.status ?? "idle"}`}>{project?.state.status ?? "idle"}</span>
      </header>

      <PromptForm disabled={running} onStart={handleStart} />
      {error && <p className="error">{error}</p>}

      <section className="layout">
        <div className="left">
          <StoryboardPanel projectId={project?.project_id ?? ""} shots={project?.state.shots ?? []} />
          <FinalVideoPanel project={project} />
        </div>
        <div className="right">
          <MediaPreview project={project} />
          <AgentLogPanel events={events} />
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
