import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createProject, getConfig, getEvents, getProject, listProjects, startAgent } from "./api";
import type { ProjectSummary } from "./api";
import { ProjectList } from "./components/ProjectList";
import { AgentLogPanel } from "./components/AgentLogPanel";
import { CastPanel } from "./components/CastPanel";
import { FinalVideoPanel } from "./components/FinalVideoPanel";
import { MediaPreview } from "./components/MediaPreview";
import { PromptForm } from "./components/PromptForm";
import { ReportPanel } from "./components/ReportPanel";
import { StoryboardPanel } from "./components/StoryboardPanel";
import type { AgentEvent, Project, ProjectCreateInput, RuntimeConfig } from "./types";
import "./style.css";

function App() {
  const [project, setProject] = useState<Project | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const eventOffset = useRef(0);

  const running = project?.state.status === "running" || isStarting;
  const shots = project?.state.shots ?? [];
  const activeShot = useMemo(
    () => shots.find((shot) => shot.id === selected) ?? shots[shots.length - 1] ?? null,
    [shots, selected],
  );

  useEffect(() => {
    getConfig().then(setConfig).catch(() => undefined);
  }, []);

  const refreshProjects = useCallback(() => {
    listProjects().then(setProjects).catch(() => undefined);
  }, []);

  useEffect(() => {
    refreshProjects();
  }, [refreshProjects]);

  function openProject(projectId: string) {
    setError("");
    setEvents([]);
    setSelected("");
    eventOffset.current = 0;
    getProject(projectId).then(setProject).catch((err) => setError(String(err)));
  }

  async function handleStart(input: ProjectCreateInput) {
    setError("");
    setIsStarting(true);
    setEvents([]);
    setSelected("");
    eventOffset.current = 0;
    try {
      const created = await createProject(input);
      setProject(created);
      await startAgent(created.project_id);
      refreshProjects();
    } catch (err) {
      setError(err instanceof Error ? err.message : "启动失败");
    } finally {
      setIsStarting(false);
    }
  }

  useEffect(() => {
    if (!project?.project_id) return;
    const timer = window.setInterval(async () => {
      try {
        const [nextProject, nextEvents] = await Promise.all([
          getProject(project.project_id),
          getEvents(project.project_id, eventOffset.current),
        ]);
        setProject(nextProject);
        eventOffset.current = nextEvents.offset;
        if (nextProject.state.status !== "running") refreshProjects();
        if (nextEvents.events.length > 0) {
          setEvents((current) => [...current, ...nextEvents.events]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "刷新失败");
      }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [project?.project_id, refreshProjects]);

  const status = project?.state.status ?? "idle";

  return (
    <main className="app">
      <header>
        <div>
          <p className="eyebrow">LangGraph · DeepSeek · 火山方舟 · Video-Bench 维度</p>
          <h1>竖屏短剧生成 Agent</h1>
        </div>
        <div className="header-right">
          <span className={`status ${status}`}>{status}</span>
          {config && (
            <span className="muted small">
              {config.llm_model} · image:{config.providers.image} · video:{config.providers.video} ·{" "}
              {config.canvas.width}×{config.canvas.height}
            </span>
          )}
        </div>
      </header>

      <PromptForm disabled={running} onStart={handleStart} />
      {error && <p className="error">{error}</p>}
      {project?.state.error && <p className="error">{project.state.error}</p>}

      <section className="layout">
        <div className="col main-col">
          <CastPanel
            characters={project?.state.characters ?? []}
            logline={project?.state.logline ?? ""}
            setting={project?.state.setting ?? ""}
            outline={project?.state.outline ?? ""}
          />
          <StoryboardPanel
            projectId={project?.project_id ?? ""}
            shots={shots}
            selected={activeShot?.id ?? ""}
            onSelect={setSelected}
          />
          <ReportPanel
            report={project?.state.episode_report}
            memory={project?.state.keyframe_memory}
            usage={project?.state.usage}
          />
          <FinalVideoPanel project={project} />
        </div>
        <div className="col side-col">
          <MediaPreview projectId={project?.project_id ?? ""} shot={activeShot} />
          <ProjectList projects={projects} currentId={project?.project_id ?? ""} onOpen={openProject} />
          <AgentLogPanel events={events} />
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
