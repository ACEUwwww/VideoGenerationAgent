import type { AgentEvent, Project, ProjectCreateInput, RuntimeConfig } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface ProjectSummary {
  project_id: string;
  status: string;
  user_prompt: string;
  logline: string;
  created_at: string;
  updated_at: string;
}

export async function listProjects(limit = 30): Promise<ProjectSummary[]> {
  const data = await ensureOk<{ projects: ProjectSummary[] }>(await fetch(`${API_BASE}/api/projects?limit=${limit}`));
  return data.projects;
}

export async function getConfig(): Promise<RuntimeConfig> {
  return ensureOk(await fetch(`${API_BASE}/api/config`));
}

export async function createProject(input: ProjectCreateInput): Promise<Project> {
  return ensureOk(
    await fetch(`${API_BASE}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function startAgent(projectId: string): Promise<void> {
  await ensureOk(await fetch(`${API_BASE}/api/projects/${projectId}/run`, { method: "POST" }));
}

export async function getProject(projectId: string): Promise<Project> {
  return ensureOk(await fetch(`${API_BASE}/api/projects/${projectId}`));
}

export async function getEvents(projectId: string, since: number): Promise<{ offset: number; events: AgentEvent[] }> {
  return ensureOk(await fetch(`${API_BASE}/api/projects/${projectId}/events?since=${since}`));
}

export function assetUrl(projectId: string, path: string): string {
  return `${API_BASE}/api/projects/${projectId}/assets/${path}`;
}

async function ensureOk<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
