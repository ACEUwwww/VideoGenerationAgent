import type { AgentEvent, Project, ProjectCreateInput } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function createProject(input: ProjectCreateInput): Promise<Project> {
  const response = await fetch(`${API_BASE}/api/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return ensureOk(response);
}

export async function startAgent(projectId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/projects/${projectId}/run`, { method: "POST" });
  await ensureOk(response);
}

export async function getProject(projectId: string): Promise<Project> {
  const response = await fetch(`${API_BASE}/api/projects/${projectId}`);
  return ensureOk(response);
}

export async function getEvents(projectId: string, since: number): Promise<{ offset: number; events: AgentEvent[] }> {
  const response = await fetch(`${API_BASE}/api/projects/${projectId}/events?since=${since}`);
  return ensureOk(response);
}

export function assetUrl(projectId: string, path: string): string {
  return `${API_BASE}/api/projects/${projectId}/assets/${path}`;
}

async function ensureOk<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}
