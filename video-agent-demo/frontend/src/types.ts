export type AgentEventType = "Thought" | "Action" | "Observation" | "Decision";

export interface AgentEvent {
  ts: string;
  type: AgentEventType;
  message: string;
  data: Record<string, unknown>;
}

export interface Shot {
  id: string;
  title: string;
  description: string;
  status: string;
  image_prompt: string;
  motion_prompt: string;
  image_path: string;
  video_path: string;
  image_score: number;
  video_score: number;
  feedback: string;
  image_retry_count: number;
  video_retry_count: number;
}

export interface AgentState {
  project_id: string;
  user_prompt: string;
  style: string;
  shot_count: number;
  status: string;
  story_outline: string;
  shots: Shot[];
  current_shot_index: number;
  final_video_path: string;
}

export interface Project {
  project_id: string;
  status: string;
  user_prompt: string;
  style: string;
  shot_count: number;
  state: AgentState;
}

export interface ProjectCreateInput {
  user_prompt: string;
  style: string;
  shot_count: number;
}
