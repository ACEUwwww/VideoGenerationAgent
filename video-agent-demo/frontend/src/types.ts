export type AgentEventType = "Thought" | "Action" | "Observation" | "Decision";

export interface AgentEvent {
  ts: string;
  type: AgentEventType;
  message: string;
  data: Record<string, unknown>;
}

export interface SubtitleLine {
  text: string;
  start_s: number;
  end_s: number;
}

export interface Shot {
  id: string;
  index: number;
  status: string;
  scene: string;
  shot_size: string;
  camera: string;
  description: string;
  dialogue: string;
  speaker: string;
  emotion: string;
  sfx: string;
  transition: string;
  duration_s: number;
  characters: string[];
  image_prompt: string;
  negative_prompt: string;
  motion_prompt: string;
  image_path: string;
  video_path: string;
  audio_path: string;
  keyframe_path?: string;
  image_score: number;
  video_score: number;
  feedback: string;
  image_retry_count: number;
  video_retry_count: number;
  dims?: Record<string, number>;
  multidim_score?: number;
  drift_vs_anchor?: { color: number; structure: number } | null;
  subtitle_lines?: SubtitleLine[];
}

export interface Character {
  id: string;
  name: string;
  role: string;
  age_range: string;
  persona: string;
  appearance: string;
  wardrobe: string;
  voice: string;
}

export interface DimensionScore {
  // null when the dimension is not applicable to this comparison (e.g. a clip
  // whose framing differs from the anchor: whole-frame structure is not
  // comparable, so the dimension is excluded from the weighted average).
  score: number | null;
  worst?: number;
  weight: number;
  description: string;
  samples?: number;
}

export interface EpisodeReport {
  score: number;
  dimensions: Record<string, DimensionScore>;
  clips: { shot_id: string; score: number; passed: boolean }[];
}

export interface KeyframeMemory {
  anchor: string;
  entries: { shot_id: string; path: string; score: number; drift_vs_anchor: unknown }[];
}

export interface AgentState {
  project_id: string;
  user_prompt: string;
  script?: string;
  genre?: string;
  style: string;
  shot_count: number;
  status: string;
  logline: string;
  tone: string;
  setting: string;
  outline: string;
  hook: string;
  ending: string;
  characters: Character[];
  shots: Shot[];
  current_shot_index: number;
  final_video_path: string;
  cover_image_path: string;
  subtitle_path: string;
  episode_report: EpisodeReport;
  keyframe_memory: KeyframeMemory;
  usage?: { calls?: number; prompt_tokens?: number; completion_tokens?: number };
  error?: string;
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
  script: string;
  genre: string;
  style: string;
  shot_count: number;
}

export interface RuntimeConfig {
  providers: Record<string, string>;
  llm_model: string;
  canvas: { width: number; height: number; fps: number };
  video_bench_dimensions: string[];
}
