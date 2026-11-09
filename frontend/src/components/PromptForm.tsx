import { FormEvent, useState } from "react";
import type { ProjectCreateInput } from "../types";

interface Props {
  disabled: boolean;
  onStart: (input: ProjectCreateInput) => void;
}

const GENRES = ["都市情感", "悬疑", "爽文逆袭", "古装", "甜宠", "科幻"];

export function PromptForm({ disabled, onStart }: Props) {
  const [userPrompt, setUserPrompt] = useState("女孩在深夜电梯里发现镜子中的自己慢了半拍");
  const [script, setScript] = useState("");
  const [genre, setGenre] = useState("悬疑");
  const [shotCount, setShotCount] = useState(4);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onStart({
      user_prompt: userPrompt,
      script,
      genre,
      style: `电影感竖屏短剧 · ${genre}`,
      shot_count: shotCount,
    });
  }

  return (
    <form className="panel prompt-form" onSubmit={handleSubmit}>
      <div className="form-row">
        <label>
          创意 / 一句话故事
          <textarea rows={3} value={userPrompt} onChange={(e) => setUserPrompt(e.target.value)} />
        </label>
      </div>
      <div className="form-row">
        <label>
          已有剧本 / 素材（可选，会作为设定依据）
          <textarea rows={4} value={script} onChange={(e) => setScript(e.target.value)} placeholder="粘贴剧本、原著片段或人物小传..." />
        </label>
      </div>
      <div className="form-grid">
        <label>
          题材
          <select value={genre} onChange={(e) => setGenre(e.target.value)}>
            {GENRES.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </label>
        <label>
          镜头数
          <input type="number" min={1} max={12} value={shotCount} onChange={(e) => setShotCount(Number(e.target.value))} />
        </label>
        <button disabled={disabled} type="submit">
          {disabled ? "生成中..." : "开拍"}
        </button>
      </div>
    </form>
  );
}
