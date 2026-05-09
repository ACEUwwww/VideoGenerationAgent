import { FormEvent, useState } from "react";
import type { ProjectCreateInput } from "../types";

interface Props {
  disabled: boolean;
  onStart: (input: ProjectCreateInput) => void;
}

export function PromptForm({ disabled, onStart }: Props) {
  const [userPrompt, setUserPrompt] = useState("一只小狐狸穿越未来城市，寻找会发光的星星。");
  const [style, setStyle] = useState("cinematic, warm light, storybook");
  const [shotCount, setShotCount] = useState(3);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    onStart({ user_prompt: userPrompt, style, shot_count: shotCount });
  }

  return (
    <form className="panel prompt-form" onSubmit={handleSubmit}>
      <label>
        故事需求
        <textarea value={userPrompt} onChange={(event) => setUserPrompt(event.target.value)} />
      </label>
      <label>
        视频风格
        <input value={style} onChange={(event) => setStyle(event.target.value)} />
      </label>
      <label>
        镜头数量
        <input
          type="number"
          min={1}
          max={8}
          value={shotCount}
          onChange={(event) => setShotCount(Number(event.target.value))}
        />
      </label>
      <button disabled={disabled} type="submit">
        {disabled ? "Agent Running..." : "Start Agent"}
      </button>
    </form>
  );
}
