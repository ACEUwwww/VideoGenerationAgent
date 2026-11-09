import type { Character } from "../types";

interface Props {
  characters: Character[];
  logline: string;
  setting: string;
  outline: string;
}

export function CastPanel({ characters, logline, setting, outline }: Props) {
  if (!characters.length && !logline) {
    return (
      <section className="panel">
        <h2>设定集</h2>
        <p className="muted">等待 Agent 生成故事大纲与角色设定...</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>设定集</h2>
      {logline && <p className="logline">{logline}</p>}
      {setting && <p className="muted">{setting}</p>}
      <div className="cast-grid">
        {characters.map((character) => (
          <article key={character.id} className="cast-card">
            <header>
              <strong>{character.name}</strong>
              <span className="tag">{character.role}</span>
            </header>
            <p className="muted">{character.age_range} · {character.persona}</p>
            <p className="anchor">
              <span className="tag alt">外观锚点</span> {character.appearance}
            </p>
            {character.wardrobe && <p className="muted">服装：{character.wardrobe}</p>}
          </article>
        ))}
      </div>
      {outline && (
        <details>
          <summary>大纲</summary>
          <p>{outline}</p>
        </details>
      )}
    </section>
  );
}
