import { useMemo, useState } from "react";

export default function Sidebar({ teams }) {
  const [search, setSearch] = useState("");

  const matches = useMemo(() => {
    if (!search) return teams;
    const q = search.toLowerCase();
    return teams.filter((t) => t.name.toLowerCase().includes(q));
  }, [teams, search]);

  return (
    <aside className="sidebar">
      <h2>Team Lookup</h2>
      <input
        type="text"
        placeholder="Search team"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <ul className="team-list">
        {matches.map((t) => (
          <li key={t.name}>
            <span className="team-name">{t.name}</span>
            <span className="team-elo">{t.elo}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
