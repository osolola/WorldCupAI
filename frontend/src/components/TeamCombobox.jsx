import { useEffect, useId, useMemo, useRef, useState } from "react";

export default function TeamCombobox({ label, teams, value, onChange }) {
  const inputId = useId();
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const containerRef = useRef(null);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = q ? teams.filter((t) => t.name.toLowerCase().includes(q)) : teams;
    return pool.slice(0, 8);
  }, [teams, query]);

  const isValid = teams.some((t) => t.name === query);

  useEffect(() => {
    function handleClickOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
        setQuery(value);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [value]);

  function selectTeam(name) {
    onChange(name);
    setQuery(name);
    setOpen(false);
  }

  function handleKeyDown(e) {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlight((h) => Math.min(h + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (matches[highlight]) selectTeam(matches[highlight].name);
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery(value);
    }
  }

  return (
    <div className="combobox" ref={containerRef}>
      <label className="combobox-label" htmlFor={inputId}>
        {label}
      </label>
      <input
        id={inputId}
        className={isValid ? "combobox-input" : "combobox-input invalid"}
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setHighlight(0);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        placeholder="Search team"
        autoComplete="off"
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
      />
      {!isValid && <span className="combobox-hint">Pick a team from the list</span>}
      {open && matches.length > 0 && (
        <ul className="combobox-panel" role="listbox">
          {matches.map((t, i) => (
            <li
              key={t.name}
              role="option"
              aria-selected={i === highlight}
              className={i === highlight ? "active" : ""}
              onMouseDown={() => selectTeam(t.name)}
              onMouseEnter={() => setHighlight(i)}
            >
              <span>{t.name}</span>
              <span className="combobox-elo">{t.elo}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
