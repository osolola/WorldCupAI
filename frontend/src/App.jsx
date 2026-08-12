import { useEffect, useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import HeadToHead from "./components/HeadToHead";
import Replay2022 from "./components/Replay2022";
import CustomBracket from "./components/CustomBracket";
import Scoreline from "./components/Scoreline";
import TournamentOdds from "./components/TournamentOdds";
import MarketEdge from "./components/MarketEdge";
import Scorecard from "./components/Scorecard";
import { getTeams } from "./api";
import "./App.css";

const TABS = [
  { id: "head-to-head", label: "Head-to-Head" },
  { id: "scoreline", label: "Scoreline" },
  { id: "tournament-odds", label: "Tournament Odds" },
  { id: "market-edge", label: "Market Edge" },
  { id: "scorecard", label: "Scorecard" },
  { id: "replay-2022", label: "2022 Replay" },
  { id: "custom-bracket", label: "Custom Bracket" },
];

export default function App() {
  const [teams, setTeams] = useState([]);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(TABS[0].id);

  useEffect(() => {
    getTeams()
      .then(setTeams)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="app-shell">
        <Header />
        <p className="error">
          Could not reach the API: {error}. Make sure the backend is running (uvicorn backend.main:app).
        </p>
      </div>
    );
  }

  if (teams.length === 0) {
    return (
      <div className="app-shell">
        <Header />
        <p>Loading team ratings...</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Header />
      <div className="app-body">
        <Sidebar teams={teams} />
        <main className="main-content">
          <nav className="tabs">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={activeTab === tab.id ? "tab active" : "tab"}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {activeTab === "head-to-head" && <HeadToHead teams={teams} />}
          {activeTab === "scoreline" && <Scoreline teams={teams} />}
          {activeTab === "tournament-odds" && <TournamentOdds teams={teams} />}
          {activeTab === "market-edge" && <MarketEdge teams={teams} />}
          {activeTab === "scorecard" && <Scorecard />}
          {activeTab === "replay-2022" && <Replay2022 />}
          {activeTab === "custom-bracket" && <CustomBracket teams={teams} />}
        </main>
      </div>
    </div>
  );
}
