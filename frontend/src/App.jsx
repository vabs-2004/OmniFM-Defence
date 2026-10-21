import React from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import BlockchainExplorer from "./pages/BlockchainExplorer";
import Classification from "./pages/Classification";
import Dashboard from "./pages/Dashboard";
import Fusion from "./pages/Fusion";
import Segmentation from "./pages/Segmentation";
import SuperResolution from "./pages/SuperResolution";
import TrainingControl from "./pages/TrainingControl";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard" },
  { to: "/classify", label: "Classification" },
  { to: "/segment", label: "Segmentation" },
  { to: "/superres", label: "Super-Resolution" },
  { to: "/fusion", label: "Multimodal Fusion" },
  { to: "/blockchain", label: "Blockchain Explorer" },
  { to: "/train", label: "Training Control" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white">
              Ω
            </div>
            <div>
              <span className="font-bold text-white tracking-tight">
                OmniFM Blockchain
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs text-slate-400">
                Frequency-Aware Federated Medical AI
              </span>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                    isActive
                      ? "bg-indigo-600 text-white"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/classify" element={<Classification />} />
          <Route path="/segment" element={<Segmentation />} />
          <Route path="/superres" element={<SuperResolution />} />
          <Route path="/fusion" element={<Fusion />} />
          <Route path="/blockchain" element={<BlockchainExplorer />} />
          <Route path="/train" element={<TrainingControl />} />
        </Routes>
      </main>

      <footer className="border-t border-slate-900 py-4 text-center text-xs text-slate-500">
        OmniFM Research Prototype • Raw patient images remain local • SHA-256 provenance on Ethereum / Ganache (Chain ID 1337)
      </footer>
    </div>
  );
}
