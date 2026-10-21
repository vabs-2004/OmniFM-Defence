import React, { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { extractApiError, omnifmApi } from "../services/api";

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [selectedExpId, setSelectedExpId] = useState("");
  const [selectedTask, setSelectedTask] = useState("classification");
  const [jobStatus, setJobStatus] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [metricsHistory, setMetricsHistory] = useState({
    global_history: [],
    client_history: [],
  });
  const [auditEvents, setAuditEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadDashboard = async (expIdOverride, taskOverride) => {
    try {
      setError("");
      const activeTask = taskOverride ?? selectedTask;
      const [hRes, hospRes, expRes, auditRes] = await Promise.all([
        omnifmApi.getHealth(),
        omnifmApi.getHospitals(),
        omnifmApi.getExperiments({ limit: 50 }),
        omnifmApi.getBlockchainLog({ limit: 8 }),
      ]);
      setHealth(hRes);
      setHospitals(hospRes.clients || []);
      const expItems = expRes.items || [];
      setExperiments(expItems);
      setAuditEvents(auditRes.items || []);

      const targetExpId =
        expIdOverride !== undefined
          ? expIdOverride
          : selectedExpId || (expItems[0]?.experiment_id ?? "");
      if (targetExpId && targetExpId !== selectedExpId) {
        setSelectedExpId(targetExpId);
      }

      if (targetExpId) {
        const [stRes, mRes] = await Promise.all([
          omnifmApi.getTrainingStatus(targetExpId),
          omnifmApi.getMetricsHistory({ training_id: targetExpId }),
        ]);
        setJobStatus(stRes);
        setMetricsHistory(mRes);
      } else {
        const mRes = await omnifmApi.getMetricsHistory({ task_type: activeTask });
        setJobStatus(null);
        setMetricsHistory(mRes);
      }
    } catch (err) {
      setError(extractApiError(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  const handleExperimentChange = (e) => {
    const nextId = e.target.value;
    setSelectedExpId(nextId);
    const matched = experiments.find((x) => x.experiment_id === nextId);
    if (matched) {
      setSelectedTask(matched.task_type);
    }
    loadDashboard(nextId, matched?.task_type);
  };

  const handleTaskChange = (e) => {
    const nextTask = e.target.value;
    setSelectedTask(nextTask);
    const firstForTask = experiments.find((x) => x.task_type === nextTask);
    const nextExpId = firstForTask ? firstForTask.experiment_id : "";
    setSelectedExpId(nextExpId);
    loadDashboard(nextExpId, nextTask);
  };

  const globalChartData = (metricsHistory.global_history || []).map((row) => ({
    round: `R${row.round_num}`,
    round_num: row.round_num,
    ...(row.metrics || {}),
  }));

  const clientList =
    jobStatus?.clients?.length > 0 ? jobStatus.clients : hospitals;

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-white">
              OmniFM Federated Consortium Dashboard
            </h1>
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700">
              LIVE API DATA
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Sequential 3-Hospital Simulation • SpectralEncoder + GSKR + ECA + PSP + SPAlign • SHA-256 Blockchain Provenance
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Task Filter</label>
            <select
              value={selectedTask}
              onChange={handleTaskChange}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100"
            >
              <option value="classification">Classification (MedMNIST)</option>
              <option value="segmentation">Segmentation (FeTS2022 MRI)</option>
              <option value="superresolution">Super-Resolution (BreaKHis)</option>
              <option value="fusion">Multimodal Fusion (CT/PET/SPECT-MRI)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">Experiment Run</label>
            <select
              value={selectedExpId}
              onChange={handleExperimentChange}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-100 min-w-[220px]"
            >
              <option value="">-- Latest for Task --</option>
              {experiments.map((exp) => (
                <option key={exp.experiment_id} value={exp.experiment_id}>
                  {exp.experiment_id} ({exp.task_type} • {exp.status})
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => loadDashboard(selectedExpId, selectedTask)}
            className="mt-5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-950/70 border border-rose-700 text-rose-200 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Resource & Job Status Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">Active Experiment</div>
          <div className="text-lg font-semibold text-white mt-1 truncate">
            {jobStatus?.experiment_id || "No Experiment Selected"}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Status:{" "}
            <span className="font-semibold text-indigo-300">
              {jobStatus?.status || "IDLE"}
            </span>{" "}
            • Round {jobStatus?.current_round ?? 0}/{jobStatus?.total_rounds ?? 0}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">GPU Hardware Status</div>
          <div className="text-lg font-semibold text-white mt-1 truncate">
            {health?.gpu?.device_name || (health?.gpu?.cuda_available ? "CUDA GPU" : "CPU Mode")}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            VRAM Total: {health?.gpu?.total_vram_mb ? `${health.gpu.total_vram_mb} MB` : "N/A"} • Peak Run:{" "}
            {jobStatus?.peak_vram_mb ? `${jobStatus.peak_vram_mb.toFixed(1)} MB` : "0.0 MB"}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">Blockchain Node</div>
          <div className="text-lg font-semibold text-white mt-1">
            {health?.blockchain?.connected ? "Connected (Chain 1337)" : "Offline / Local SQLite Audit"}
          </div>
          <div className="text-xs text-slate-400 mt-1 truncate">
            Contract: {health?.blockchain?.contract_address || "Not configured"}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">SQLite Persistence</div>
          <div className="text-lg font-semibold text-emerald-400 mt-1">
            {health?.database?.connected ? "Connected & Verified" : "Disconnected"}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            Stored Runs: {experiments.length} • Audit Records: {auditEvents.length}
          </div>
        </div>
      </div>

      {/* Three Hospital Client Cards */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">
          Participating Hospital Nodes (Sequential Execution)
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {clientList.map((c) => {
            const upd = c.latest_update;
            return (
              <div
                key={c.client_id}
                className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-white">{c.hospital_name}</span>
                  <span className="px-2 py-0.5 text-xs rounded bg-slate-800 text-indigo-300 font-mono">
                    {c.client_id}
                  </span>
                </div>
                <div className="text-xs text-slate-400">{c.modality}</div>
                <div className="text-xs font-mono text-slate-500 truncate">
                  Wallet: {c.wallet_address || "Unassigned"}
                </div>
                {upd ? (
                  <div className="pt-2 border-t border-slate-800 text-xs space-y-1">
                    <div className="flex justify-between text-slate-300">
                      <span>Latest Round: R{upd.round_num}</span>
                      <span>Samples: {upd.num_samples}</span>
                    </div>
                    <div className="flex justify-between text-slate-300">
                      <span>Task Loss: {Number(upd.task_loss).toFixed(4)}</span>
                      <span>SPAlign: {Number(upd.align_loss).toFixed(4)}</span>
                    </div>
                    <div className="text-slate-400 font-mono truncate">
                      Hash: {upd.update_hash}
                    </div>
                  </div>
                ) : (
                  <div className="pt-2 border-t border-slate-800 text-xs text-slate-500">
                    No client updates recorded for current selection.
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Global Metrics Chart & Recent Audit Events */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h3 className="text-base font-semibold text-white mb-1">
            Measured Global Round Metrics
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Displays only real metrics persisted in SQLite (never synthetic/mock curves).
          </p>
          {loading ? (
            <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
              Loading measured metrics...
            </div>
          ) : globalChartData.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg p-4 text-center">
              No measured evaluation rounds recorded for this selection yet. Start a run in Training Control to populate live charts.
            </div>
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={globalChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="round" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      fontSize: "12px",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke="#f43f5e"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    name="Accuracy"
                    stroke="#10b981"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="dice"
                    name="Dice"
                    stroke="#38bdf8"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="ssim"
                    name="SSIM"
                    stroke="#a855f7"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h3 className="text-base font-semibold text-white mb-1">
            Recent Blockchain & Provenance Audit Feed
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Off-chain SHA-256 artifact hashes & on-chain transaction inclusion status.
          </p>
          {auditEvents.length === 0 ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg">
              No blockchain audit events recorded yet.
            </div>
          ) : (
            <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
              {auditEvents.map((ev) => (
                <div
                  key={ev.id}
                  className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs flex flex-col gap-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-indigo-300">
                      {ev.action_type} • R{ev.round_num ?? 0}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded font-medium ${
                        ev.status === "CONFIRMED"
                          ? "bg-emerald-950 text-emerald-300"
                          : "bg-amber-950 text-amber-300"
                      }`}
                    >
                      {ev.status}
                    </span>
                  </div>
                  <div className="text-slate-400">
                    Exp: <span className="font-mono">{ev.experiment_id || "N/A"}</span>{" "}
                    {ev.client_id ? `• Client: ${ev.client_id}` : ""}
                  </div>
                  {ev.artifact_hash && (
                    <div className="font-mono text-slate-500 truncate">
                      SHA-256: {ev.artifact_hash}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
