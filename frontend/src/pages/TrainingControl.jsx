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

export default function TrainingControl() {
  const [taskType, setTaskType] = useState("classification");
  const [algorithm, setAlgorithm] = useState("omnifm");
  const [executionMode, setExecutionMode] = useState("single_laptop");
  const [numRounds, setNumRounds] = useState(5);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(4);
  const [maxSamples, setMaxSamples] = useState(16);
  const [scaleFactor, setScaleFactor] = useState(2);
  const [ablations, setAblations] = useState({
    enable_gskr: true,
    enable_eca: true,
    enable_psp: true,
    enable_spalign: true,
  });

  const [activeId, setActiveId] = useState("");
  const [statusData, setStatusData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const pollStatus = async (targetId) => {
    const idToQuery = targetId ?? activeId;
    try {
      const st = await omnifmApi.getTrainingStatus(idToQuery || undefined);
      setStatusData(st);
      if (st?.training_id) {
        if (!activeId) setActiveId(st.training_id);
        const [logRes, mRes] = await Promise.all([
          omnifmApi.getTrainingLogs(st.training_id, { limit: 100, offset: 0 }),
          omnifmApi.getMetricsHistory({ training_id: st.training_id }),
        ]);
        setLogs(logRes.logs || []);
        setMetricsHistory(mRes.global_history || []);
      }
    } catch (err) {
      // Ignore initial 404 if no experiment exists yet
    }
  };

  useEffect(() => {
    pollStatus();
    const interval = setInterval(() => {
      pollStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, [activeId]);

  const handleStart = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const modeConfig =
        executionMode === "distributed_two_laptop"
          ? {
              client_ids: ["client_1", "client_2"],
              remote_client_ids: ["client_2"],
            }
          : {};
      const res = await omnifmApi.startTraining({
        task_type: taskType,
        num_rounds: Number(numRounds),
        run_async: true,
        config: {
          algorithm,
          local_epochs: Number(localEpochs),
          batch_size: Number(batchSize),
          max_samples: Number(maxSamples),
          scale_factor: Number(scaleFactor),
          ...modeConfig,
          ...ablations,
        },
      });
      setActiveId(res.training_id);
      await pollStatus(res.training_id);
    } catch (err) {
      setError(extractApiError(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = async () => {
    if (!statusData?.training_id) return;
    setError("");
    try {
      await omnifmApi.cancelTraining(statusData.training_id);
      await pollStatus(statusData.training_id);
    } catch (err) {
      setError(extractApiError(err));
    }
  };

  const handleResume = async () => {
    if (!statusData?.training_id) return;
    setError("");
    try {
      await omnifmApi.resumeTraining(statusData.training_id, { run_async: true });
      await pollStatus(statusData.training_id);
    } catch (err) {
      setError(extractApiError(err));
    }
  };

  const chartData = metricsHistory.map((row) => ({
    round: `R${row.round_num}`,
    ...(row.metrics || {}),
  }));

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h1 className="text-2xl font-bold text-white">
          Federated Training Orchestration & Control
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Configure sequential 3-hospital single-laptop training (default failsafe) or 2-laptop distributed federation (Laptop A GPU + Laptop B remote CPU), monitor live rounds, or cancel/resume from verified SHA-256 checkpoints.
        </p>
      </div>

      {error && (
        <div className="bg-rose-950/70 border border-rose-700 text-rose-200 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <form
          onSubmit={handleStart}
          className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4"
        >
          <h2 className="text-base font-semibold text-white">
            Job Configuration
          </h2>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Execution Mode
            </label>
            <select
              value={executionMode}
              onChange={(e) => setExecutionMode(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="single_laptop">
                Mode A: Single-Laptop Failsafe (Default — 3 Local Hospitals)
              </option>
              <option value="distributed_two_laptop">
                Mode B: Two-Laptop Distributed (Laptop A GPU + Laptop B Remote CPU)
              </option>
            </select>
          </div>

          {executionMode === "distributed_two_laptop" && (
            <div className="bg-indigo-950/50 border border-indigo-800/70 rounded-lg p-3 text-xs text-indigo-200 space-y-1">
              <div className="font-semibold">
                Two-Laptop Distributed Setup (client_1 GPU + client_2 Remote CPU)
              </div>
              <div className="text-indigo-300">
                Ensure <code className="text-white">FL_CLIENT_AUTH_TOKEN</code> is set on both machines, then run on Laptop B:
              </div>
              <div className="bg-slate-950 border border-slate-800 rounded p-2 font-mono text-[11px] text-emerald-300 break-all">
                python -m backend.fl.remote_client --coordinator-url http://&lt;LAPTOP_A_IP&gt;:8000 --client-id client_2 --device cpu
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs text-slate-400 mb-1">Task Type</label>
            <select
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="classification">Classification (MedMNIST)</option>
              <option value="segmentation">Segmentation (FeTS 2022)</option>
              <option value="superresolution">Super-Resolution (BreaKHis)</option>
              <option value="fusion">Multimodal Fusion (CT/PET/SPECT-MRI)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Federated Algorithm / Baseline
            </label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="omnifm">
                OmniFM (Spectral + GSKR + ECA + PSP + SPAlign)
              </option>
              <option value="fedavg">FedAvg Baseline (Shared Backbone)</option>
              <option value="fedprox">
                FedProx Baseline (Proximal Term μ=0.01)
              </option>
              <option value="fedper">
                FedPer Baseline (Personalized Head/Decoder)
              </option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-400 mb-1">Rounds</label>
              <input
                type="number"
                min={1}
                max={100}
                value={numRounds}
                onChange={(e) => setNumRounds(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Local Epochs
              </label>
              <input
                type="number"
                min={1}
                max={10}
                value={localEpochs}
                onChange={(e) => setLocalEpochs(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                min={1}
                max={64}
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                Samples / Client
              </label>
              <input
                type="number"
                min={4}
                max={5000}
                value={maxSamples}
                onChange={(e) => setMaxSamples(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
              />
            </div>
          </div>

          {taskType === "superresolution" && (
            <div>
              <label className="block text-xs text-slate-400 mb-1">
                SR Scale Factor
              </label>
              <select
                value={scaleFactor}
                onChange={(e) => setScaleFactor(Number(e.target.value))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
              >
                <option value={2}>x2</option>
                <option value={4}>x4</option>
                <option value={8}>x8</option>
              </select>
            </div>
          )}

          {algorithm === "omnifm" && (
            <div className="space-y-1.5 pt-2 border-t border-slate-800">
              <div className="text-xs font-medium text-slate-300">
                OmniFM Component Ablation Switches
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                {Object.keys(ablations).map((key) => (
                  <label key={key} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={ablations[key]}
                      onChange={(e) =>
                        setAblations({ ...ablations, [key]: e.target.checked })
                      }
                    />
                    {key.replace("enable_", "").toUpperCase()}
                  </label>
                ))}
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium"
          >
            {submitting ? "Starting Job..." : "Start Federated Training"}
          </button>
        </form>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-white">
                  Current Job Lifecycle & Checkpoint Status
                </h2>
                <div className="text-xs text-slate-400 font-mono mt-0.5">
                  ID: {statusData?.training_id || "None"} • Mode:{" "}
                  <span className="text-emerald-300 font-semibold">
                    {statusData?.execution_mode === "distributed"
                      ? "DISTRIBUTED (2-LAPTOP)"
                      : "SINGLE-LAPTOP"}
                  </span>{" "}
                  • Status:{" "}
                  <span className="text-indigo-300 font-semibold">
                    {statusData?.status || "IDLE"}
                  </span>{" "}
                  • Round {statusData?.current_round ?? 0}/
                  {statusData?.total_rounds ?? 0}
                </div>
                {statusData?.awaiting_remote_clients?.length > 0 && (
                  <div className="text-xs text-amber-300 font-mono mt-1">
                    Waiting for remote hospital worker(s):{" "}
                    {statusData.awaiting_remote_clients.join(", ")}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCancel}
                  disabled={!statusData?.training_id}
                  className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white rounded-lg text-xs font-medium"
                >
                  Request Safe Cancel
                </button>
                <button
                  onClick={handleResume}
                  disabled={!statusData?.training_id}
                  className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white rounded-lg text-xs font-medium"
                >
                  Resume from Checkpoint
                </button>
              </div>
            </div>

            {statusData?.error_message && (
              <div className="bg-rose-950/70 border border-rose-700 text-rose-200 p-3 rounded-lg text-xs">
                <strong>Error Detail:</strong> {statusData.error_message}
              </div>
            )}

            {statusData?.latest_checkpoint && (
              <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 space-y-1">
                <div>
                  Latest Verified Checkpoint: Round{" "}
                  {statusData.latest_checkpoint.round_num}
                </div>
                <div className="truncate text-slate-400">
                  SHA-256: {statusData.latest_checkpoint.model_hash}
                </div>
              </div>
            )}

            {chartData.length > 0 && (
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
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
                      dataKey="task_loss"
                      name="Task Loss"
                      stroke="#f43f5e"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="align_loss"
                      name="SPAlign Loss"
                      stroke="#eab308"
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-2">
              Execution Logs
            </h3>
            <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs text-slate-300 space-y-1">
              {logs.length === 0 ? (
                <div className="text-slate-500">No training logs recorded yet.</div>
              ) : (
                logs.map((line, idx) => <div key={idx}>{line}</div>)
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
