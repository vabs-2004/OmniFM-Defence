import React, { useEffect, useState } from "react";
import { extractApiError, omnifmApi } from "../services/api";

export default function BlockchainExplorer() {
  const [logData, setLogData] = useState({
    total: 0,
    confirmed_transactions: 0,
    completed_rounds: 0,
    audit_disclaimer: "",
    items: [],
  });
  const [filters, setFilters] = useState({
    experiment_id: "",
    task_type: "",
    round_num: "",
    client_id: "",
    action_type: "",
  });
  const [verifyForm, setVerifyForm] = useState({
    artifact_id: "",
    experiment_id: "",
    round_num: "1",
    client_id: "",
    provided_hash: "",
  });
  const [verifyResult, setVerifyResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchLogs = async () => {
    setLoading(true);
    setError("");
    try {
      const params = { limit: 100 };
      if (filters.experiment_id.trim())
        params.experiment_id = filters.experiment_id.trim();
      if (filters.task_type) params.task_type = filters.task_type;
      if (filters.round_num !== "")
        params.round_num = Number(filters.round_num);
      if (filters.client_id) params.client_id = filters.client_id;
      if (filters.action_type) params.action_type = filters.action_type;
      const res = await omnifmApi.getBlockchainLog(params);
      setLogData(res);
    } catch (err) {
      setError(extractApiError(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const handleVerify = async (e) => {
    e.preventDefault();
    setError("");
    setVerifyResult(null);
    try {
      const payload = {};
      if (verifyForm.artifact_id.trim())
        payload.artifact_id = verifyForm.artifact_id.trim();
      if (verifyForm.experiment_id.trim())
        payload.experiment_id = verifyForm.experiment_id.trim();
      if (verifyForm.round_num !== "")
        payload.round_num = Number(verifyForm.round_num);
      if (verifyForm.client_id.trim())
        payload.client_id = verifyForm.client_id.trim();
      if (verifyForm.provided_hash.trim())
        payload.provided_hash = verifyForm.provided_hash.trim();
      const res = await omnifmApi.verifyArtifact(payload);
      setVerifyResult(res);
    } catch (err) {
      setError(extractApiError(err));
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h1 className="text-2xl font-bold text-white">
          Blockchain Audit & SHA-256 Provenance Explorer
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Inspect immutable round/client audit events and recompute canonical SHA-256 checkpoint hashes.
        </p>
        <div className="mt-3 bg-indigo-950/50 border border-indigo-800/80 text-indigo-200 px-3 py-2 rounded-lg text-xs">
          <strong>Important Distinction:</strong>{" "}
          {logData.audit_disclaimer ||
            "On-chain transaction inclusion and SHA-256 hash matching verify artifact provenance and immutability after recording; they do NOT constitute cryptographic proof of honest client training or raw dataset quality."}
        </div>
      </div>

      {error && (
        <div className="bg-rose-950/70 border border-rose-700 text-rose-200 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Summary Counts */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">
            Total Audit Events
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {logData.total}
          </div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">
            Confirmed On-Chain Transactions
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {logData.confirmed_transactions}
          </div>
        </div>
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
          <div className="text-xs uppercase text-slate-400">
            Completed Aggregated Rounds
          </div>
          <div className="text-2xl font-bold text-indigo-400 mt-1">
            {logData.completed_rounds}
          </div>
        </div>
      </div>

      {/* Artifact Hash Verification Tool */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 className="text-base font-semibold text-white">
          Artifact Provenance & SHA-256 Hash Verifier
        </h2>
        <form
          onSubmit={handleVerify}
          className="grid grid-cols-1 md:grid-cols-6 gap-3 items-end"
        >
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Experiment ID
            </label>
            <input
              type="text"
              placeholder="e.g. exp_cls_api"
              value={verifyForm.experiment_id}
              onChange={(e) =>
                setVerifyForm({ ...verifyForm, experiment_id: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Round #</label>
            <input
              type="number"
              value={verifyForm.round_num}
              onChange={(e) =>
                setVerifyForm({ ...verifyForm, round_num: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Client ID (blank = Global)
            </label>
            <input
              type="text"
              placeholder="client_1"
              value={verifyForm.client_id}
              onChange={(e) =>
                setVerifyForm({ ...verifyForm, client_id: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-xs text-slate-400 mb-1">
              Optional Expected SHA-256 Hash (0x...)
            </label>
            <input
              type="text"
              placeholder="Leave blank to recompute from stored checkpoint file"
              value={verifyForm.provided_hash}
              onChange={(e) =>
                setVerifyForm({ ...verifyForm, provided_hash: e.target.value })
              }
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono"
            />
          </div>
          <button
            type="submit"
            className="py-1.5 px-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium"
          >
            Verify Hash
          </button>
        </form>

        {verifyResult && (
          <div
            className={`p-3 rounded-lg border text-xs space-y-1 ${
              verifyResult.match
                ? "bg-emerald-950/60 border-emerald-700 text-emerald-200"
                : "bg-rose-950/60 border-rose-700 text-rose-200"
            }`}
          >
            <div className="font-semibold">
              Artifact Hash Match:{" "}
              {verifyResult.match ? "VERIFIED (TRUE)" : "MISMATCH (FALSE)"} •
              Method: {verifyResult.verification_method}
            </div>
            <div className="font-mono break-all">
              Recorded Hash: {verifyResult.recorded_hash}
            </div>
            <div className="font-mono break-all">
              Recomputed Hash: {verifyResult.recomputed_hash || "N/A"}
            </div>
            <div>
              On-Chain Transaction Included:{" "}
              {verifyResult.transaction_included_on_chain ? "Yes" : "No (Off-chain SQLite record)"}
            </div>
          </div>
        )}
      </div>

      {/* Filters & Event Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Experiment Filter
            </label>
            <input
              type="text"
              placeholder="All experiments"
              value={filters.experiment_id}
              onChange={(e) =>
                setFilters({ ...filters, experiment_id: e.target.value })
              }
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Task</label>
            <select
              value={filters.task_type}
              onChange={(e) =>
                setFilters({ ...filters, task_type: e.target.value })
              }
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              <option value="">All Tasks</option>
              <option value="classification">Classification</option>
              <option value="segmentation">Segmentation</option>
              <option value="superresolution">Super-Resolution</option>
              <option value="fusion">Fusion</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Client</label>
            <select
              value={filters.client_id}
              onChange={(e) =>
                setFilters({ ...filters, client_id: e.target.value })
              }
              className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              <option value="">All Clients</option>
              <option value="client_1">client_1</option>
              <option value="client_2">client_2</option>
              <option value="client_3">client_3</option>
            </select>
          </div>
          <button
            onClick={fetchLogs}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-medium"
          >
            Apply Filters
          </button>
        </div>

        {loading ? (
          <div className="text-xs text-slate-400 py-8 text-center">
            Loading audit records...
          </div>
        ) : logData.items.length === 0 ? (
          <div className="text-xs text-slate-500 py-8 text-center">
            No audit events match the selected filters.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2">ID</th>
                  <th className="py-2">Experiment</th>
                  <th className="py-2">Round</th>
                  <th className="py-2">Client</th>
                  <th className="py-2">Action</th>
                  <th className="py-2">Artifact SHA-256</th>
                  <th className="py-2">Tx Hash</th>
                  <th className="py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {logData.items.map((item) => (
                  <tr key={item.id} className="border-b border-slate-800/60">
                    <td className="py-2">{item.id}</td>
                    <td className="py-2 font-mono">
                      {item.experiment_id || "—"}
                    </td>
                    <td className="py-2">R{item.round_num ?? 0}</td>
                    <td className="py-2 font-mono">{item.client_id || "Global"}</td>
                    <td className="py-2 text-indigo-300">{item.action_type}</td>
                    <td className="py-2 font-mono max-w-[160px] truncate">
                      {item.artifact_hash || "—"}
                    </td>
                    <td className="py-2 font-mono max-w-[140px] truncate">
                      {item.tx_hash || "Off-chain"}
                    </td>
                    <td className="py-2">
                      <span
                        className={`px-2 py-0.5 rounded ${
                          item.status === "CONFIRMED"
                            ? "bg-emerald-950 text-emerald-300"
                            : "bg-amber-950 text-amber-300"
                        }`}
                      >
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
