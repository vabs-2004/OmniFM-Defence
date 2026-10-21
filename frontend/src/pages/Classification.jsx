import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { extractApiError, omnifmApi } from "../services/api";

export default function Classification() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [datasetName, setDatasetName] = useState("dermamnist");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    omnifmApi
      .getMetricsHistory({ task_type: "classification" })
      .then((res) => setHistory(res.global_history || []))
      .catch(() => {});
  }, []);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file to classify.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("dataset_name", datasetName);
      const res = await omnifmApi.predictClassify(formData);
      setResult(res);
    } catch (err) {
      setError(extractApiError(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h1 className="text-2xl font-bold text-white">
          Multi-Hospital Medical Image Classification
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          28×28 ResNet-18 + OmniFM Spectral Conditioning with Personalized Client Classifier Heads (DermaMNIST: 7, PathMNIST: 9, BloodMNIST: 8).
        </p>
      </div>

      {error && (
        <div className="bg-rose-950/70 border border-rose-700 text-rose-200 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <form
          onSubmit={handleSubmit}
          className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4"
        >
          <h2 className="text-base font-semibold text-white">
            1. Select Hospital Head & Upload Image
          </h2>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Dataset / Hospital Classifier Head
            </label>
            <select
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="dermamnist">
                Hospital A — DermaMNIST (7 Dermatoscopic Classes)
              </option>
              <option value="pathmnist">
                Hospital B — PathMNIST (9 Colon Pathology Classes)
              </option>
              <option value="bloodmnist">
                Hospital C — BloodMNIST (8 Blood Cell Classes)
              </option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Medical Image File (PNG/JPEG)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white"
            />
          </div>

          {previewUrl && (
            <div className="flex flex-col items-center p-3 bg-slate-950 rounded-lg border border-slate-800">
              <img
                src={previewUrl}
                alt="Uploaded preview"
                className="w-28 h-28 object-cover rounded border border-slate-700"
              />
              <span className="text-xs text-slate-400 mt-1">
                Resized to 28×28 RGB for MedMNIST Backbone
              </span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium"
          >
            {loading ? "Running OmniFM Inference..." : "Classify Image"}
          </button>
        </form>

        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h2 className="text-base font-semibold text-white">
            2. Classification Output & Uncalibrated Softmax Distribution
          </h2>

          {!result ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg">
              Upload an image and click Classify Image to inspect class probabilities.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-xs text-slate-400">Predicted Class</div>
                  <div className="text-sm font-semibold text-emerald-400 mt-1">
                    [{result.predicted_class}] {result.predicted_label}
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-xs text-slate-400">
                    Softmax Score (Uncalibrated)
                  </div>
                  <div className="text-sm font-semibold text-white mt-1">
                    {(result.confidence * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-xs text-slate-400">Model Version</div>
                  <div className="text-xs font-mono text-indigo-300 mt-1 truncate">
                    {result.model_version}
                  </div>
                </div>
              </div>

              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={result.class_probabilities || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="class_index" stroke="#94a3b8" fontSize={12} />
                    <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        fontSize: "12px",
                      }}
                    />
                    <Bar dataKey="probability" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-amber-950/50 border border-amber-800/70 text-amber-200 px-3 py-2 rounded text-xs">
                {result.clinical_disclaimer}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Stored Evaluation History */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-base font-semibold text-white mb-2">
          Stored Classification Evaluation History (Accuracy & Macro-F1)
        </h3>
        {history.length === 0 ? (
          <div className="text-xs text-slate-500">
            No stored classification evaluation rounds yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2">Experiment</th>
                  <th className="py-2">Algorithm</th>
                  <th className="py-2">Round</th>
                  <th className="py-2">Accuracy</th>
                  <th className="py-2">Macro-F1</th>
                  <th className="py-2">Task Loss</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, idx) => (
                  <tr key={idx} className="border-b border-slate-800/60">
                    <td className="py-2 font-mono">{h.experiment_id}</td>
                    <td className="py-2">{h.algorithm}</td>
                    <td className="py-2">R{h.round_num}</td>
                    <td className="py-2">
                      {h.metrics?.accuracy !== undefined
                        ? Number(h.metrics.accuracy).toFixed(4)
                        : "N/A"}
                    </td>
                    <td className="py-2">
                      {h.metrics?.macro_f1 !== undefined
                        ? Number(h.metrics.macro_f1).toFixed(4)
                        : "N/A"}
                    </td>
                    <td className="py-2">
                      {h.metrics?.task_loss !== undefined
                        ? Number(h.metrics.task_loss).toFixed(4)
                        : "N/A"}
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
