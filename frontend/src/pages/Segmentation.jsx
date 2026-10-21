import React, { useEffect, useState } from "react";
import { extractApiError, omnifmApi } from "../services/api";

export default function Segmentation() {
  const [file, setFile] = useState(null);
  const [referenceMask, setReferenceMask] = useState(null);
  const [clientId, setClientId] = useState("client_1");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    omnifmApi
      .getMetricsHistory({ task_type: "segmentation" })
      .then((res) => setHistory(res.global_history || []))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an MRI slice image.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("client_id", clientId);
      if (referenceMask) {
        formData.append("reference_mask", referenceMask);
      }
      const res = await omnifmApi.predictSegment(formData);
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
          Brain Tumor MRI Segmentation (FeTS 2022 T1ce)
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Lightweight U-Net with Bottleneck OmniFM Conditioning (128×128). Dice and IoU are reported ONLY when a ground-truth reference mask is provided or on labeled validation sets.
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
            1. Upload T1ce MRI Slice
          </h2>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Hospital Partition Context
            </label>
            <select
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="client_1">Hospital A (client_1)</option>
              <option value="client_2">Hospital B (client_2)</option>
              <option value="client_3">Hospital C (client_3)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              MRI Slice Image (PNG/JPEG)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Optional Ground-Truth Mask (Required for Dice/IoU)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setReferenceMask(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-slate-700 file:text-white"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium"
          >
            {loading ? "Segmenting Slice..." : "Run Segmentation"}
          </button>
        </form>

        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h2 className="text-base font-semibold text-white">
            2. Segmentation Mask, Probability Map & Verified Metrics
          </h2>

          {!result ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg">
              Upload an MRI slice to view the predicted binary mask and probability heatmap.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Input Slice (128×128)
                  </span>
                  <img
                    src={result.input_base64}
                    alt="MRI Input"
                    className="w-36 h-36 object-contain rounded border border-slate-700"
                  />
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Probability Map
                  </span>
                  <img
                    src={result.probability_map_base64}
                    alt="Probability Map"
                    className="w-36 h-36 object-contain rounded border border-slate-700"
                  />
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Binary Tumor Mask
                  </span>
                  <img
                    src={result.mask_base64}
                    alt="Binary Mask"
                    className="w-36 h-36 object-contain rounded border border-slate-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-slate-400">Model Version</div>
                  <div className="font-mono text-indigo-300 mt-1">
                    {result.model_version}
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-slate-400">Predicted Mask Coverage</div>
                  <div className="text-white font-semibold mt-1">
                    {(result.tumor_pixel_fraction * 100).toFixed(2)}% pixels
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-slate-400">Reference Metrics (Dice / IoU)</div>
                  {result.metrics ? (
                    <div className="text-emerald-400 font-semibold mt-1">
                      Dice: {result.metrics.dice.toFixed(4)} • IoU:{" "}
                      {result.metrics.iou.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-amber-300 mt-1">
                      Withheld (No ground-truth mask uploaded)
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-base font-semibold text-white mb-2">
          Labeled Validation Evaluation History (Dice & IoU)
        </h3>
        {history.length === 0 ? (
          <div className="text-xs text-slate-500">
            No stored segmentation evaluation rounds yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2">Experiment</th>
                  <th className="py-2">Round</th>
                  <th className="py-2">Dice</th>
                  <th className="py-2">IoU</th>
                  <th className="py-2">Task Loss</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, idx) => (
                  <tr key={idx} className="border-b border-slate-800/60">
                    <td className="py-2 font-mono">{h.experiment_id}</td>
                    <td className="py-2">R{h.round_num}</td>
                    <td className="py-2">
                      {h.metrics?.dice !== undefined
                        ? Number(h.metrics.dice).toFixed(4)
                        : "N/A"}
                    </td>
                    <td className="py-2">
                      {h.metrics?.iou !== undefined
                        ? Number(h.metrics.iou).toFixed(4)
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
