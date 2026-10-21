import React, { useEffect, useState } from "react";
import { extractApiError, omnifmApi } from "../services/api";

export default function SuperResolution() {
  const [file, setFile] = useState(null);
  const [referenceHr, setReferenceHr] = useState(null);
  const [scale, setScale] = useState(2);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    omnifmApi
      .getMetricsHistory({ task_type: "superresolution" })
      .then((res) => setHistory(res.global_history || []))
      .catch(() => {});
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a low-resolution histopathology patch.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("scale", String(scale));
      if (referenceHr) {
        formData.append("reference_hr", referenceHr);
      }
      const res = await omnifmApi.predictSuperResolution(formData);
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
          Histopathology Super-Resolution (BreaKHis x2 / x4 / x8)
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Lightweight Residual SR Network + OmniFM Conditioning. Reconstructs 128×128 (x2), 256×256 (x4), or 512×512 (x8) from 64×64 LR patches. PSNR & SSIM are reported ONLY when a paired HR reference is supplied.
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
            1. Configure Scale & Upload Patch
          </h2>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Upsampling Scale Factor
            </label>
            <select
              value={scale}
              onChange={(e) => setScale(Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value={2}>x2 (64×64 → 128×128)</option>
              <option value={4}>x4 (64×64 → 256×256)</option>
              <option value={8}>x8 (64×64 → 512×512)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Low-Resolution Input Image
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
              Optional High-Resolution Target (Required for PSNR / SSIM)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setReferenceHr(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-slate-700 file:text-white"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium"
          >
            {loading ? "Upsampling..." : `Enhance Image (x${scale})`}
          </button>
        </form>

        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h2 className="text-base font-semibold text-white">
            2. Before / After Comparison & Reference Metrics
          </h2>

          {!result ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg">
              Upload a histopathology patch and select x2, x4, or x8 scale to compare LR vs SR output.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Low-Resolution Input (64×64)
                  </span>
                  <img
                    src={result.input_base64}
                    alt="LR Input"
                    className="w-44 h-44 object-contain rounded border border-slate-700"
                    style={{ imageRendering: "pixelated" }}
                  />
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Super-Resolved Output (x{result.scale}:{" "}
                    {result.output_shape?.[2]}×{result.output_shape?.[3]})
                  </span>
                  <img
                    src={result.enhanced_base64}
                    alt="SR Output"
                    className="w-44 h-44 object-contain rounded border border-slate-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-slate-400">Model Version</div>
                  <div className="font-mono text-indigo-300 mt-1">
                    {result.model_version}
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="text-slate-400">
                    Paired Reference Metrics (PSNR / SSIM)
                  </div>
                  {result.metrics ? (
                    <div className="text-emerald-400 font-semibold mt-1">
                      PSNR: {result.metrics.psnr.toFixed(2)} dB • SSIM:{" "}
                      {result.metrics.ssim.toFixed(4)}
                    </div>
                  ) : (
                    <div className="text-amber-300 mt-1">
                      Withheld (No paired HR reference image uploaded)
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
          Paired Validation Evaluation History (PSNR & SSIM)
        </h3>
        {history.length === 0 ? (
          <div className="text-xs text-slate-500">
            No stored super-resolution evaluation rounds yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2">Experiment</th>
                  <th className="py-2">Round</th>
                  <th className="py-2">PSNR (dB)</th>
                  <th className="py-2">SSIM</th>
                  <th className="py-2">L1 Loss</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, idx) => (
                  <tr key={idx} className="border-b border-slate-800/60">
                    <td className="py-2 font-mono">{h.experiment_id}</td>
                    <td className="py-2">R{h.round_num}</td>
                    <td className="py-2">
                      {h.metrics?.psnr !== undefined
                        ? Number(h.metrics.psnr).toFixed(2)
                        : "N/A"}
                    </td>
                    <td className="py-2">
                      {h.metrics?.ssim !== undefined
                        ? Number(h.metrics.ssim).toFixed(4)
                        : "N/A"}
                    </td>
                    <td className="py-2">
                      {h.metrics?.l1 !== undefined
                        ? Number(h.metrics.l1).toFixed(4)
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
