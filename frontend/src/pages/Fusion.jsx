import React, { useState } from "react";
import { extractApiError, omnifmApi } from "../services/api";

export default function Fusion() {
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);
  const [modalityPair, setModalityPair] = useState("CT-MRI");
  const [isRegistered, setIsRegistered] = useState(true);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!fileA || !fileB) {
      setError("Please upload both Source Modality A and Source Modality B images.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file_a", fileA);
      formData.append("file_b", fileB);
      formData.append("modality_pair", modalityPair);
      formData.append("is_registered", String(isRegistered));
      const res = await omnifmApi.predictFusion(formData);
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
          Multimodal Medical Image Fusion (CT–MRI, PET–MRI, SPECT–MRI)
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          U2Fusion-Inspired Dense Backbone + OmniFM Spectral Conditioning. Fuses verified co-registered modality pairs (128×128).
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
            1. Select Modality Pair & Upload Sources
          </h2>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Modality Pair & Hospital Node
            </label>
            <select
              value={modalityPair}
              onChange={(e) => setModalityPair(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white"
            >
              <option value="CT-MRI">Hospital A — CT–MRI Pair</option>
              <option value="PET-MRI">Hospital B — PET–MRI Pair</option>
              <option value="SPECT-MRI">Hospital C — SPECT–MRI Pair</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Source Modality A (CT / PET / SPECT)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFileA(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white"
            />
          </div>

          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Source Modality B (Structural MRI)
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFileB(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white"
            />
          </div>

          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={isRegistered}
              onChange={(e) => setIsRegistered(e.target.checked)}
              className="rounded border-slate-700 bg-slate-800"
            />
            Confirm images are spatially co-registered / aligned
          </label>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium"
          >
            {loading ? "Fusing Modalities..." : "Run Multimodal Fusion"}
          </button>
        </form>

        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
          <h2 className="text-base font-semibold text-white">
            2. Three-Panel Display: Source A • Fused Output • Source B
          </h2>

          {!result ? (
            <div className="h-64 flex items-center justify-center text-slate-500 text-sm border border-dashed border-slate-800 rounded-lg">
              Upload a pair of medical images to inspect the 3-panel fusion comparison.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Source A ({result.modality_pair.split("-")[0]})
                  </span>
                  <img
                    src={result.source_a_base64}
                    alt="Source A"
                    className="w-36 h-36 object-contain rounded border border-slate-700"
                  />
                </div>
                <div className="bg-slate-950 border border-indigo-700/70 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs font-semibold text-indigo-300 mb-2">
                    OmniFM Fused Output
                  </span>
                  <img
                    src={result.fused_base64}
                    alt="Fused Output"
                    className="w-36 h-36 object-contain rounded border border-indigo-600"
                  />
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 flex flex-col items-center">
                  <span className="text-xs text-slate-400 mb-2">
                    Source B ({result.modality_pair.split("-")[1]})
                  </span>
                  <img
                    src={result.source_b_base64}
                    alt="Source B"
                    className="w-36 h-36 object-contain rounded border border-slate-700"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-1">
                  <div className="text-slate-400">Registration & Model Status</div>
                  <div className="text-white">{result.registration_note}</div>
                  <div className="font-mono text-indigo-300">
                    Version: {result.model_version}
                  </div>
                </div>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-1">
                  <div className="text-slate-400">
                    Computed Fusion Metrics ({result.metrics_status})
                  </div>
                  {result.metrics ? (
                    <div className="text-emerald-400 space-y-0.5">
                      <div>
                        Mean SSIM to Sources:{" "}
                        {result.metrics.ssim_to_sources.toFixed(4)} • Mean VIF:{" "}
                        {result.metrics.vif_to_sources.toFixed(4)}
                      </div>
                      <div>
                        Spatial Frequency (No-Ref):{" "}
                        {result.metrics.spatial_frequency_no_ref.toFixed(4)}
                      </div>
                    </div>
                  ) : (
                    <div className="text-amber-300">
                      Metrics withheld because input pair was marked unregistered.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
