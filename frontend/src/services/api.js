import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

export function extractApiError(err) {
  if (err?.response?.data?.detail) {
    const detail = err.response.data.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return err?.message || "Unexpected API communication error.";
}

export const omnifmApi = {
  getHealth: () => apiClient.get("/api/health").then((r) => r.data),
  getHospitals: () => apiClient.get("/api/hospital").then((r) => r.data),
  registerHospital: (payload) =>
    apiClient.post("/api/hospital/register", payload).then((r) => r.data),

  getExperiments: (params = {}) =>
    apiClient.get("/api/experiments", { params }).then((r) => r.data),
  startTraining: (payload) =>
    apiClient.post("/api/training/start", payload).then((r) => r.data),
  getTrainingStatus: (trainingId) =>
    apiClient
      .get("/api/training/status", {
        params: trainingId ? { training_id: trainingId } : {},
      })
      .then((r) => r.data),
  cancelTraining: (trainingId) =>
    apiClient.post(`/api/training/${trainingId}/cancel`).then((r) => r.data),
  resumeTraining: (trainingId, payload = { run_async: true }) =>
    apiClient
      .post(`/api/training/${trainingId}/resume`, payload)
      .then((r) => r.data),
  getTrainingLogs: (trainingId, params = { limit: 100, offset: 0 }) =>
    apiClient
      .get(`/api/training/${trainingId}/logs`, { params })
      .then((r) => r.data),

  getMetricsHistory: (params = {}) =>
    apiClient.get("/api/metrics/history", { params }).then((r) => r.data),
  getBlockchainLog: (params = {}) =>
    apiClient.get("/api/blockchain/log", { params }).then((r) => r.data),
  verifyArtifact: (payload) =>
    apiClient.post("/api/blockchain/verify", payload).then((r) => r.data),
  getArtifacts: (params = {}) =>
    apiClient.get("/api/artifacts", { params }).then((r) => r.data),

  predictClassify: (formData) =>
    apiClient
      .post("/api/predict/classify", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data),
  predictSegment: (formData) =>
    apiClient
      .post("/api/predict/segment", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data),
  predictSuperResolution: (formData) =>
    apiClient
      .post("/api/predict/superresolution", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data),
  predictFusion: (formData) =>
    apiClient
      .post("/api/predict/fusion", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data),
};
