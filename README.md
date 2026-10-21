# OmniFM Blockchain — Frequency-Aware Federated Medical AI

A local research prototype integrating a **Frequency-Aware Medical Foundation Model (OmniFM)**—featuring a 2D FFT `SpectralEncoder`, Global Spectral Knowledge Retrieval (`GSKR`), Embedding Cross-Attention (`ECA`), Prefix-Suffix Spectral Prompting (`PSP`), and Spectral-Prompt Alignment (`SPAlign`)—with an **Ethereum/Ganache Smart Contract Audit Trail (`FederatedLearning.sol`)**, **FastAPI + SQLite Persistence**, and a **7-Page React + Tailwind Dashboard**.

---

## 1. Architecture & Core Principles

1. **Sequential 3-Hospital Simulation on a Single 6 GB GPU:**
   - Simulates `client_1`, `client_2`, and `client_3` sequentially so peak VRAM stays well below the 6 GB ceiling (`< 250 MB` measured across all 4 tasks).
2. **Strict Parameter-Sharing Policy (`ParameterSharingPolicy`):**
   - **Shared:** CNN/U-Net/SR/Fusion backbones, `SpectralEncoder`, `EmbeddingCrossAttention`, `global_prefix_prompt`, and spatial token projections.
   - **Personalized (Never Aggregated in FedAvg):** `client_suffix_prompt` and dataset-specific classification heads (`dermamnist`: 7 classes, `pathmnist`: 9 classes, `bloodmnist`: 8 classes).
3. **Off-Chain SHA-256 Artifacts & On-Chain Provenance:**
   - Raw patient images and identifiers never leave the client or appear in logs/on-chain.
   - Canonical tensor `state_dict` SHA-256 digests (`0x`-prefixed `bytes32`) are recorded in SQLite and on-chain in `FederatedLearning.sol`.

---

## 2. Quick Setup

### 2.1 Environment Configuration
Copy `.env.example` to `.env` (optional—safe defaults work out of the box):
```powershell
Copy-Item .env.example .env
```

### 2.2 Python Virtual Environment
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2.3 Blockchain Smart Contract (`blockchain/`)
```powershell
cd blockchain
npm install
npx hardhat compile
npx hardhat test
cd ..
```

### 2.4 Frontend Application (`frontend/`)
```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## 3. Running the Full Stack Locally

Open **3 PowerShell terminals** from `C:\projects\OmniFM\OmniFM`:

### Terminal 1 — Local Blockchain JSON-RPC Node & Contract Deployment (Optional / Audit Mode)
```powershell
cd C:\projects\OmniFM\OmniFM\blockchain
node scripts/rpc_server.js
```
In another terminal (once `rpc_server.js` prints `Listening on http://127.0.0.1:8545`), deploy `FederatedLearning.sol`:
```powershell
cd C:\projects\OmniFM\OmniFM\blockchain
npx hardhat run scripts/deploy.js --network localhost
```
Copy the printed `CONTRACT_ADDRESS` and development private keys into `.env` if you want live on-chain transactions (`BLOCKCHAIN_AUDIT_ENABLED=true`).

### Terminal 2 — FastAPI Backend Server (`http://127.0.0.1:8000`)
```powershell
cd C:\projects\OmniFM\OmniFM
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- System Health Check: `http://127.0.0.1:8000/api/health`

### Terminal 3 — React + Tailwind Frontend (`http://localhost:5173`)
```powershell
cd C:\projects\OmniFM\OmniFM\frontend
npm run dev
```

---

## 4. Verification & Test Suite Commands

Run all unit, integration, and GPU smoke tests from `C:\projects\OmniFM\OmniFM`:

```powershell
# Phase 0 & 1: Config, Privacy Logging, Canonical Hashing, and Web3 Connector
.\venv\Scripts\python.exe -m unittest tests.unit.test_config tests.unit.test_logging_and_health tests.unit.test_hashing tests.integration.test_web3_connector -v

# Phase 1: Solidity & Hardhat Smart Contract Tests
cd blockchain; npx hardhat test; cd ..

# Phase 2: OmniFM Spectral, GSKR, ECA, PSP, SPAlign & Task Adapters (CPU + GPU)
.\venv\Scripts\python.exe -m unittest tests.unit.test_omnifm_components -v

# Phase 3: Dataset Loaders & Patient-Split Leakage Verification
.\venv\Scripts\python.exe -m unittest tests.unit.test_datasets_and_splits -v

# Phase 4: Metrics & Real-Batch Task Model GPU Smoke Tests
.\venv\Scripts\python.exe -m unittest tests.unit.test_metrics_and_task_models -v

# Phase 5 & 6.1: Federated Aggregation, Baselines, SQLite Persistence & 3-Client Smoke Tests
.\venv\Scripts\python.exe -m unittest tests.unit.test_fl_aggregation_and_baselines tests.smoke.test_fl_smoke_all_tasks -v

# Phase 6 & 7: FastAPI Endpoints & Frontend Route Verification
.\venv\Scripts\python.exe -m unittest tests.integration.test_api_endpoints tests.integration.test_frontend_and_api_routes -v
```

---

## 5. Phase 8 Commands — 20-Round Experiments, Baselines, Ablations & Results Export

All Phase 8 experiments can be run via [`backend/fl/run_experiments.py`](file:///C:/projects/OmniFM/OmniFM/backend/fl/run_experiments.py):

### Task 8.1 — Full 20-Round OmniFM Runs (Per Task)
```powershell
# 1. Classification (DermaMNIST / PathMNIST / BloodMNIST) — 20 Rounds
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode main --task classification --rounds 20 --max-train-samples 128 --max-val-samples 64

# 2. Segmentation (FeTS 2022 T1ce Brain MRI) — 20 Rounds
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode main --task segmentation --rounds 20 --max-train-samples 32 --max-val-samples 16

# 3. Super-Resolution (BreaKHis Histopathology x2) — 20 Rounds
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode main --task superresolution --scale 2 --rounds 20 --max-train-samples 32 --max-val-samples 16

# 4. Multimodal Fusion (CT-MRI / PET-MRI / SPECT-MRI) — 20 Rounds
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode main --task fusion --rounds 20 --max-train-samples 32 --max-val-samples 16
```

### Task 8.2 — Baselines (`FedAvg`, `FedProx`, `FedPer`)
```powershell
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode baselines --task classification --rounds 20 --max-train-samples 128 --max-val-samples 64
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode baselines --task segmentation --rounds 20 --max-train-samples 32 --max-val-samples 16
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode baselines --task superresolution --scale 2 --rounds 20 --max-train-samples 32 --max-val-samples 16
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode baselines --task fusion --rounds 20 --max-train-samples 32 --max-val-samples 16
```

### Task 8.3 — OmniFM Component Ablations (`no_gskr`, `no_eca`, `no_psp`, `no_spalign`)
```powershell
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode ablations --task classification --rounds 20 --max-train-samples 128 --max-val-samples 64
```

### Task 8.4 — Export Measured SQLite Results to `results/results.md` and `results/metrics_summary.csv`
```powershell
.\venv\Scripts\python.exe -m backend.fl.run_experiments --mode export
```
