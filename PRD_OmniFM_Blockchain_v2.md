# OmniFM Blockchain Federated Learning — Product Requirements Document (PRD)

**Document status:** Implementation-ready draft  
**Project type:** Research prototype / local simulation  
**Research basis:** *OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging*  
**Project variation:** OmniFM-inspired federated learning with blockchain-based audit logging  
**Target hardware:** NVIDIA RTX 3050 6 GB VRAM, 16 GB RAM  
**Primary execution mode:** Three logical hospital clients simulated sequentially on one machine

---

## 1. Executive Summary

OmniFM Blockchain Federated Learning is a research prototype for collaborative medical-image model training across simulated hospitals. Each hospital keeps its dataset local and trains a task-specific model using an OmniFM-inspired spectral-conditioning pipeline. A federated server aggregates compatible shared parameters and maintains a global spectral knowledge bank. A local Ethereum-compatible blockchain (Ganache) records client registration, model-update hashes, aggregation events, and global-model hashes.

The system supports four image tasks:

1. Medical image classification
2. Brain MRI tumor segmentation
3. Histopathology image super-resolution
4. Heterogeneous medical image fusion

The project combines three distinct concerns:

- **Machine learning:** task models, OmniFM components, metrics, and baselines.
- **Federated orchestration:** client training, compatible-parameter aggregation, knowledge-bank updates, checkpoints, and round state.
- **Auditability:** blockchain events and cryptographic hashes that make recorded artifacts verifiable.

Blockchain is an engineering extension to the research framework, not a component of the original OmniFM paper. The prototype is not a production medical device and must not claim clinical readiness, formal regulatory compliance, or absolute privacy.

---

## 2. Problem Statement

Medical institutions may be unable or unwilling to pool raw patient images. Independent training can limit the amount and diversity of data available to each institution. Federated learning allows clients to train locally and exchange model updates rather than raw images, but model performance can degrade when clients have different data distributions, imaging modalities, or task requirements.

The original OmniFM paper proposes frequency-aware mechanisms intended to support a reusable federated optimization pipeline across medical imaging tasks and modality distributions. This project adapts that idea to a constrained local prototype and adds blockchain-based audit records.

### 2.1 Goals

- Keep raw datasets within their assigned logical client.
- Implement and test the spectral encoder, Global Spectral Knowledge Retrieval (GSKR), Embedding-wise Cross-Attention (ECA), Prefix–Suffix Spectral Prompting (PSP), and Spectral-Proximal Alignment (SPAlign).
- Support four distinct task-specific model/training adapters.
- Support a reproducible federated training loop with compatible-parameter aggregation.
- Maintain a persisted server-side spectral knowledge bank.
- Record verifiable hashes and round lifecycle events on Ganache.
- Provide an API and dashboard to start experiments, monitor progress, evaluate models, and inspect audit events.
- Compare OmniFM-inspired training against valid task-specific baselines.

### 2.2 Non-goals

- Production deployment across real hospitals.
- A claim of HIPAA/GDPR compliance based solely on federated learning or blockchain.
- Proof that a client trained honestly merely because its transaction is signed.
- Storing patient images, labels, model weights, or raw spectral embeddings on-chain.
- Reproducing the paper's reported scores or full experimental scale.
- Implementing medical Visual Question Answering (VQA) in the first version.
- Guaranteeing that OmniFM outperforms every baseline on every task.

---

## 3. Research Fidelity and Project Deviations

The project must clearly distinguish mechanisms inspired by the paper from engineering decisions made for this implementation.

### 3.1 OmniFM-inspired mechanisms

- FFT magnitude-spectrum extraction and low-frequency filtering.
- A learned spectral token encoder.
- Retrieval of top-k global spectral prototypes from a server-side bank.
- Cross-attention between backbone representations and retrieved prototypes.
- Prefix/suffix prompt integration with client-specific personalization.
- Spectral alignment regularization added to the task loss.
- A reusable federated orchestration layer with task-specific models.

### 3.2 Engineering additions

- Ganache local blockchain and Solidity audit contract.
- Model-update and global-checkpoint hash records.
- FastAPI REST service, SQLite persistence, and React dashboard.
- Job lifecycle management, resumable checkpoints, and artifact metadata.
- Local sequential client simulation to fit available hardware.

### 3.3 Reduced-scale experimental settings

The paper uses task-specific experimental protocols and, for some tasks, substantially more communication rounds than this prototype. The initial settings below are **prototype defaults**, not a reproduction of the paper's full protocol. They may be reduced further during smoke tests and must be recorded in experiment metadata.

---

## 4. Users and Primary Workflows

### 4.1 Researcher / operator

- Select a task and experiment configuration.
- Start, monitor, cancel, and resume training.
- Review per-client and global evaluation metrics.
- Inspect model versions and training logs.
- Verify recorded model hashes and transaction events.

### 4.2 Simulated hospital client

- Has a stable client ID and a separately provisioned wallet address.
- Has a private local dataset partition.
- Trains only on its assigned data.
- Retains client-specific parameters locally where configured.
- Submits permitted shared updates and spectral embeddings through the FL transport.
- Signs blockchain transactions using its own configured account.

### 4.3 Core end-to-end workflow

1. Initialize the task configuration and compatible model architecture.
2. Register each logical hospital and its wallet address.
3. Initialize the global model and server knowledge bank.
4. For each communication round, provide each participating client with the current compatible shared model state and required retrieval context.
5. Train clients sequentially: client 1, client 2, client 3.
6. Extract the permitted shared parameter update and spectral embedding.
7. Compute a deterministic hash of the serialized shared update; submit update metadata to the FL server and record the hash on-chain.
8. Validate submissions and aggregate only compatible shared parameters.
9. Update and persist the global spectral knowledge bank according to the configured retrieval/pruning policy.
10. Save the global checkpoint, compute its hash, and record the aggregation/global-model event on-chain.
11. Broadcast the next-round shared state and retrieval context.
12. Record metrics and repeat until completion, cancellation, or failure.

Raw medical images must never be sent to the server or blockchain by the federated-training workflow.

---

## 5. Scope: Four Supported Tasks

### Task 1 — Medical Image Classification

**Purpose:** Evaluate classification across heterogeneous medical-image datasets while preserving valid label semantics.

**Datasets:** MedMNIST v2 subsets:
- DermaMNIST: 7 classes.
- PathMNIST: 9 classes.
- BloodMNIST: 8 classes.

**Critical label-space rule:** These datasets have different label meanings. They must not be treated as a single shared classification head with directly averaged weights unless a valid common label mapping is explicitly defined and documented.

**Initial supported configuration:** Use a shared representation/spectral pipeline with task- or client-specific classification heads. Aggregate only parameters that have identical architecture and compatible semantics. Evaluate each dataset/head using its own class labels. If a paper-faithful cross-modality classification benchmark is later implemented, add a separately defined combined label space and partition protocol; do not silently conflate it with this configuration.

**Backbone:** ResNet-18 adapted for small inputs. Use a suitable first convolution and pooling configuration for 28×28 images.

**Input/output:**
- Input: MedMNIST RGB image, usually 28×28.
- Output: dataset-specific class prediction, confidence, and class probabilities.

**Metrics:** Accuracy, macro-F1; also report per-class metrics when useful.

**Prototype defaults:** 3 clients; 5 rounds for smoke test; 20 rounds for prototype; 1 local epoch initially, with batch size 32 if memory permits.

### Task 2 — Brain MRI Tumor Segmentation

**Purpose:** Segment GD-enhancing tumor regions in T1ce MRI slices.

**Dataset:** FeTS2022, subject-level partitions where institution metadata is available. If true institution partitions are unavailable, document the simulated partition strategy and do not claim actual institution-level separation.

**Backbone:** Lightweight U-Net/MONAI-compatible model.

**Input/output:**
- Input: 2D axial T1ce MRI slice resized/padded to 128×128, with consistent intensity normalization.
- Output: binary tumor mask.

**Metrics:** Dice and IoU computed against ground-truth masks on a held-out labeled evaluation set.

**Prototype defaults:** 3 clients; 5-round smoke test; 20 rounds target; 1 local epoch; batch size 4–8 subject to measured VRAM.

### Task 3 — Histopathology Image Super-Resolution

**Purpose:** Reconstruct higher-resolution histopathology image patches from synthetically downsampled inputs.

**Dataset:** BreaKHis, subject/patient-level splitting to prevent leakage. Magnification-specific client distributions may be used to simulate non-IID data.

**Model:** Lightweight convolutional encoder-decoder or residual SR network.

**Input/output:**
- Input: 64×64 low-resolution RGB patch.
- Output: x2, x4, or x8 reconstruction, with output size defined by the scale factor.

**Data generation:** Create low-resolution inputs from high-resolution targets using a documented downsampling kernel and degradation pipeline. Use identical degradation rules for training and evaluation unless a separate robustness test is intended.

**Metrics:** PSNR and SSIM against paired high-resolution ground truth.

**Prototype defaults:** 3 clients; 5-round smoke test; 20 rounds target; 1–3 local epochs; batch size 4–8 subject to memory.

### Task 4 — Heterogeneous Multimodal Image Fusion

**Purpose:** Fuse paired images from different modalities while preserving complementary structural and functional information.

**Dataset:** Harvard Medical School medical image fusion dataset or a verified, appropriately licensed dataset with documented paired examples. Proposed client pairs: CT–MRI, PET–MRI, SPECT–MRI, only where valid pairs are actually available.

**Model:** U2Fusion-inspired implementation. The exact implementation and licensing/provenance of any reused code must be documented.

**Input/output:**
- Input: two paired images, spatially registered or validated as aligned, resized to a consistent 128×128 representation.
- Output: one fused image.

**Metrics:** SSIM and L1 only where a suitable reference is defined; VIF and other no-reference fusion metrics where the implementation and interpretation are documented. Do not report metrics without their required reference data.

**Prototype defaults:** 3 logical clients; 5-round smoke test; 20 rounds target; 1 local epoch initially; batch size 1–4 subject to measured VRAM.

**Important:** Fusion training may have a different objective and model structure from supervised classification/segmentation/SR. The common FL lifecycle is reusable, but task loss, model forward pass, and evaluation must remain task-specific.

---

## 6. OmniFM Components and Interfaces

All components must be independently testable before integration. The embedding dimension defaults to 128, and GSKR top-k defaults to 2. These are configurable experiment parameters.

### 6.1 FFT spectral encoding — `backend/omnifm/spectral.py`

**Input:** image tensor `(B, C, H, W)`.

**Processing:**
1. Apply a documented FFT over spatial dimensions.
2. Compute magnitude spectrum.
3. Center the spectrum if required by the low-pass mask implementation.
4. Apply a configurable low-pass mask.
5. Pass the filtered spectrum through a lightweight learned spectral mixing/projection network.
6. Pool and normalize the resulting spectral token.

**Output:** `s`, shape `(B, 128)`, finite values, L2-normalized where the norm is nonzero.

Tests must cover RGB/grayscale input, odd/even dimensions, batch size 1, finite gradients, and CPU execution.

### 6.2 Global Spectral Knowledge Retrieval — `backend/omnifm/gskr.py`

**Input:** local token `(B, D)` and bank `(M, D)`.

**Behavior:**
- Compute cosine similarity between local tokens and stored prototypes.
- Retrieve up to `k` valid prototypes.
- Handle an empty bank and a bank with fewer than `k` entries explicitly.
- Avoid using the current sample's token as its own global neighbor when that would leak the query into the retrieval target; define the insertion/retrieval ordering.
- Return retrieved prototypes and optional similarity scores/mask.

**Output:** retrieved prototypes with shape `(B, K_actual, D)` or a documented padded/masked shape.

The bank must define capacity, prototype IDs, round/client/task metadata, update ordering, persistence format, and pruning rules. Task-specific banks or task-aware retrieval filters must be supported so semantically unrelated tasks are not inadvertently treated as interchangeable.

### 6.3 Embedding-wise Cross-Attention — `backend/omnifm/eca.py`

**Inputs:**
- Backbone representation converted to a documented token shape `(B, L, D)`.
- Retrieved spectral prototypes `(B, K, D)`.

**Behavior:** Backbone tokens act as queries; retrieved prototypes act as keys and values. Project dimensions and attention-head count must be configurable and valid.

**Output:** fused representation with a declared shape, residual/normalization behavior, and a defined path back to the relevant task adapter.

### 6.4 Prefix–Suffix Spectral Prompting — `backend/omnifm/psp.py`

Construct a sequence conceptually equivalent to:

`r' = [Z_prefix | r_backbone | c_client]`

- Prefix tokens encode retrieved/fused spectral context.
- Backbone representation retains task information.
- Suffix/CLS prompts may be client-specific and remain local unless the experiment explicitly specifies otherwise.

**Architecture adapters are mandatory.** CNN feature maps cannot be concatenated directly with token sequences without conversion and a compatible decoder. Segmentation must preserve spatial alignment/skip connections; SR and fusion must preserve the output's spatial structure. Each model adapter must document where and how prompting is injected.

### 6.5 Spectral-Proximal Alignment — `backend/omnifm/spalign.py`

Default formula:

`L_align = mean(||s - mean(Sg)||²)`

Total objective:

`L_total = L_task + lambda_align * L_align`

- Default `lambda_align = 0.1`, configurable.
- Define behavior when no retrieved prototype exists; do not silently calculate a meaningless alignment loss.
- Ensure the loss is finite and gradients flow through the intended trainable modules.
- Log task loss and alignment loss separately.

### 6.6 Required component tests

- Input/output shape checks.
- Empty and undersized knowledge bank.
- NaN/Inf checks.
- Backward pass and gradient checks.
- CPU smoke tests and GPU tests where available.
- Batch size 1.
- Serialization/reload for the knowledge bank.
- A complete forward/backward pass for every task adapter.

---

## 7. Federated Learning Design

### 7.1 Execution model

The initial version simulates three clients sequentially on one machine to conserve VRAM. Each logical client must have its own:
- client ID and configuration,
- dataset partition,
- optimizer state,
- client-specific parameters,
- local checkpoint directory,
- wallet configuration.

The clients must not share local data or personalized parameters through accidental global mutable state.

### 7.2 Server responsibilities

- Initialize and version the global shared model.
- Maintain the round state and expected-client list.
- Dispatch the current compatible shared state to each client.
- Validate client/task/round identity and update metadata.
- Aggregate only the configured shared parameter set.
- Maintain and persist the spectral knowledge bank.
- Record per-client and global metrics.
- Save and hash each global checkpoint.
- Recover safely after an interrupted run.

### 7.3 Client responsibilities

- Load only its assigned local partition.
- Receive the shared model state and retrieval context.
- Perform local training with its task-specific loss.
- Preserve local parameters designated as personalized.
- Return compatible shared parameter updates and permitted spectral embeddings.
- Report sample count, training duration, metrics, and status.
- Create the update hash over a documented canonical serialized artifact.

### 7.4 Aggregation

Implement weighted FedAvg first for compatible shared parameters:

`global_parameter = sum(n_i * client_parameter_i) / sum(n_i)`

where `n_i` is the number of eligible local training examples for that round. The exact policy for partial participation, failed clients, and unequal task distributions must be explicit.

Do not average:
- task heads with incompatible output dimensions or label meanings,
- client-specific prompt/CLS parameters intended to remain local,
- optimizer state unless a separately defined algorithm requires it.

FedProx and FedPer may be added as task-appropriate baselines. FedPer requires a clear shared/personalized layer split. All baselines must use equivalent data partitions, comparable training budgets, and the same evaluation protocol.

### 7.5 Round state machine

Supported states:
- `CREATED`
- `WAITING_FOR_CLIENTS`
- `TRAINING`
- `UPDATES_VALIDATED`
- `AGGREGATING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

The server must prevent aggregation from running twice for the same experiment/round. It must define whether all clients are required or a configured quorum is sufficient. The first version should use all configured clients unless explicitly changed.

### 7.6 Hardware and precision

- Target Python 3.11 in a pinned environment.
- Target NVIDIA RTX 3050 with 6 GB VRAM and 16 GB system RAM.
- Run clients sequentially and release GPU memory between clients where appropriate.
- Use the modern PyTorch AMP API, with device-aware autocast and gradient scaling where appropriate for FP16.
- Add configurable batch sizes, image sizes, workers, and local epochs.
- Record peak GPU memory and training duration.
- Do not promise that every task fits before running a measured smoke test.
- CPU fallback should be available for unit tests and small debugging runs.

---

## 8. Blockchain Audit Design

### 8.1 Purpose and trust boundary

Ganache is a local development chain. Blockchain records make submitted hashes and event history tamper-evident within that chain; they do not prove that a model was trained honestly or that the underlying data are correct.

The FL server is responsible for numerical validation and aggregation. The contract is responsible for enforcing audit-event rules and recording events.

### 8.2 Wallet provisioning

- Each logical hospital uses a separately configured wallet address.
- Wallet creation/private-key provisioning happens outside the contract.
- Register only public addresses and non-sensitive metadata.
- Never commit private keys or secrets to source control.
- Local Ganache private keys are development-only and must never be reused in production.

### 8.3 Contract — `blockchain/contracts/FederatedLearning.sol`

Solidity version: `^0.8.19`, subject to the pinned compiler/toolchain.

Required operations (exact function signatures may be adjusted during implementation, but the semantics must remain):

1. `registerHospital(name, wallet)`  
   Register an authorized client identity and emit `HospitalRegistered`.

2. `submitModelUpdate(round, taskType, modelHash, clientId)`  
   Record a client update hash, prevent unauthorized and duplicate submissions for the same experiment/round/client, and emit `ModelUpdateSubmitted`.

3. `checkAllSubmitted(round)`  
   Determine whether the configured expected clients have submitted valid updates for that experiment and round.

4. `triggerAggregation(round)`  
   Record the aggregation event only when submission requirements are met and aggregation has not already been recorded.

5. `storeGlobalModel(round, globalModelHash)`  
   Record the global checkpoint hash and emit `GlobalModelStored`.

6. Audit query operations  
   Provide event-based history and/or bounded on-chain records. Do not design an unbounded array-return function that becomes impractical as history grows.

### 8.4 Experiment and round identity

Every audit record must be unambiguously linked to:
- experiment ID,
- task type,
- round number,
- client ID,
- registered wallet address,
- action type,
- model/update hash where applicable,
- timestamp,
- transaction hash,
- relevant global-model version.

Use a unique key such as `(experimentId, round, clientId)` to reject duplicate updates.

### 8.5 Hashing and artifact provenance

- Define exactly which artifact bytes are hashed.
- Use deterministic serialization and a documented hash format (SHA-256 off-chain).
- Avoid hashing an arbitrary in-memory Python object representation.
- Store model weights and large artifacts off-chain.
- Link the hash to artifact metadata and the checkpoint version in the database.
- The frontend must distinguish an on-chain transaction's successful inclusion from verification that a supplied artifact matches the recorded hash.

### 8.6 Failure handling

- If the blockchain is unavailable, the training job must follow an explicit policy: either fail the audit-required round or mark audit status as pending/failed. Do not silently report it as audited.
- Do not claim a transaction succeeded until the receipt confirms it.
- Record failed transactions and retry attempts off-chain.
- Ensure retries do not create duplicate logical submissions.

### 8.7 Privacy

Never store raw medical images, patient identifiers, labels, private keys, or raw training data on-chain. Treat spectral embeddings and model updates as potentially sensitive; apply access controls and minimize transmitted information. Federated learning and hashing alone are not formal differential privacy.

---

## 9. Backend and Persistence

### 9.1 Technology

- Python 3.11
- PyTorch 2.x and a compatible CUDA build
- Flower (`flwr`) for FL simulation, with version pinned and API usage verified
- FastAPI and Uvicorn
- MONAI and nibabel for MRI processing where needed
- torchmetrics or validated metric implementations
- SQLAlchemy + SQLite
- Web3.py for Ganache interaction
- NumPy, Pillow, and dataset-specific loaders

Pin exact versions in the environment lock/requirements file after confirming compatibility. Do not assume a CUDA toolkit version automatically determines the correct PyTorch wheel; install a compatible supported build and verify `torch.cuda.is_available()`.

### 9.2 Proposed structure

```text
omnifm-blockchain/
├── PRD.md
├── AGENTS.md
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── blockchain/
│   ├── hardhat.config.js
│   ├── contracts/
│   │   └── FederatedLearning.sol
│   ├── scripts/
│   │   └── deploy.js
│   └── test/
│       └── fl_test.js
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── fl/
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── aggregation.py
│   │   ├── round_manager.py
│   │   └── protocols.py
│   ├── omnifm/
│   │   ├── spectral.py
│   │   ├── gskr.py
│   │   ├── eca.py
│   │   ├── psp.py
│   │   └── spalign.py
│   ├── models/
│   │   ├── classification.py
│   │   ├── segmentation.py
│   │   ├── superresolution.py
│   │   └── fusion.py
│   ├── tasks/
│   │   ├── base.py
│   │   ├── classification.py
│   │   ├── segmentation.py
│   │   ├── superresolution.py
│   │   └── fusion.py
│   ├── blockchain/
│   │   └── web3_connector.py
│   ├── database/
│   │   ├── db.py
│   │   └── models.py
│   ├── api/
│   │   └── endpoints/
│   ├── services/
│   │   ├── experiment_service.py
│   │   ├── artifact_service.py
│   │   └── audit_service.py
│   └── utils/
│       ├── metrics.py
│       ├── preprocessing.py
│       ├── hashing.py
│       └── logging.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── pages/
│       └── components/
├── datasets/                 # local-only; not committed
├── artifacts/                # local-only; not committed
└── tests/
    ├── unit/
    ├── integration/
    └── smoke/
```

The exact directory layout may evolve, but responsibilities must remain separated. Dataset files, checkpoints, `.env`, wallet keys, and generated model artifacts must be excluded from Git.

### 9.3 Database entities

At minimum, define:
- `HospitalClient`
- `Experiment`
- `TrainingRound`
- `ClientUpdate`
- `GlobalModelVersion`
- `MetricRecord`
- `BlockchainTransaction`
- `ArtifactMetadata`

Store structured metadata and status in SQLite; store large model artifacts and images as files referenced by metadata. Avoid storing large binary checkpoint data directly in SQLite.

---

## 10. REST API Contract

All routes return structured JSON, use explicit validation, and return actionable error messages. Long-running training must not block the request until all rounds complete.

### 10.1 System and clients

`GET /api/health`
- Returns API, database, GPU, and blockchain connection status.

`POST /api/hospital/register`
- Body: `{ "hospital_name": "...", "modality": "...", "client_id": "..." }`
- Registers a logical client and its configured wallet address.
- Private keys are never accepted in the request body.
- Returns client ID, wallet address, and registration status.

`GET /api/hospital`
- Lists registered logical clients and their status; never exposes private keys.

### 10.2 Experiments and training

`POST /api/training/start`
- Body: `{ "task_type": "classification|segmentation|superresolution|fusion", "num_rounds": 20, "config": {} }`
- Validates configuration and creates an asynchronous training job.
- Returns `{ "training_id": "...", "status": "CREATED" }`.

`GET /api/training/status?training_id=...`
- Returns status, current round, total rounds, per-client status, latest metrics, and error details when applicable.

`POST /api/training/{training_id}/cancel`
- Requests cancellation at a safe round/client boundary.

`POST /api/training/{training_id}/resume`
- Resumes from a valid checkpoint if the job is recoverable.

`GET /api/training/{training_id}/logs`
- Returns paginated or bounded logs.

### 10.3 Predictions

`POST /api/predict/classify`
- Multipart image and explicit dataset/head/model version where needed.
- Returns prediction, probabilities, confidence, and model version.

`POST /api/predict/segment`
- Multipart image.
- Returns mask image or encoded mask, model version, and preprocessing metadata.
- Dice/IoU are returned only when a ground-truth mask is supplied or a benchmark evaluation is being run.

`POST /api/predict/superresolution`
- Multipart image plus scale `2`, `4`, or `8`.
- Returns enhanced image and model version.
- PSNR/SSIM are returned only when a suitable reference image is supplied or evaluation is performed on paired test data.

`POST /api/predict/fusion`
- Multipart image pair plus modality-pair identifier.
- Returns fused image, model version, and preprocessing metadata.
- Metrics are returned only when the metric's reference/data requirements are met.

### 10.4 Audit and metrics

`GET /api/blockchain/log`
- Returns paginated audit events, transaction hashes, block timestamps, and confirmation status.

`POST /api/blockchain/verify`
- Body contains an artifact identifier or uploaded artifact and the expected experiment/round/client context.
- Recomputes the documented hash and compares it with the recorded hash.
- Returns `match: true|false` with the relevant record.

`GET /api/metrics/history?task_type=classification&training_id=...`
- Returns per-round and per-client metrics, global evaluation metrics, and baseline identifiers.

`GET /api/experiments`
- Lists experiments with task, configuration, status, and timestamps.

---

## 11. Frontend Requirements

Use React 18, Tailwind CSS, React Router, Axios, Recharts, and ethers.js if the frontend needs direct blockchain interaction. Prefer backend-provided audit data for consistent event interpretation; direct wallet interaction is optional for this local prototype.

### 11.1 Dashboard `/`

- Experiment selector and task selector.
- Current round and job status.
- Three client status cards.
- Per-client and global metric charts.
- Recent blockchain event feed.
- Hardware/resource status where available.
- Clearly distinguish measured metrics from mock/demo data.

### 11.2 Classification `/classify`

- Image upload.
- Dataset/head selection where required.
- Predicted class and probabilities.
- Accuracy/macro-F1 charts based on stored evaluation runs.
- No implication that confidence equals calibrated clinical certainty.

### 11.3 Segmentation `/segment`

- MRI image upload.
- Original image and predicted mask/overlay.
- Dice and IoU only for labeled evaluation.
- Model version and preprocessing details.

### 11.4 Super-resolution `/superres`

- Image upload and x2/x4/x8 selector.
- Before/after comparison.
- PSNR/SSIM for paired-reference evaluation only.

### 11.5 Fusion `/fusion`

- Two image uploaders and modality-pair selector.
- Three-panel display: source A, fused output, source B.
- Document whether the images were registered/aligned.
- Display only valid computed metrics.

### 11.6 Blockchain explorer `/blockchain`

- Event table with experiment, round, client, action, hash, timestamp, transaction hash, and status.
- Filters by experiment, task, round, client, and action.
- Artifact hash verification.
- Counts of completed rounds and transactions.
- Clear separation between transaction inclusion and proof of training correctness.

### 11.7 Training control `/train`

- Task selection and validated configuration.
- Start/cancel/resume controls.
- Round and client progress.
- Loss and evaluation curves.
- Client failure and retry display.
- Current job state and checkpoint version.

All pages must use real API data. Any mock mode must be explicitly labelled and must never be mixed with real experiment records.

---

## 12. Environment Configuration

Provide `.env.example` with placeholders only:

```dotenv
GANACHE_URL=http://127.0.0.1:8545
GANACHE_CHAIN_ID=1337
CONTRACT_ADDRESS=
SERVER_PRIVATE_KEY=
CLIENT1_PRIVATE_KEY=
CLIENT2_PRIVATE_KEY=
CLIENT3_PRIVATE_KEY=

DATABASE_URL=sqlite:///./omnifm.db
API_PORT=8000
SPALIGN_LAMBDA=0.1
GSKR_TOP_K=2
EMBEDDING_DIM=128
MAX_KNOWLEDGE_BANK_SIZE=1000
ARTIFACT_DIR=./artifacts
```

Never commit the actual `.env` or private keys. Validate chain ID and contract address on startup. Fail clearly if blockchain auditing is required but the configured chain/contract is unavailable.

---

## 13. Testing Strategy

### 13.1 Unit tests

- Spectral encoding shapes and finite values.
- GSKR retrieval for empty, one-entry, and multi-entry banks.
- ECA shape compatibility and backward pass.
- PSP adapters for CNN classification, segmentation, SR, and fusion.
- SPAlign behavior with valid and absent prototypes.
- Deterministic artifact hashing.
- Aggregation with known small parameter tensors.
- Protection of personalized and incompatible parameters.
- Metric implementations against known examples.
- Contract authorization and duplicate-submission protection.

### 13.2 Integration tests

- Register clients and verify wallet addresses.
- Run one local client update and verify its metadata/hash.
- Run one complete two-client round.
- Confirm aggregation occurs once only after submission conditions are met.
- Verify global checkpoint hash against the saved artifact.
- Persist/reload the knowledge bank.
- Recover a training job from a checkpoint.
- Confirm blockchain failure is surfaced according to policy.

### 13.3 Smoke tests

Run in this order:
1. Environment and GPU check.
2. Dataset loading and visualization.
3. One forward/backward pass for each task.
4. One client training epoch.
5. Two-client, one-round FedAvg baseline.
6. One OmniFM-enabled round.
7. Blockchain registration/submission/aggregation audit.
8. API integration.
9. Frontend integration.
10. Reduced multi-round run with VRAM monitoring.

---

## 14. Implementation Plan

### Phase 0 — Repository and environment
- Create repository, environment, configuration and documentation.
- Pin compatible package versions.
- Add health checks, logging, and a reproducible seed/config system.

**Exit criteria:** Environment imports correctly; GPU status is reported; no secrets or datasets are tracked by Git.

### Phase 1 — Data and baseline task adapters
- Implement loaders, preprocessing, splits and metrics.
- Implement task models without OmniFM integration.
- Run forward/backward and evaluation smoke tests.

**Exit criteria:** Every task can train and evaluate on a tiny controlled subset.

### Phase 2 — OmniFM components
- Implement spectral encoder, GSKR, ECA, PSP, SPAlign.
- Define task-specific integration adapters.
- Add unit tests and shape/gradient checks.

**Exit criteria:** Every component and each task's full forward/backward pass succeeds.

### Phase 3 — Federated orchestration
- Implement logical clients, server, round state machine, and weighted FedAvg.
- Define shared/personalized parameter policy.
- Persist checkpoints and the knowledge bank.

**Exit criteria:** A two-client one-round run produces a valid global checkpoint and reproducible metrics.

### Phase 4 — Blockchain contract and connector
- Implement and test client registration, update submissions, round checks, aggregation event, and global hash storage.
- Connect training lifecycle events to contract calls.

**Exit criteria:** Duplicate or unauthorized submissions are rejected; saved artifact hashes can be verified.

### Phase 5 — API and persistence
- Add experiment/job lifecycle, cancellation, resume, prediction and metrics endpoints.
- Add schema validation and error handling.

**Exit criteria:** API tests pass using real training results and audit events.

### Phase 6 — Frontend
- Implement dashboard, task pages, training control and blockchain explorer.
- Connect to actual API data and label any demo mode.

**Exit criteria:** All routes load, API errors are handled, and displayed values match persisted records.

### Phase 7 — Experiments and analysis
- Run baseline and OmniFM configurations on identical splits.
- Add ablations for GSKR, ECA, PSP and SPAlign.
- Report resource use and performance honestly.

**Exit criteria:** Results are reproducible enough to compare, including cases where OmniFM does not outperform a baseline.

---

## 15. Definition of Done

### Software acceptance criteria

- [ ] All four task adapters load data and complete a forward/backward smoke test.
- [ ] Each task produces valid predictions and its appropriate evaluation metrics.
- [ ] At least one complete FL round runs with compatible aggregation.
- [ ] Personalized parameters remain local according to configuration.
- [ ] The knowledge bank updates, saves and reloads.
- [ ] Every accepted client update has a linked audit record when audit mode is enabled.
- [ ] The global checkpoint hash matches the saved artifact.
- [ ] Duplicate submissions and duplicate aggregation are prevented.
- [ ] Training jobs report accurate lifecycle states and errors.
- [ ] API endpoints pass integration tests.
- [ ] All seven frontend pages display real data.
- [ ] Training can run sequentially within the measured hardware limits for the tested configuration.
- [ ] Secrets, datasets, and large artifacts are excluded from source control.

### Research evaluation criteria

- [ ] Compare against task-appropriate FedAvg baselines using identical data splits and comparable training budgets.
- [ ] Report accuracy/macro-F1 for classification, Dice/IoU for segmentation, PSNR/SSIM for SR, and documented fusion metrics.
- [ ] Run component ablations where computationally feasible.
- [ ] Record seeds, configurations, dataset splits, software versions, and checkpoints.
- [ ] Report measured outcomes without hard-coding an expected win.

**Research target, not a software guarantee:** demonstrate measurable improvements from one or more OmniFM components under explicitly defined heterogeneous-client settings. If OmniFM does not outperform a baseline, report and investigate that result rather than modifying evaluation rules.

---

## 16. Instructions for AI Coding Agents

1. Read this PRD and `AGENTS.md` before editing code.
2. Implement one phase at a time and satisfy its exit criteria before advancing.
3. Do not generate the entire project as one untested code dump.
4. Do not create fake metrics, fabricated blockchain transactions, or placeholder model outputs that look real.
5. Do not assume client model heads or all parameters can be averaged.
6. Do not claim an algorithm matches the paper unless its implementation has been checked against the described mechanism.
7. Write tests alongside each component.
8. Use small synthetic inputs for fast tests; reserve full datasets and multi-round training for explicit experiments.
9. Keep all data loading, task losses, metrics, model adapters, aggregation, audit recording, and API handlers in separate modules.
10. Record architectural deviations and explain why they were necessary.
11. Do not change the experiment protocol silently; store the effective configuration with every run.
12. If a required dataset, pairing, metric reference, package API, or paper detail is unavailable, document the uncertainty and ask for a decision rather than inventing facts.
13. Prioritize correctness and a working vertical slice over superficial UI completeness.
14. Keep blockchain optional for local algorithm tests, but enforce the configured audit policy in full integration runs.
15. Treat the output as a research prototype, not a clinically validated medical system.
