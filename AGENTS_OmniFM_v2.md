# AGENTS.md — OmniFM Blockchain Federated Learning

## 1. Required Reading and Task Discipline

1. Read `PRD.md` before changing code.
2. Read `TASKS.md` and select the first unfinished task unless the user explicitly names another task.
3. Work on **one task at a time**. Do not start later phases until the current task's acceptance criteria pass.
4. Before editing, state a short plan and list the files expected to change.
5. Implement the smallest complete change, run its tests/smoke test, and report the actual command and result.
6. Update `TASKS.md` only when the task's acceptance criteria are met. Do not mark work complete just because code was written.
7. If a required detail is ambiguous or conflicts with the PRD, identify the conflict and ask before making a consequential architectural choice.
8. Do not fabricate test output, training metrics, blockchain transactions, or completed features.

## 2. Project Summary

This is a local research prototype for federated medical-image learning across three simulated hospital clients. It combines task-specific models, OmniFM-inspired spectral mechanisms (FFT spectral encoding, GSKR, ECA, PSP, SPAlign), a federated training lifecycle, a server-side spectral knowledge bank, a FastAPI service, a React dashboard, and an optional local Ganache audit layer.

Blockchain auditing is an engineering extension to OmniFM, not a component of the original paper. This project is not a clinically validated medical system and must not claim production HIPAA/GDPR compliance.

## 3. Hard Constraints

### Hardware and execution

- Target hardware: NVIDIA RTX 3050 6 GB VRAM and 16 GB RAM.
- Initially simulate three logical clients sequentially on one machine; do not keep multiple clients' training batches/models resident on GPU at the same time.
- Release client-specific GPU allocations when practical between clients.
- Use the configured image sizes as defaults: classification 28×28, segmentation 128×128, SR input patches 64×64, fusion 128×128.
- Start with a one-round, one-epoch, tiny-subset smoke test before long runs.
- Measure actual peak GPU memory using `torch.cuda.max_memory_allocated()` when CUDA is available.
- Do not silently change image dimensions, task definitions, or evaluation protocol to avoid an error. If memory is insufficient, first reduce batch size, then use gradient accumulation where valid, then consider a documented model/embedding reduction with approval.
- CPU execution must remain available for unit tests and debugging.

### Python and dependencies

- Use Python 3.11 in `./venv` unless the existing repository environment explicitly establishes another working version.
- Inspect the installed PyTorch/CUDA environment before changing dependencies. Do not reinstall or upgrade PyTorch without explicit need and approval.
- Pin compatible dependencies after confirming their APIs. Do not assume a CUDA toolkit label alone determines the correct PyTorch package.
- Use the current supported PyTorch AMP API, such as `torch.amp.autocast` and `torch.amp.GradScaler` where appropriate for the installed PyTorch version. Handle CPU and CUDA paths correctly.
- Do not require FP16 for operations or devices where it is unsupported or numerically unsafe. Use a documented precision policy and test numerical stability.

### Local-only and privacy

- Keep the default build free and local: Ganache, local database, local datasets and local artifact storage. Do not introduce paid APIs, cloud services, public chains, or remote data upload without explicit approval.
- Ganache defaults to `http://127.0.0.1:8545`, chain ID 1337. Read contract address and development keys from `.env`; validate them at startup.
- Use only local development keys for Ganache. Never commit keys or reuse them in production.
- Raw medical images and patient identifiers must not be written to the blockchain or application logs, and must not be sent to the central server by the client-training workflow.
- Transmitted model updates and spectral embeddings can still be sensitive. Do not describe them as inherently anonymous or privacy-preserving. Document exactly which tensors/metadata are transmitted.
- Never commit `.env`, `venv/`, `node_modules/`, datasets, checkpoints (`*.pt`, `*.pth`), or generated artifacts.
- Do not edit, move, or delete files under `datasets/` unless the user explicitly authorizes it. Treat downloaded/extracted datasets as untrusted input.

## 4. Commands

Use commands appropriate to the actual shell and installed dependencies. Verify that scripts and paths exist before claiming a command is valid.

```bash
# Activate Python environment
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Install dependencies only after checking the existing environment
pip install -r requirements.txt

# Backend (from repository root; adjust module path only if the actual layout requires it)
uvicorn backend.main:app --reload --port 8000

# Blockchain
cd blockchain
npx hardhat compile
npx hardhat test
npx hardhat run scripts/deploy.js --network ganache

# Frontend (use the package manager and script actually defined in package.json)
cd frontend
npm install
npm run dev

# Tests (adjust to the repository's actual test layout)
pytest -q
```

Do not assume `npm start`, `backend/tests`, or a particular import path exists without checking the repository. Avoid destructive commands.

## 5. Code Style

### Python

- Add type hints to public functions and meaningful docstrings describing tensor shapes and behavior.
- Use Black-compatible formatting and avoid wildcard imports.
- Add shape comments or annotations at non-obvious tensor transformations, not mechanically on every line.
- Keep configuration in a validated settings module loaded from environment/config files. Avoid magic numbers for embedding dimensions, top-k, alignment weight, rounds, and batch sizes.
- Separate model code, task adapters, datasets, training, aggregation, persistence, API routes, and blockchain calls.
- Raise actionable exceptions; do not silently swallow failures.

### React

- Use functional components and hooks.
- Follow the project's existing build tooling (Vite or another already-configured bundler); do not assume `npm start` if it is not defined.
- Use Tailwind utilities if configured and keep components small and understandable.
- Display real API values as real values. Any demo/mock mode must be explicitly labelled.

### Solidity

- Use the Solidity compiler version pinned by Hardhat and the PRD (target `^0.8.19` if compatible).
- Emit events for meaningful state changes.
- Validate authorization, experiment/round identity, duplicate submissions, and allowed state transitions.
- Keep the contract simple and testable; do not enforce an arbitrary line-count limit at the cost of correctness.
- Do not store images, patient data, model weights, raw spectral embeddings, or private keys on-chain.

## 6. Architecture Rules

### 6.1 Task models and OmniFM adapters

1. OmniFM components should be reusable modules and should not import concrete task models. Task-specific adapters may import and compose OmniFM components.
2. Do not assume every task backbone naturally emits `(B, L, D)`. CNN backbones may emit `(B, C, H, W)`; an adapter must explicitly tokenize/project features and preserve spatial structure needed by decoders.
3. Each task adapter must define:
   - input and target format,
   - preprocessing,
   - backbone/feature representation,
   - OmniFM insertion point and tensor shapes,
   - task head/decoder,
   - task loss,
   - evaluation metrics,
   - prediction postprocessing.
4. The common FL orchestration lifecycle should be reusable, but task losses, data handling, evaluation, output shapes and eligible shared parameters are task-specific.
5. Do not force classification, segmentation, super-resolution and fusion into one identical model-forward implementation if doing so breaks spatial structure or task semantics.
6. Every component must document its input/output shapes and handle batch size 1.

### 6.2 Shared and personalized parameters

1. Define an explicit parameter-sharing policy for each task before aggregation.
2. Aggregate only parameters whose names, shapes, architecture, and semantics are compatible across participating clients.
3. Never average classification heads with incompatible label spaces or output dimensions.
4. Keep parameters configured as personalized—such as client-specific suffix/CLS prompts or personalized task heads—local and exclude them from FedAvg.
5. Do not assume every backbone or OmniFM parameter is shared merely because it has a matching shape. The policy must be explicit and tested.
6. Add tests proving shared parameters are aggregated and personalized/incompatible parameters are not.

### 6.3 Federated lifecycle

1. Implement weighted FedAvg for compatible shared parameters as the first baseline.
2. Use an explicit round state machine and prevent duplicate aggregation for a given experiment/round.
3. Define whether all clients are required. For the initial prototype, use all configured clients unless the PRD/config explicitly states otherwise.
4. Validate update metadata, client identity, task type, experiment ID, round ID, parameter names/shapes, finite tensor values, and sample counts before aggregation.
5. Save round metadata, per-client results, global checkpoints and errors for recovery.
6. Keep blockchain recording separate from numerical aggregation. The FL algorithm must be unit-testable without Ganache.
7. If blockchain auditing is required for an experiment and a transaction fails, report audit failure/pending status according to configured policy. Never silently claim the round was audited.

### 6.4 OmniFM spectral knowledge bank

1. The knowledge bank is server-managed and persisted to `backend/fl/knowledge_bank.pt` unless the repository establishes another documented path.
2. Define prototype metadata, capacity, update ordering, task/client context, retrieval filters, and pruning behavior.
3. GSKR must handle an empty bank and fewer than `k` prototypes.
4. Avoid self-retrieval where the current sample's token has just been inserted and would become its own retrieval target.
5. Do not mix semantically incompatible task prototypes without a deliberate, tested design.
6. Document what is transmitted from clients to the server. Do not assume embeddings are non-sensitive.
7. Test save/reload and deterministic retrieval behavior.

### 6.5 Hashing and blockchain

Record the following logical events:
1. Hospital/client registration.
2. Accepted client model-update hash per experiment/round/client.
3. Aggregation event after submission requirements pass.
4. Global-model hash per completed round.
5. Round-0/global initialization hash if configured by the PRD.

Do not record individual local training steps, broadcasts, or knowledge-bank updates on-chain unless the PRD is explicitly revised.

For hashes:
- Define exactly which artifact is hashed.
- Use deterministic serialization (for example, state-dict keys sorted consistently, with tensor dtype, shape, and bytes represented unambiguously).
- Use SHA-256 off-chain as configured by the PRD.
- Do not hash arbitrary Python object representations.
- Keep artifact bytes off-chain; record hashes and minimal metadata on-chain.
- Verify transaction receipts before marking events confirmed.
- Prevent unauthorized and duplicate logical submissions.
- Distinguish “the artifact matches its recorded hash” from “the training was honest/correct”; a hash proves the former, not the latter.

### 6.6 API and persistence

- Long-running training must be represented as a job; the start endpoint must return promptly.
- Persist experiment configuration, round/client states, metrics, artifact metadata and blockchain transaction references.
- Provide clear `CREATED`, `WAITING_FOR_CLIENTS`, `TRAINING`, `UPDATES_VALIDATED`, `AGGREGATING`, `COMPLETED`, `FAILED`, and `CANCELLED` states, or map to an equally explicit documented state machine.
- Prediction metrics that require ground truth (e.g. Dice, IoU, PSNR, SSIM) must not be fabricated when no reference is available.
- Keep large checkpoints/images in artifact storage, not directly in SQLite.
- Do not expose private keys in API responses, frontend code, logs, or database records.

## 7. Datasets

The following paths are expected based on the current project notes. Check that files exist before relying on them, and document discrepancies rather than inventing a dataset structure.

| Task | Expected location | Required checks |
|---|---|---|
| Classification | `datasets/medmnist/{dermamnist,pathmnist,bloodmnist}.npz` | Inspect actual NPZ keys/shapes; verify label meaning and train/validation/test separation |
| Segmentation | `datasets/fets2022/` | Identify archives/files; use T1ce and corresponding segmentation labels; prevent subject leakage |
| Super-resolution | `datasets/breakhis/` | Inspect archive layout; split by patient/subject before generating patches; document magnification-based client partitions |
| Fusion | `datasets/fusion/{CT-MRI,PET-MRI,SPECT-MRI}/` | Verify actual source, licenses, pair filenames, alignment, and availability of each modality pair |

Do not claim true institution-level partitions unless institution metadata supports them. If clients are simulated through random or magnification-based splits, label them as simulated non-IID clients.

For classification, DermaMNIST, PathMNIST, and BloodMNIST use different label semantics. Do not directly average their independent classifier heads. Use task/client-specific heads or define and validate an explicit common label space.

## 8. Testing Expectations

Before integrating a component into a full experiment, add tests for:

- Input/output shapes and batch size 1.
- Finite outputs and gradients.
- Empty/undersized knowledge bank behavior.
- CPU forward/backward; GPU tests when available.
- Aggregation with small known tensors and unequal sample counts.
- Exclusion of personalized/incompatible parameters.
- Deterministic artifact hashes and hash verification.
- Contract authorization, duplicate-submission rejection, and round-state transitions.
- Checkpoint and knowledge-bank persistence/reload.
- Metric correctness on known toy examples.
- API validation and failure states.

Before long training:
1. Run one client, one epoch, tiny subset.
2. Run a two-client, one-round FL test.
3. Run a full three-client, one-round test.
4. Confirm memory use, metric computation, checkpoint reload and audit records.
5. Only then start the configured multi-round experiment.

## 9. Workflow and Reporting

For every task:
1. State the plan and expected files to change.
2. Implement only that task.
3. Run the relevant test/smoke command.
4. Report exact commands and actual outcomes, including failures.
5. Update `TASKS.md` only if acceptance criteria pass.
6. Identify deviations from `PRD.md` and request approval for consequential changes.

Do not:
- generate the entire repository in one untested code dump,
- replace real training with hard-coded/demo metrics,
- claim tests pass without running them,
- change dataset files or silently change image sizes,
- average incompatible model parameters,
- commit secrets, datasets, or model checkpoints,
- claim clinical validity or formal privacy guarantees without evidence.

## 10. Reference Equations

Use the equations and definitions in `PRD.md` as the project's intended specification. The current summary is:

- Spectral token: `s = Pool(σ(W2 · σ(W1 · FreqMix(S_LP))))`, followed by the documented normalization.
- ECA: `Z = Attention(Q, K, V)` with queries from backbone features and keys/values from retrieved prototypes.
- PSP: `r' = [Z_prefix || r_backbone || c_client]`, with task-specific shape adapters.
- SPAlign: `L_align = mean(||s − mean(S_g)||²)`.
- Total loss: `L_total = L_task + lambda_align * L_align`, default `lambda_align = 0.1`.
- GSKR default `top_k = 2`, embedding dimension default `128`.

If the paper, PRD, and existing code disagree, report the discrepancy and resolve it explicitly. Do not silently claim paper fidelity.
