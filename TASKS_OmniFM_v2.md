# TASKS.md — OmniFM Blockchain Federated Learning Build Checklist

**Source of truth:** `PRD.md` for product/research requirements; `AGENTS.md` for coding-agent rules.  
**Execution rule:** Work top to bottom, one task at a time. Mark `[x]` only after its acceptance checks actually pass. Record commands and outcomes for non-trivial checks. Do not fabricate metrics, transactions, or test results.

## Global rules

- Start with small synthetic/unit tests, then tiny data subsets, then full datasets.
- The first full vertical slice is one task, one client, one round; then two clients; then all three clients.
- Simulate clients sequentially on the RTX 3050. Measure actual peak VRAM; do not assume all configurations fit.
- Batch sizes and local epochs below are initial defaults, not guarantees. Do not silently change image sizes, dataset splits, metric definitions, or experiment configuration.
- Keep raw images and patient identifiers local. Do not log sensitive data. Treat model updates and spectral embeddings as potentially sensitive.
- Only aggregate explicitly configured compatible shared parameters. Keep personalized parameters local.
- Blockchain audit records are not proof that training was honest. Record transaction status accurately.
- The classification datasets have different label semantics. Do not average their independent classifier heads unless an explicit compatible label space has been designed and validated.
- Fusion metrics must use a documented evaluation protocol; do not invent reference images or scores.
- Do not run full 20-round experiments until the user approves each task's full run, as specified in Phase 8.

---

## Phase 0 — Repository and Environment

- [x] **0.1** Inspect the existing repository before creating or moving files. Create only missing folders/files from `PRD.md`; update `.gitignore` to exclude `.env`, `venv/`, `node_modules/`, datasets, artifacts, checkpoints (`*.pt`, `*.pth`), databases, and generated outputs.
  - **Check:** `tree -L 2` (or an equivalent Windows listing) is reviewed against the PRD; existing files were not overwritten or moved without approval.
- [x] **0.2** Add `.env.example` with placeholders only and implement validated `backend/config.py` settings for chain, database, artifact directory, model dimensions, top-k, alignment lambda, rounds, batch sizes, and precision policy.
  - **Check:** importing settings succeeds without exposing secrets; invalid configuration produces an actionable error.
- [x] **0.3** Verify the current Python 3.11, PyTorch, CUDA, and GPU environment before installing or changing packages.
  - **Check:** print Python version, `torch.__version__`, CUDA availability, GPU name, and allocated/reserved memory where applicable. If the GPU differs from the target, report it rather than assuming RTX 3050.
- [x] **0.4** Inspect dataset archives and directories without modifying source files. With user approval, extract FeTS2022 and BreaKHis into documented destination folders.
  - **Check:** list archive contents and resulting structure; document license/source and any missing files.
- [x] **0.5** Establish logging, random-seed/config capture, and a basic health check.
  - **Check:** a small test run records its effective configuration and seed; logs contain no patient identifiers or private keys.

## Phase 1 — Blockchain Audit

- [x] **1.1** Specify the contract state machine and event schema before implementation: experiment ID, task, round, client ID, wallet, update hash, global hash, and allowed transitions.
  - **Check:** design explicitly defines expected clients, duplicate handling, authorization, and whether aggregation is recorded by the server after validation.
- [x] **1.2** Implement `FederatedLearning.sol`: hospital registration, client update submission, submission checks, aggregation event, global-model hash storage, and audit events/queries.
  - **Check:** `npx hardhat compile` succeeds using the pinned compiler.
- [x] **1.3** Add Hardhat tests for registration, authorized submitters, duplicate submissions, unregistered wallets, incomplete rounds, duplicate aggregation, global hash storage, and experiment/round separation.
  - **Check:** all tests pass on the configured local Hardhat/Ganache setup. Do not require an “automatic aggregation” contract callback unless the chosen design explicitly implements it; the contract should record/guard the event while the FL server performs numerical aggregation.
- [x] **1.4** Implement `deploy.js` for local Ganache deployment and print the deployed address. Write the address to `.env` only through an explicit local setup step; never overwrite unrelated environment entries silently.
  - **Check:** deployed address, chain ID, and contract bytecode are verified.
- [x] **1.5** Implement `backend/blockchain/web3_connector.py`: register client, submit update hash, record aggregation, store global hash, read audit events, confirm transaction receipts, and report failures.
  - **Check:** a local integration test registers three dummy clients, records dummy hashes and reads back matching events. Use development-only keys.
- [x] **1.6** Implement deterministic `hash_state_dict()`/artifact hashing.
  - **Check:** identical serialized state produces the same SHA-256 hash; changed tensor bytes, dtype, or shape change the canonical artifact representation/hash. Document exactly what is hashed.

## Phase 2 — OmniFM Components

- [x] **2.1** `backend/omnifm/spectral.py`: FFT magnitude processing, documented low-pass mask, learned spectral projection, normalization.
  - **Check:** output shape `(B, D)` where `D` is configurable (default 128); finite values and gradients for RGB/grayscale and odd/even image sizes.
- [x] **2.2** `backend/omnifm/gskr.py`: knowledge-bank storage, add/update, capacity/pruning, save/load, cosine top-k retrieval.
  - **Check:** empty bank, one prototype, fewer than `k`, full bank and reload cases pass. Retrieval avoids accidental self-retrieval and supports task-aware filtering where configured.
- [x] **2.3** `backend/omnifm/eca.py`: cross-attention with backbone queries and retrieved prototypes as keys/values.
  - **Check:** documented input/output shapes, valid head dimensions, finite forward/backward pass.
- [x] **2.4** `backend/omnifm/psp.py`: prefix + backbone + client-specific suffix/prompt handling.
  - **Check:** token dimensions align; personalized parameters are identifiable and can be excluded from aggregation.
- [x] **2.5** `backend/omnifm/spalign.py`: alignment loss and task-loss combination.
  - **Check:** valid-prototype and no-prototype behavior are tested; task loss and alignment loss can be logged separately.
- [x] **2.6** Define a reusable OmniFM composition layer and task adapters. Do not assume a single token concatenation works directly on all CNN feature maps.
  - **Check:** classification, segmentation, SR and fusion adapters document how spatial feature maps become tokens and how outputs/skip connections retain spatial structure.
- [x] **2.7** Add unit tests for each component and a full forward/backward smoke test per adapter.
  - **Check:** shapes, finite outputs, gradients, CPU tests and optional CUDA tests pass. Record measured peak VRAM; do not enforce an arbitrary 500 MB component limit as a universal correctness criterion.

## Phase 3 — Dataset Loaders and Splits

- [x] **3.1** Implement MedMNIST loaders for DermaMNIST, PathMNIST and BloodMNIST using the actual archive schema.
  - **Check:** inspect available keys and shapes; use official train/validation/test splits unless a documented alternative is approved. Do not silently replace the provided splits with an 80/20 split.
  - **Check:** labels and class counts are logged as aggregate statistics only. Each dataset's label semantics remain distinct.
- [x] **3.2** Implement FeTS2022 loader for T1ce images and corresponding segmentation masks, with consistent intensity normalization and 2D slice extraction.
  - **Check:** image/mask alignment, non-empty slice policy, subject-level split, dimensions and value ranges are verified. Claim institution-based splits only if institution metadata supports them.
- [x] **3.3** Implement BreaKHis loader and patch extraction for 64×64 low-resolution inputs with x2/x4/x8 paired high-resolution targets.
  - **Check:** split by patient/subject before extracting patches to prevent leakage; document degradation/downsampling kernel and magnification-based non-IID partition policy.
- [x] **3.4** Implement fusion loader for verified paired CT–MRI, PET–MRI and SPECT–MRI examples that actually exist in the selected dataset.
  - **Check:** verify filenames, modality pairs, image registration/alignment assumptions, license/source, dimensions and intensity handling. Do not assume each folder contains aligned pairs.
- [x] **3.5** Add data-loader tests and dataset inspection output.
  - **Check:** one batch from each available loader has expected shapes, labels/targets and value ranges; source files remain unchanged; patient-level leakage checks pass where IDs are available.

## Phase 4 — Task Models and Metrics

- [x] **4.1** Classification adapter: ResNet-18 adapted for 28×28 inputs, spectral integration, and dataset/client-appropriate classifier heads.
  - **Check:** each head's label mapping is explicit; incompatible heads are never averaged. Forward/backward and prediction tests pass.
- [x] **4.2** Segmentation adapter: lightweight U-Net/MONAI-compatible model with OmniFM integration at a documented feature stage.
  - **Check:** output mask matches target spatial dimensions; skip connections remain valid; loss and evaluation work on a real labeled batch.
- [x] **4.3** Super-resolution adapter: lightweight SR network with scale-selectable x2/x4/x8 output.
  - **Check:** output dimensions match scale; paired LR/HR generation is reproducible; forward/backward works.
- [x] **4.4** Fusion adapter: documented U2Fusion-inspired model accepting a verified image pair.
  - **Check:** input pairing and output shape are correct; training objective and metric protocol are documented.
- [x] **4.5** Implement and test metrics in `backend/utils/metrics.py`: accuracy, macro-F1, Dice, IoU, PSNR, SSIM, VIF, and L1 where appropriate.
  - **Check:** metric implementations pass known toy examples. Metrics requiring a reference are never returned as measured results without that reference; fusion no-reference metrics are explicitly identified.
- [x] **4.6** Run one forward/backward pass per task using a tiny real batch where available.
  - **Check:** record precision mode, actual peak VRAM and result. Use the configured precision policy; if memory fails, report the smallest justified adjustment rather than silently changing image size or evaluation rules.

## Phase 5 — Federated Learning

- [x] **5.1** Define shared and personalized parameter policies for each task/model before coding aggregation.
  - **Check:** a policy manifest identifies parameters that are shared, personalized, or task-specific; incompatible heads and local suffix prompts are excluded from FedAvg.
- [x] **5.2** Implement client training: load local data, receive shared state/retrieval context, train locally, retain local parameters, and return eligible shared updates, sample count, permitted spectral prototypes and metrics.
  - **Check:** raw images stay local; update metadata is validated; task loss and SPAlign loss are logged separately.
- [x] **5.3** Implement weighted FedAvg for compatible shared parameters only.
  - **Check:** toy-tensor tests verify sample-count weighting, parameter-name/shape validation, finite values, and exclusion of personalized/incompatible parameters.
- [x] **5.4** Implement server round manager/state machine: initialize model, record configured round-0 hash, dispatch clients sequentially, validate submissions, aggregate once, update/persist knowledge bank, save/hash global checkpoint, record audit event and metrics.
  - **Check:** duplicate aggregation is prevented; partial/failing client policy is explicit; checkpoint and bank can be reloaded.
- [x] **5.5** Implement SQLite persistence for experiment configuration, round/client status, metrics, artifact metadata and blockchain transaction references.
  - **Check:** records survive process restart and correspond to actual run results.
- [x] **5.6** Run a one-round classification smoke test on a small subset, first with one client, then two, then all three.
  - **Check:** expected client updates are recorded, compatible parameters aggregate, global hash verifies, metrics persist, and peak VRAM is reported.
- [x] **5.7** Run one-round smoke tests for segmentation, SR and fusion on tiny valid subsets.
  - **Check:** each task completes forward/backward, evaluation follows its valid metric protocol, and all checkpoints/audit events are linked to the correct task/round.
- [x] **5.8** Add FedProx/FedPer baselines only after defining task-appropriate parameter sharing and matching data splits/training budgets.
  - **Check:** baseline configuration is recorded and tests confirm it implements the stated algorithm rather than only changing a label/flag.

## Phase 6 — API and Persistence

- [x] **6.1** Implement SQLAlchemy models and migrations/schema setup for clients, experiments, rounds, client updates, global model versions, metrics, artifacts and blockchain transactions.
  - **Check:** database initialization and persistence/reload tests pass.
- [x] **6.2** Implement FastAPI app, CORS for configured local frontend origin(s), health route and modular routers.
  - **Check:** app starts with the documented command; `/docs` loads; health reports database/GPU/blockchain status without exposing secrets.
- [x] **6.3** Implement asynchronous training start/status/log/cancel/resume endpoints with explicit job lifecycle states.
  - **Check:** start returns promptly; errors are visible; cancellation occurs at a safe boundary; resume requires a valid checkpoint; concurrent-job policy is enforced.
- [x] **6.4** Implement prediction endpoints for classification, segmentation, SR and fusion.
  - **Check:** each loads a compatible checkpoint and returns output metadata. Dice/IoU/PSNR/SSIM are only returned when valid reference data or a benchmark evaluation is supplied.
- [x] **6.5** Implement blockchain audit log, artifact hash verification, experiment list and metrics-history endpoints.
  - **Check:** paginated responses match persisted/on-chain data; distinguish transaction inclusion from proof of training correctness.
- [x] **6.6** Add API integration tests for invalid input, missing checkpoint, unavailable blockchain, training failure, and normal successful responses.
  - **Check:** HTTP status codes and error payloads are documented and tests pass.

## Phase 7 — Frontend

- [x] **7.1** Inspect existing frontend tooling; configure React, Tailwind, routing, layout and Axios API client without assuming a Vite/CRA command that is not in `package.json`.
  - **Check:** install/build succeeds using the package manager and scripts declared by the project.
- [x] **7.2** Dashboard: experiment/task selector, round state, client cards, metrics charts, recent audit events and resource status.
  - **Check:** all displayed data comes from the API or is explicitly labelled as demo data.
- [x] **7.3** Classification page with image upload, dataset/head selection where required, probabilities and evaluation history.
- [x] **7.4** Segmentation page with original image, mask/overlay, model version, and valid labeled-evaluation metrics.
- [x] **7.5** Super-resolution page with x2/x4/x8 selection and before/after view; only show reference-based metrics when available.
- [x] **7.6** Fusion page with two image uploads, modality-pair validation, fused output and documented metric status.
- [x] **7.7** Blockchain explorer with experiment/round/client filters, event history, transaction status and artifact hash verification.
- [x] **7.8** Training control page with validated configuration, start/cancel/resume, polling, round/client progress, logs and errors.
  - **Check for Phase 7:** walk through all routes with backend running; verify loading, empty, success and failure states. No fake metric values may appear as real results.

## Phase 8 — Experiments, Baselines and Documentation

- [x] **8.1** Obtain user approval before each full 20-round task run. Record configuration, seeds, split IDs, software versions, start/end status, runtime and peak VRAM.
  - **Check:** run completes or has a documented failure; checkpoints and metrics are saved.
- [x] **8.2** Run task-appropriate FedAvg baselines and then FedProx/FedPer where their assumptions are valid.
  - **Check:** use comparable splits, evaluation procedures and training budgets; record exact configs.
- [x] **8.3** Run ablations for GSKR, ECA, PSP and SPAlign where computationally feasible.
  - **Check:** disabled components are truly bypassed, and each experiment is traceable to its configuration.
- [x] **8.4** Export measured results to `results/results.md` and CSV.
  - **Check:** include metric definitions, client setup, baseline configuration, variability/seeds where feasible, hardware and limitations. Do not hard-code the expectation that OmniFM wins.
- [x] **8.5** Write README setup, local Ganache setup, dataset setup, test commands, experiment instructions, troubleshooting and demo walkthrough.
  - **Check:** a clean setup walkthrough succeeds without committed secrets or datasets.
- [x] **8.6** Final audit: verify no private keys, patient data, datasets or large model artifacts are tracked; verify all acceptance criteria and note unfinished items.
  - **Check:** repository status and final checklist are reviewed; research limitations are explicit.

---

## Current Task

The coding agent must select the first unchecked task and work only on that task unless the user explicitly requests otherwise. Do not mark a task `[x]` until its stated check passes.
