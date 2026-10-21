# FederatedLearning Smart Contract Specification (Task 1.1)

## 1. Purpose and Trust Boundary

`FederatedLearning.sol` provides an on-chain audit log and round-state guard for the OmniFM federated learning prototype on a local Ethereum-compatible chain (Ganache / Hardhat).

- **Off-chain numerical aggregation:** The FL server validates client tensor updates and performs weighted FedAvg off-chain. The smart contract does **not** perform tensor arithmetic or automatic callback aggregation; instead, it enforces authorization, duplicate-submission prevention, quorum verification, and state-transition ordering.
- **Provenance vs. honesty:** A recorded SHA-256 hash (`bytes32`) proves that a specific serialized artifact existed and was committed at a given block timestamp by a registered wallet. It does not by itself prove that local training was performed honestly.
- **Privacy:** Only public wallet addresses, non-sensitive identifiers (`experimentId`, `taskType`, `round`, `clientId`), and 32-byte SHA-256 artifact hashes (`bytes32`) are stored on-chain. Raw images, patient identifiers, model weights, and spectral embeddings are never stored on-chain.

---

## 2. Actors and Authorization Rules

| Actor | On-chain Identity | Permitted Operations |
|---|---|---|
| **FL Server (`owner`)** | Deployer address (`msg.sender` at deployment) | `registerHospital`, `setExpectedClientCount`, `triggerAggregation`, `storeGlobalModel` |
| **Hospital Client** | Registered wallet (`Hospital.wallet`) mapped to a unique `clientId` | `submitModelUpdate` (must sign with the exact wallet registered for `clientId`) |
| **Anyone (Read-only)** | Any caller | `checkAllSubmitted`, `getHospital`, `getRoundStatus`, `getClientSubmission`, `getGlobalModelHash`, paginated list queries |

### Authorization Checks
1. **Unregistered wallet rejection:** `submitModelUpdate` reverts if `msg.sender` is not registered or does not match the registered wallet for the supplied `clientId`.
2. **Duplicate hospital registration prevention:** Re-registering an existing `clientId` or an already-assigned `wallet` address reverts.
3. **Server-only guards:** `registerHospital`, `triggerAggregation`, and `storeGlobalModel` revert if `msg.sender != owner`.

---

## 3. Experiment, Task, Round, and Client Identity

Every round and client submission is keyed by:
- `experimentId` (`string`, defaults to `"default"` in convenience overloads)
- `round` (`uint256`, `0` for initial global checkpoint, `>= 1` for federated communication rounds)
- `clientId` (`string`, e.g., `"client_1"`, `"client_2"`, `"client_3"`)
- `taskType` (`string`, e.g., `"classification"`, `"segmentation"`, `"superresolution"`, `"fusion"`)

Composite storage keys:
- Round key: `keccak256(abi.encode(experimentId, round))`
- Submission key: `keccak256(abi.encode(experimentId, round, clientId))`

Using `abi.encode` (rather than `abi.encodePacked`) prevents hash collisions across variable-length string fields (`experimentId`, `clientId`).

---

## 4. Round State Machine and Allowed Transitions

### 4.1 Expected Clients Policy
- By default, a round requires submissions from all currently registered hospitals (`expectedClients = registeredHospitalCount`, must be `>= 1`).
- An experiment may optionally lock or specify its expected client count when the first action on that experiment occurs.

### 4.2 Round 0 (Initial Global Model)
- `round == 0` represents the initial server model before client training.
- Allowed transition: `UNINITIALIZED -> GLOBAL_MODEL_STORED` via `storeGlobalModel(experimentId, 0, taskType, globalModelHash)` called by `owner`.
- Calling `storeGlobalModel` a second time for `(experimentId, 0)` reverts (`GlobalModelAlreadyStored`).

### 4.3 Communication Rounds (`round >= 1`)

```text
[CollectingUpdates]
   │  (each registered client calls submitModelUpdate once)
   ▼
[AllSubmitted]  (submissionCount == expectedClients)
   │  (server validates updates off-chain, runs FedAvg, calls triggerAggregation)
   ▼
[Aggregated]    (round.aggregated == true)
   │  (server saves checkpoint, computes SHA-256, calls storeGlobalModel)
   ▼
[GlobalModelStored] (round.globalModelStored == true)
```

### 4.4 Transition Guards and Duplicate Handling
1. **`submitModelUpdate`:**
   - Requires `round >= 1`.
   - Requires `modelHash != bytes32(0)`.
   - Requires caller `msg.sender` to be the registered wallet for `clientId`.
   - Reverts with `DuplicateSubmission` if `(experimentId, round, clientId)` has already been submitted.
   - Reverts with `RoundAlreadyAggregated` if aggregation has already been triggered for `(experimentId, round)`.
2. **`checkAllSubmitted`:**
   - Returns `true` if and only if `registeredHospitalCount > 0` and `round.submissionCount >= expectedClientsForRound`.
3. **`triggerAggregation`:**
   - Requires `msg.sender == owner`.
   - Requires `round >= 1`.
   - Reverts with `IncompleteRoundSubmissions` if `checkAllSubmitted(experimentId, round)` is `false`.
   - Reverts with `DuplicateAggregation` if `round.aggregated` is already `true`.
   - Sets `round.aggregated = true` and `round.aggregatedAt = block.timestamp`.
4. **`storeGlobalModel`:**
   - Requires `msg.sender == owner`.
   - Requires `globalModelHash != bytes32(0)`.
   - If `round >= 1`, reverts with `AggregationNotTriggered` unless `round.aggregated == true`.
   - Reverts with `GlobalModelAlreadyStored` if `round.globalModelStored == true`.
   - Stores `round.globalModelHash = globalModelHash`, `round.globalModelStored = true`, and `round.globalStoredAt = block.timestamp`.

---

## 5. Event Schema

```solidity
event HospitalRegistered(
    string clientId,
    string name,
    address indexed wallet,
    uint256 timestamp
);

event ModelUpdateSubmitted(
    string experimentId,
    uint256 indexed round,
    string clientId,
    address indexed wallet,
    string taskType,
    bytes32 modelHash,
    uint256 timestamp
);

event AggregationTriggered(
    string experimentId,
    uint256 indexed round,
    uint256 submissionCount,
    uint256 timestamp
);

event GlobalModelStored(
    string experimentId,
    uint256 indexed round,
    string taskType,
    bytes32 globalModelHash,
    uint256 timestamp
);
```

---

## 6. Bounded Query Interface

To avoid unbounded dynamic array returns as history grows:
- `getHospital(string clientId)` returns `(string clientId, string name, address wallet, bool registered, uint256 registeredAt)`.
- `getHospitalByWallet(address wallet)` returns the hospital record for that wallet.
- `getRegisteredHospitalCount()` returns total registered hospitals.
- `getRegisteredClientIds(uint256 offset, uint256 limit)` returns a bounded slice of registered `clientId` strings.
- `getClientSubmission(string experimentId, uint256 round, string clientId)` returns `(bytes32 modelHash, string taskType, address submitter, uint256 timestamp, bool exists)`.
- `getRoundStatus(string experimentId, uint256 round)` returns `(uint256 submissionCount, uint256 expectedCount, bool allSubmitted, bool aggregated, bool globalModelStored, bytes32 globalModelHash, uint256 aggregatedAt, uint256 globalStoredAt)`.
- `getGlobalModelHash(string experimentId, uint256 round)` returns `bytes32`.
