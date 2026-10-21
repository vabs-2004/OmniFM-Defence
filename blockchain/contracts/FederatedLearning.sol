// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FederatedLearning
 * @notice On-chain audit log and round-state guard for OmniFM Federated Learning.
 * @dev Numerical aggregation is performed off-chain by the FL server after validating
 *      client updates. This contract enforces hospital registration, submitter wallet
 *      authorization, duplicate submission rejection, quorum verification, and
 *      round lifecycle state transitions.
 */
contract FederatedLearning {
    string public constant DEFAULT_EXPERIMENT_ID = "default";

    address public immutable owner;

    struct Hospital {
        string clientId;
        string name;
        address wallet;
        bool registered;
        uint256 registeredAt;
    }

    struct ClientSubmission {
        bytes32 modelHash;
        string taskType;
        address submitter;
        uint256 timestamp;
        bool exists;
    }

    struct RoundStatus {
        uint256 submissionCount;
        uint256 expectedCount;
        bool initialized;
        bool aggregated;
        bool globalModelStored;
        bytes32 globalModelHash;
        string taskType;
        uint256 aggregatedAt;
        uint256 globalStoredAt;
    }

    string[] private _registeredClientIds;
    mapping(bytes32 => Hospital) private _hospitalsByIdHash;
    mapping(address => bytes32) private _clientIdHashByWallet;
    mapping(bytes32 => uint256) private _experimentExpectedClients;
    mapping(bytes32 => RoundStatus) private _rounds;
    mapping(bytes32 => ClientSubmission) private _submissions;

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

    error OnlyServerAuthorized(address caller);
    error InvalidAddress();
    error EmptyIdentifier(string field);
    error InvalidHash();
    error InvalidRoundNumber(uint256 round);
    error HospitalAlreadyRegistered(string clientId);
    error WalletAlreadyRegistered(address wallet);
    error UnregisteredHospital(string clientId);
    error UnauthorizedClientWallet(address caller, string clientId, address expectedWallet);
    error NoHospitalsRegistered();
    error DuplicateSubmission(string experimentId, uint256 round, string clientId);
    error RoundAlreadyAggregated(string experimentId, uint256 round);
    error IncompleteRoundSubmissions(
        string experimentId,
        uint256 round,
        uint256 submitted,
        uint256 expected
    );
    error DuplicateAggregation(string experimentId, uint256 round);
    error AggregationNotTriggered(string experimentId, uint256 round);
    error GlobalModelAlreadyStored(string experimentId, uint256 round);
    error InvalidExpectedClientCount(uint256 count);

    modifier onlyOwner() {
        if (msg.sender != owner) {
            revert OnlyServerAuthorized(msg.sender);
        }
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // =========================================================================
    // 1. Hospital Registration
    // =========================================================================

    /**
     * @notice Register a hospital client with explicit clientId, display name, and wallet.
     */
    function registerHospital(
        string calldata clientId,
        string calldata name,
        address wallet
    ) external onlyOwner {
        _registerHospitalInternal(clientId, name, wallet);
    }

    /**
     * @notice Convenience overload matching `registerHospital(name, wallet)` where clientId = name.
     */
    function registerHospital(string calldata name, address wallet) external onlyOwner {
        _registerHospitalInternal(name, name, wallet);
    }

    function _registerHospitalInternal(
        string calldata clientId,
        string calldata name,
        address wallet
    ) internal {
        if (bytes(clientId).length == 0) {
            revert EmptyIdentifier("clientId");
        }
        if (bytes(name).length == 0) {
            revert EmptyIdentifier("name");
        }
        if (wallet == address(0)) {
            revert InvalidAddress();
        }

        bytes32 idHash = _hashString(clientId);
        if (_hospitalsByIdHash[idHash].registered) {
            revert HospitalAlreadyRegistered(clientId);
        }
        if (_clientIdHashByWallet[wallet] != bytes32(0)) {
            revert WalletAlreadyRegistered(wallet);
        }

        _hospitalsByIdHash[idHash] = Hospital({
            clientId: clientId,
            name: name,
            wallet: wallet,
            registered: true,
            registeredAt: block.timestamp
        });
        _clientIdHashByWallet[wallet] = idHash;
        _registeredClientIds.push(clientId);

        emit HospitalRegistered(clientId, name, wallet, block.timestamp);
    }

    /**
     * @notice Optional configuration of expected client count for a specific experiment.
     */
    function setExperimentExpectedClients(
        string calldata experimentId,
        uint256 expectedCount
    ) external onlyOwner {
        if (bytes(experimentId).length == 0) {
            revert EmptyIdentifier("experimentId");
        }
        if (expectedCount == 0 || expectedCount > _registeredClientIds.length) {
            revert InvalidExpectedClientCount(expectedCount);
        }
        _experimentExpectedClients[_hashString(experimentId)] = expectedCount;
    }

    // =========================================================================
    // 2. Client Model Update Submission
    // =========================================================================

    /**
     * @notice Submit a client model-update hash for a specific experiment and round.
     */
    function submitModelUpdate(
        string calldata experimentId,
        uint256 round,
        string calldata taskType,
        bytes32 modelHash,
        string calldata clientId
    ) external {
        _submitModelUpdateInternal(experimentId, round, taskType, modelHash, clientId);
    }

    /**
     * @notice Convenience overload using the default experiment ID.
     */
    function submitModelUpdate(
        uint256 round,
        string calldata taskType,
        bytes32 modelHash,
        string calldata clientId
    ) external {
        _submitModelUpdateInternal(DEFAULT_EXPERIMENT_ID, round, taskType, modelHash, clientId);
    }

    function _submitModelUpdateInternal(
        string memory experimentId,
        uint256 round,
        string calldata taskType,
        bytes32 modelHash,
        string calldata clientId
    ) internal {
        if (bytes(experimentId).length == 0) {
            revert EmptyIdentifier("experimentId");
        }
        if (round == 0) {
            revert InvalidRoundNumber(round);
        }
        if (bytes(taskType).length == 0) {
            revert EmptyIdentifier("taskType");
        }
        if (modelHash == bytes32(0)) {
            revert InvalidHash();
        }

        bytes32 idHash = _hashString(clientId);
        Hospital storage hospital = _hospitalsByIdHash[idHash];
        if (!hospital.registered) {
            revert UnregisteredHospital(clientId);
        }
        if (msg.sender != hospital.wallet) {
            revert UnauthorizedClientWallet(msg.sender, clientId, hospital.wallet);
        }

        bytes32 rKey = _roundKey(experimentId, round);
        RoundStatus storage roundState = _rounds[rKey];
        if (roundState.aggregated) {
            revert RoundAlreadyAggregated(experimentId, round);
        }

        bytes32 sKey = _submissionKey(experimentId, round, clientId);
        if (_submissions[sKey].exists) {
            revert DuplicateSubmission(experimentId, round, clientId);
        }

        if (!roundState.initialized) {
            uint256 expected = _resolveExpectedClients(experimentId);
            roundState.expectedCount = expected;
            roundState.initialized = true;
            roundState.taskType = taskType;
        }

        _submissions[sKey] = ClientSubmission({
            modelHash: modelHash,
            taskType: taskType,
            submitter: msg.sender,
            timestamp: block.timestamp,
            exists: true
        });

        roundState.submissionCount += 1;

        emit ModelUpdateSubmitted(
            experimentId,
            round,
            clientId,
            msg.sender,
            taskType,
            modelHash,
            block.timestamp
        );
    }

    // =========================================================================
    // 3. Round Submission Check & Aggregation Recording
    // =========================================================================

    function checkAllSubmitted(
        string calldata experimentId,
        uint256 round
    ) external view returns (bool) {
        return _checkAllSubmittedInternal(experimentId, round);
    }

    function checkAllSubmitted(uint256 round) external view returns (bool) {
        return _checkAllSubmittedInternal(DEFAULT_EXPERIMENT_ID, round);
    }

    function _checkAllSubmittedInternal(
        string memory experimentId,
        uint256 round
    ) internal view returns (bool) {
        if (round == 0 || _registeredClientIds.length == 0) {
            return false;
        }
        RoundStatus storage roundState = _rounds[_roundKey(experimentId, round)];
        if (!roundState.initialized || roundState.expectedCount == 0) {
            return false;
        }
        return roundState.submissionCount >= roundState.expectedCount;
    }

    /**
     * @notice Record the aggregation event after all expected clients have submitted.
     */
    function triggerAggregation(
        string calldata experimentId,
        uint256 round
    ) external onlyOwner {
        _triggerAggregationInternal(experimentId, round);
    }

    /**
     * @notice Convenience overload using the default experiment ID.
     */
    function triggerAggregation(uint256 round) external onlyOwner {
        _triggerAggregationInternal(DEFAULT_EXPERIMENT_ID, round);
    }

    function _triggerAggregationInternal(string memory experimentId, uint256 round) internal {
        if (round == 0) {
            revert InvalidRoundNumber(round);
        }
        if (_registeredClientIds.length == 0) {
            revert NoHospitalsRegistered();
        }

        bytes32 rKey = _roundKey(experimentId, round);
        RoundStatus storage roundState = _rounds[rKey];

        if (roundState.aggregated) {
            revert DuplicateAggregation(experimentId, round);
        }

        uint256 expected = roundState.initialized
            ? roundState.expectedCount
            : _resolveExpectedClients(experimentId);

        if (!roundState.initialized || roundState.submissionCount < expected) {
            revert IncompleteRoundSubmissions(
                experimentId,
                round,
                roundState.submissionCount,
                expected
            );
        }

        roundState.aggregated = true;
        roundState.aggregatedAt = block.timestamp;

        emit AggregationTriggered(
            experimentId,
            round,
            roundState.submissionCount,
            block.timestamp
        );
    }

    // =========================================================================
    // 4. Global Model Hash Storage
    // =========================================================================

    /**
     * @notice Store the global model checkpoint hash for a completed round (or round 0 initialization).
     */
    function storeGlobalModel(
        string calldata experimentId,
        uint256 round,
        string calldata taskType,
        bytes32 globalModelHash
    ) external onlyOwner {
        _storeGlobalModelInternal(experimentId, round, taskType, globalModelHash);
    }

    /**
     * @notice Convenience overload using the default experiment ID.
     */
    function storeGlobalModel(
        uint256 round,
        bytes32 globalModelHash
    ) external onlyOwner {
        _storeGlobalModelInternal(DEFAULT_EXPERIMENT_ID, round, "unspecified", globalModelHash);
    }

    function _storeGlobalModelInternal(
        string memory experimentId,
        uint256 round,
        string memory taskType,
        bytes32 globalModelHash
    ) internal {
        if (bytes(experimentId).length == 0) {
            revert EmptyIdentifier("experimentId");
        }
        if (globalModelHash == bytes32(0)) {
            revert InvalidHash();
        }

        bytes32 rKey = _roundKey(experimentId, round);
        RoundStatus storage roundState = _rounds[rKey];

        if (roundState.globalModelStored) {
            revert GlobalModelAlreadyStored(experimentId, round);
        }

        if (round > 0 && !roundState.aggregated) {
            revert AggregationNotTriggered(experimentId, round);
        }

        if (!roundState.initialized) {
            roundState.initialized = true;
            roundState.taskType = taskType;
        } else if (bytes(taskType).length > 0 && keccak256(bytes(taskType)) != keccak256(bytes("unspecified"))) {
            roundState.taskType = taskType;
        }

        roundState.globalModelHash = globalModelHash;
        roundState.globalModelStored = true;
        roundState.globalStoredAt = block.timestamp;

        emit GlobalModelStored(
            experimentId,
            round,
            roundState.taskType,
            globalModelHash,
            block.timestamp
        );
    }

    // =========================================================================
    // 5. Bounded Audit Queries
    // =========================================================================

    function getRegisteredHospitalCount() external view returns (uint256) {
        return _registeredClientIds.length;
    }

    function getRegisteredClientIds(
        uint256 offset,
        uint256 limit
    ) external view returns (string[] memory slice) {
        uint256 total = _registeredClientIds.length;
        if (offset >= total || limit == 0) {
            return new string[](0);
        }
        uint256 end = offset + limit;
        if (end > total) {
            end = total;
        }
        uint256 count = end - offset;
        slice = new string[](count);
        for (uint256 i = 0; i < count; i++) {
            slice[i] = _registeredClientIds[offset + i];
        }
    }

    function getHospital(
        string calldata clientId
    )
        external
        view
        returns (
            string memory id,
            string memory name,
            address wallet,
            bool registered,
            uint256 registeredAt
        )
    {
        Hospital storage h = _hospitalsByIdHash[_hashString(clientId)];
        return (h.clientId, h.name, h.wallet, h.registered, h.registeredAt);
    }

    function getHospitalByWallet(
        address wallet
    )
        external
        view
        returns (
            string memory id,
            string memory name,
            address hospitalWallet,
            bool registered,
            uint256 registeredAt
        )
    {
        bytes32 idHash = _clientIdHashByWallet[wallet];
        Hospital storage h = _hospitalsByIdHash[idHash];
        return (h.clientId, h.name, h.wallet, h.registered, h.registeredAt);
    }

    function getClientSubmission(
        string calldata experimentId,
        uint256 round,
        string calldata clientId
    )
        external
        view
        returns (
            bytes32 modelHash,
            string memory taskType,
            address submitter,
            uint256 timestamp,
            bool exists
        )
    {
        ClientSubmission storage sub = _submissions[_submissionKey(experimentId, round, clientId)];
        return (sub.modelHash, sub.taskType, sub.submitter, sub.timestamp, sub.exists);
    }

    function getRoundStatus(
        string calldata experimentId,
        uint256 round
    )
        external
        view
        returns (
            uint256 submissionCount,
            uint256 expectedCount,
            bool allSubmitted,
            bool aggregated,
            bool globalModelStored,
            bytes32 globalModelHash,
            uint256 aggregatedAt,
            uint256 globalStoredAt
        )
    {
        RoundStatus storage r = _rounds[_roundKey(experimentId, round)];
        bool complete = _checkAllSubmittedInternal(experimentId, round);
        return (
            r.submissionCount,
            r.expectedCount,
            complete,
            r.aggregated,
            r.globalModelStored,
            r.globalModelHash,
            r.aggregatedAt,
            r.globalStoredAt
        );
    }

    function getGlobalModelHash(
        string calldata experimentId,
        uint256 round
    ) external view returns (bytes32) {
        return _rounds[_roundKey(experimentId, round)].globalModelHash;
    }

    // =========================================================================
    // Internal Helpers
    // =========================================================================

    function _resolveExpectedClients(string memory experimentId) internal view returns (uint256) {
        uint256 totalRegistered = _registeredClientIds.length;
        if (totalRegistered == 0) {
            revert NoHospitalsRegistered();
        }
        uint256 configured = _experimentExpectedClients[_hashString(experimentId)];
        return configured > 0 ? configured : totalRegistered;
    }

    function _hashString(string memory value) internal pure returns (bytes32) {
        return keccak256(bytes(value));
    }

    function _roundKey(
        string memory experimentId,
        uint256 round
    ) internal pure returns (bytes32) {
        return keccak256(abi.encode(experimentId, round));
    }

    function _submissionKey(
        string memory experimentId,
        uint256 round,
        string memory clientId
    ) internal pure returns (bytes32) {
        return keccak256(abi.encode(experimentId, round, clientId));
    }
}
