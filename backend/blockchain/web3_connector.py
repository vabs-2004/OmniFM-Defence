"""Web3 connector for the FederatedLearning smart contract on Ganache / Hardhat (Task 1.5).

Handles hospital registration, client model-update hash submissions, round quorum
checks, aggregation recording, global checkpoint hash storage, transaction receipt
verification, and audit event queries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eth_account import Account
from pydantic import SecretStr
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError, Web3Exception

from backend.config import Settings, settings
from backend.utils.hashing import normalize_bytes32_hex, to_bytes32
from backend.utils.logging import setup_logger

logger = setup_logger("omnifm.blockchain")

DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "blockchain"
    / "artifacts"
    / "contracts"
    / "FederatedLearning.sol"
    / "FederatedLearning.json"
)


class BlockchainError(RuntimeError):
    """Base error for blockchain connector failures."""


class BlockchainConnectionError(BlockchainError):
    """Raised when the RPC node or contract is unreachable or misconfigured."""


class BlockchainTransactionError(BlockchainError):
    """Raised when a transaction fails to sign, broadcast, or reverts on-chain."""


@dataclass(frozen=True)
class TxReceiptRecord:
    """Verified on-chain transaction receipt summary."""

    action: str
    tx_hash: str
    block_number: int
    gas_used: int
    status: str
    sender: str
    contract_address: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_contract_artifact(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load compiled ABI and bytecode for FederatedLearning.sol."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Compiled FederatedLearning artifact not found at {path}. "
            "Run 'npx hardhat compile' in ./blockchain first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_secret(key: str | SecretStr | None) -> str:
    if key is None:
        return ""
    if isinstance(key, SecretStr):
        return key.get_secret_value().strip()
    return str(key).strip()


class Web3Connector:
    """Client for interacting with `FederatedLearning.sol` over HTTP JSON-RPC."""

    def __init__(
        self,
        cfg: Settings | None = None,
        w3: Web3 | None = None,
        contract_address: str | None = None,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        self.cfg = cfg or settings
        self.artifact_path = Path(artifact_path)
        self.w3 = w3 or Web3(
            Web3.HTTPProvider(self.cfg.ganache_url, request_kwargs={"timeout": 15})
        )
        self.contract_address = (contract_address or self.cfg.contract_address or "").strip()
        self._artifact: dict[str, Any] | None = None
        self._contract: Contract | None = None

        self._client_keys: dict[str, SecretStr] = {
            "client_1": self.cfg.client1_private_key,
            "client_2": self.cfg.client2_private_key,
            "client_3": self.cfg.client3_private_key,
        }

    @property
    def artifact(self) -> dict[str, Any]:
        if self._artifact is None:
            self._artifact = load_contract_artifact(self.artifact_path)
        return self._artifact

    def is_connected(self) -> bool:
        """Return True if the JSON-RPC endpoint is reachable."""
        try:
            return bool(self.w3.is_connected())
        except Exception:
            return False

    def is_contract_deployed(self) -> bool:
        """Return True if RPC is connected, chain ID matches, and contract bytecode exists at CONTRACT_ADDRESS."""
        try:
            info = self.verify_chain_and_contract()
            return bool(info.get("connected") and int(info.get("bytecode_len", 0)) > 0)
        except Exception:
            return False

    def verify_chain_and_contract(self) -> dict[str, Any]:
        """Verify RPC connectivity, chain ID, and deployed contract bytecode."""
        if not self.is_connected():
            raise BlockchainConnectionError(
                f"Cannot connect to blockchain RPC at {self.cfg.ganache_url}."
            )

        actual_chain_id = int(self.w3.eth.chain_id)
        if actual_chain_id != self.cfg.ganache_chain_id:
            raise BlockchainConnectionError(
                f"Chain ID mismatch: expected {self.cfg.ganache_chain_id}, got {actual_chain_id}."
            )

        if not self.contract_address:
            raise BlockchainConnectionError("CONTRACT_ADDRESS is not configured.")

        checksum_addr = Web3.to_checksum_address(self.contract_address)
        code = self.w3.eth.get_code(checksum_addr)
        if not code or len(code) == 0:
            raise BlockchainConnectionError(
                f"No contract bytecode found at address {checksum_addr} on chain {actual_chain_id}."
            )

        return {
            "connected": True,
            "chain_id": actual_chain_id,
            "contract_address": checksum_addr,
            "bytecode_len": len(code),
        }

    @property
    def contract(self) -> Contract:
        """Return bound Web3 contract instance after verifying bytecode exists."""
        if self._contract is None:
            info = self.verify_chain_and_contract()
            self._contract = self.w3.eth.contract(
                address=info["contract_address"],
                abi=self.artifact["abi"],
            )
        return self._contract

    def get_contract_owner(self) -> str:
        """Query the immutable `owner` address stored in the deployed FederatedLearning contract."""
        raw_owner = self.contract.functions.owner().call()
        return Web3.to_checksum_address(raw_owner)

    def get_configured_server_address(self) -> str | None:
        """Return the checksum address derived from `SERVER_PRIVATE_KEY`, or None if unset."""
        raw = _unwrap_secret(self.cfg.server_private_key)
        if not raw:
            return None
        return self.get_wallet_address(raw)

    def get_configured_client_address(self, client_id: str) -> str | None:
        """Return the checksum address derived from the client's configured private key, or None if unset."""
        raw = _unwrap_secret(self._client_keys.get(client_id))
        if not raw:
            return None
        return self.get_wallet_address(raw)

    def set_client_private_key(self, client_id: str, private_key: str | SecretStr) -> None:
        """Register or override a development private key for a logical client in memory."""
        raw = _unwrap_secret(private_key)
        self._client_keys[client_id] = SecretStr(raw)

    def get_wallet_address(self, private_key: str | SecretStr) -> str:
        """Derive the checksum Ethereum address from a private key."""
        raw = _unwrap_secret(private_key)
        if not raw:
            raise ValueError("Private key is empty; cannot derive wallet address.")
        pk = raw if raw.startswith("0x") else f"0x{raw}"
        acct = Account.from_key(pk)
        return Web3.to_checksum_address(acct.address)

    def deploy_contract(
        self,
        deployer_private_key: str | SecretStr | None = None,
    ) -> tuple[str, TxReceiptRecord]:
        """Deploy a fresh instance of FederatedLearning.sol and bind this connector to it."""
        if not self.is_connected():
            raise BlockchainConnectionError(
                f"Cannot connect to blockchain RPC at {self.cfg.ganache_url}."
            )

        key = _unwrap_secret(deployer_private_key or self.cfg.server_private_key)
        if not key:
            raise BlockchainTransactionError(
                "Missing SERVER_PRIVATE_KEY for 'deploy_contract'."
            )
        factory = self.w3.eth.contract(
            abi=self.artifact["abi"],
            bytecode=self.artifact["bytecode"],
        )
        tx_call = factory.constructor()
        receipt = self._send_transaction(
            tx_call=tx_call,
            private_key=key,
            action="deploy_contract",
        )
        self.contract_address = receipt.contract_address
        self._contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.contract_address),
            abi=self.artifact["abi"],
        )
        return self.contract_address, receipt

    def _send_transaction(
        self,
        tx_call: Any,
        private_key: str | SecretStr | None,
        action: str,
        sender_address: str | None = None,
    ) -> TxReceiptRecord:
        """Sign (or use explicit sender_address), broadcast, and verify a contract transaction.

        Never silently falls back to `w3.eth.accounts[0]` when both `private_key` and
        `sender_address` are missing.
        """
        if not self.is_connected():
            raise BlockchainConnectionError(
                f"Blockchain RPC unreachable at {self.cfg.ganache_url} during '{action}'."
            )

        raw_key = _unwrap_secret(private_key)
        if not raw_key and not sender_address:
            raise BlockchainTransactionError(
                f"Missing private key for transaction '{action}'; refusing to silently sign with an unintended account."
            )

        try:
            if raw_key:
                pk = raw_key if raw_key.startswith("0x") else f"0x{raw_key}"
                account = Account.from_key(pk)
                sender = Web3.to_checksum_address(account.address)
                nonce = self.w3.eth.get_transaction_count(sender, "pending")
                chain_id = int(self.w3.eth.chain_id)
                gas_price = int(self.w3.eth.gas_price)

                estimated_gas = int(tx_call.estimate_gas({"from": sender}))
                gas_limit = max(int(estimated_gas * 1.25), estimated_gas + 25_000)

                tx_dict = tx_call.build_transaction(
                    {
                        "from": sender,
                        "nonce": nonce,
                        "chainId": chain_id,
                        "gas": gas_limit,
                        "gasPrice": gas_price,
                    }
                )
                signed = self.w3.eth.account.sign_transaction(tx_dict, private_key=pk)
                tx_hash_bytes = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            else:
                sender = Web3.to_checksum_address(str(sender_address))
                tx_hash_bytes = tx_call.transact({"from": sender})

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=30)
        except (ContractLogicError, Web3Exception, ValueError) as exc:
            raise BlockchainTransactionError(
                f"Blockchain transaction '{action}' failed: {exc}"
            ) from exc

        tx_hash_hex = "0x" + receipt["transactionHash"].hex().lstrip("0x")
        if len(tx_hash_hex) != 66:
            tx_hash_hex = "0x" + receipt["transactionHash"].hex()

        if int(receipt.get("status", 0)) != 1:
            raise BlockchainTransactionError(
                f"Transaction '{action}' ({tx_hash_hex}) reverted on-chain (status={receipt.get('status')})."
            )

        contract_addr = (
            receipt.get("contractAddress")
            or receipt.get("to")
            or self.contract_address
            or ""
        )
        return TxReceiptRecord(
            action=action,
            tx_hash=tx_hash_hex,
            block_number=int(receipt["blockNumber"]),
            gas_used=int(receipt["gasUsed"]),
            status="CONFIRMED",
            sender=sender,
            contract_address=str(contract_addr),
        )

    # =========================================================================
    # Contract Operations
    # =========================================================================

    def register_hospital(
        self,
        client_id: str,
        name: str,
        wallet_address: str,
        server_private_key: str | SecretStr | None = None,
    ) -> TxReceiptRecord:
        """Register a hospital client on-chain using the server/owner account."""
        key = _unwrap_secret(server_private_key or self.cfg.server_private_key)
        if not key:
            raise BlockchainTransactionError(
                "Missing SERVER_PRIVATE_KEY required for owner-only 'register_hospital'."
            )
        checksum_wallet = Web3.to_checksum_address(wallet_address)
        fn = self.contract.get_function_by_signature(
            "registerHospital(string,string,address)"
        )(client_id, name, checksum_wallet)
        return self._send_transaction(
            tx_call=fn,
            private_key=key,
            action="register_hospital",
        )

    def register_client(
        self,
        client_address: str,
        client_id: str,
        hospital_name: str,
        modality: str = "multi-modality",
        server_private_key: str | SecretStr | None = None,
    ) -> TxReceiptRecord:
        """Compatibility wrapper around `register_hospital`."""
        _ = modality
        return self.register_hospital(
            client_id=client_id,
            name=hospital_name,
            wallet_address=client_address,
            server_private_key=server_private_key,
        )

    def ensure_hospital_registered(
        self,
        client_id: str,
        name: str | None = None,
        wallet_address: str | None = None,
        hospital_name: str | None = None,
        modality: str = "multi-modality",
        server_private_key: str | SecretStr | None = None,
    ) -> dict[str, Any]:
        """Ensure `client_id` is registered on-chain with its expected wallet address."""
        _ = modality
        effective_name = (name or hospital_name or f"Hospital_{client_id}").strip()
        resolved_wallet = wallet_address or self.get_configured_client_address(client_id)
        if not resolved_wallet:
            raise BlockchainTransactionError(
                f"Cannot register '{client_id}': no wallet_address or client private key configured."
            )
        checksum_wallet = Web3.to_checksum_address(resolved_wallet)
        existing = self.get_hospital(client_id)
        if existing["registered"]:
            on_chain_wallet = Web3.to_checksum_address(existing["wallet"])
            if on_chain_wallet != checksum_wallet:
                raise BlockchainTransactionError(
                    f"On-chain wallet mismatch for '{client_id}': contract has {on_chain_wallet}, "
                    f"configured wallet is {checksum_wallet}."
                )
            return {
                "already_registered": True,
                "client_id": client_id,
                "wallet": on_chain_wallet,
                "tx_hash": None,
                "block_number": None,
                "sender": None,
            }

        receipt = self.register_hospital(
            client_id=client_id,
            name=effective_name,
            wallet_address=checksum_wallet,
            server_private_key=server_private_key,
        )
        return {
            "already_registered": False,
            "client_id": client_id,
            "wallet": checksum_wallet,
            "tx_hash": receipt.tx_hash,
            "block_number": receipt.block_number,
            "sender": receipt.sender,
        }

    def submit_model_update(
        self,
        experiment_id: str,
        round_num: int,
        task_type: str,
        model_hash: str | bytes,
        client_id: str,
        client_private_key: str | SecretStr | None = None,
        client_wallet_address: str | None = None,
    ) -> TxReceiptRecord:
        """Submit a client's model-update hash signed by that client's configured wallet."""
        hash_bytes32 = to_bytes32(model_hash)
        key = _unwrap_secret(client_private_key or self._client_keys.get(client_id))
        if not key and not client_wallet_address:
            env_var_map = {
                "client_1": "CLIENT1_PRIVATE_KEY",
                "client_2": "CLIENT2_PRIVATE_KEY",
                "client_3": "CLIENT3_PRIVATE_KEY",
            }
            var_name = env_var_map.get(client_id, f"{client_id.upper()}_PRIVATE_KEY")
            raise BlockchainTransactionError(
                f"Missing required private key variable '{var_name}' for client '{client_id}' during 'submit_model_update'."
            )
        fn = self.contract.get_function_by_signature(
            "submitModelUpdate(string,uint256,string,bytes32,string)"
        )(experiment_id, int(round_num), task_type, hash_bytes32, client_id)
        return self._send_transaction(
            tx_call=fn,
            private_key=key,
            action="submit_model_update",
            sender_address=client_wallet_address,
        )

    def set_experiment_expected_clients(
        self,
        experiment_id: str,
        expected_count: int,
        server_private_key: str | SecretStr | None = None,
    ) -> TxReceiptRecord:
        """Configure the expected client quorum for `experiment_id` on-chain (owner-only)."""
        key = _unwrap_secret(server_private_key or self.cfg.server_private_key)
        if not key:
            raise BlockchainTransactionError(
                "Missing SERVER_PRIVATE_KEY required for owner-only 'set_experiment_expected_clients'."
            )
        fn = self.contract.get_function_by_signature(
            "setExperimentExpectedClients(string,uint256)"
        )(experiment_id, int(expected_count))
        return self._send_transaction(
            tx_call=fn,
            private_key=key,
            action="set_experiment_expected_clients",
        )

    def check_all_submitted(self, experiment_id: str, round_num: int) -> bool:
        """Return True if all expected hospitals have submitted for `(experiment_id, round_num)`."""
        fn = self.contract.get_function_by_signature("checkAllSubmitted(string,uint256)")
        return bool(fn(experiment_id, int(round_num)).call())

    def trigger_aggregation(
        self,
        experiment_id: str,
        round_num: int,
        server_private_key: str | SecretStr | None = None,
    ) -> TxReceiptRecord:
        """Record the round aggregation event on-chain after submission quorum is met."""
        key = _unwrap_secret(server_private_key or self.cfg.server_private_key)
        if not key:
            raise BlockchainTransactionError(
                "Missing SERVER_PRIVATE_KEY required for owner-only 'trigger_aggregation'."
            )
        fn = self.contract.get_function_by_signature("triggerAggregation(string,uint256)")(
            experiment_id, int(round_num)
        )
        return self._send_transaction(
            tx_call=fn,
            private_key=key,
            action="trigger_aggregation",
        )

    def store_global_model(
        self,
        experiment_id: str,
        round_num: int,
        task_type: str,
        global_model_hash: str | bytes,
        server_private_key: str | SecretStr | None = None,
    ) -> TxReceiptRecord:
        """Store a global model checkpoint SHA-256 hash on-chain."""
        key = _unwrap_secret(server_private_key or self.cfg.server_private_key)
        if not key:
            raise BlockchainTransactionError(
                "Missing SERVER_PRIVATE_KEY required for owner-only 'store_global_model'."
            )
        hash_bytes32 = to_bytes32(global_model_hash)
        fn = self.contract.get_function_by_signature(
            "storeGlobalModel(string,uint256,string,bytes32)"
        )(experiment_id, int(round_num), task_type, hash_bytes32)
        return self._send_transaction(
            tx_call=fn,
            private_key=key,
            action="store_global_model",
        )

    # =========================================================================
    # Read / Query Operations
    # =========================================================================

    def get_hospital(self, client_id: str) -> dict[str, Any]:
        """Fetch on-chain registration record for `client_id`."""
        res = self.contract.functions.getHospital(client_id).call()
        return {
            "client_id": res[0],
            "name": res[1],
            "wallet": res[2],
            "registered": bool(res[3]),
            "registered_at": int(res[4]),
        }

    def get_client_submission(
        self,
        experiment_id: str,
        round_num: int,
        client_id: str,
    ) -> dict[str, Any]:
        """Fetch on-chain client submission metadata for `(experiment_id, round_num, client_id)`."""
        res = self.contract.functions.getClientSubmission(
            experiment_id, int(round_num), client_id
        ).call()
        raw_hash: bytes = res[0]
        hex_hash = "0x" + raw_hash.hex()
        return {
            "model_hash": hex_hash,
            "task_type": res[1],
            "submitter": res[2],
            "timestamp": int(res[3]),
            "exists": bool(res[4]),
        }

    def get_round_status(self, experiment_id: str, round_num: int) -> dict[str, Any]:
        """Fetch on-chain round state for `(experiment_id, round_num)`."""
        res = self.contract.functions.getRoundStatus(experiment_id, int(round_num)).call()
        return {
            "submission_count": int(res[0]),
            "expected_count": int(res[1]),
            "all_submitted": bool(res[2]),
            "aggregated": bool(res[3]),
            "global_model_stored": bool(res[4]),
            "global_model_hash": "0x" + res[5].hex(),
            "aggregated_at": int(res[6]),
            "global_stored_at": int(res[7]),
        }

    def get_global_model_hash(self, experiment_id: str, round_num: int) -> str:
        """Fetch the recorded 0x-prefixed global model hash for `(experiment_id, round_num)`."""
        raw: bytes = self.contract.functions.getGlobalModelHash(
            experiment_id, int(round_num)
        ).call()
        return "0x" + raw.hex()

    def verify_model_hash(
        self,
        experiment_id: str,
        round_num: int,
        expected_hash: str | bytes,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify an expected SHA-256 hash against the on-chain contract record."""
        norm_expected = normalize_bytes32_hex(expected_hash)
        if client_id:
            sub = self.get_client_submission(experiment_id, int(round_num), client_id)
            exists = bool(sub["exists"])
            on_chain_hash = normalize_bytes32_hex(sub["model_hash"]) if exists else None
            return {
                "exists": exists,
                "on_chain_hash": on_chain_hash,
                "expected_hash": norm_expected,
                "match": bool(exists and on_chain_hash == norm_expected),
                "submitter": sub["submitter"] if exists else None,
                "timestamp": sub["timestamp"] if exists else None,
            }

        r_status = self.get_round_status(experiment_id, int(round_num))
        exists = bool(r_status["global_model_stored"])
        on_chain_hash = (
            normalize_bytes32_hex(r_status["global_model_hash"]) if exists else None
        )
        return {
            "exists": exists,
            "on_chain_hash": on_chain_hash,
            "expected_hash": norm_expected,
            "match": bool(exists and on_chain_hash == norm_expected),
            "timestamp": r_status["global_stored_at"] if exists else None,
        }

    def get_audit_events(
        self,
        experiment_id: str | None = None,
        round_num: int | None = None,
        client_id: str | None = None,
        from_block: int = 0,
        to_block: int | str = "latest",
    ) -> list[dict[str, Any]]:
        """Read and normalize audit events from the contract logs."""
        events: list[dict[str, Any]] = []

        # 1. HospitalRegistered
        hosp_logs = self.contract.events.HospitalRegistered().get_logs(
            from_block=from_block, to_block=to_block
        )
        for log in hosp_logs:
            args = log["args"]
            if client_id and args["clientId"] != client_id:
                continue
            events.append(
                {
                    "event_type": "HospitalRegistered",
                    "experiment_id": None,
                    "round": None,
                    "client_id": args["clientId"],
                    "name": args["name"],
                    "wallet": args["wallet"],
                    "task_type": None,
                    "model_hash": None,
                    "timestamp": int(args["timestamp"]),
                    "tx_hash": "0x" + log["transactionHash"].hex(),
                    "block_number": int(log["blockNumber"]),
                    "log_index": int(log["logIndex"]),
                }
            )

        # 2. ModelUpdateSubmitted
        sub_logs = self.contract.events.ModelUpdateSubmitted().get_logs(
            from_block=from_block, to_block=to_block
        )
        for log in sub_logs:
            args = log["args"]
            if experiment_id and args["experimentId"] != experiment_id:
                continue
            if round_num is not None and int(args["round"]) != int(round_num):
                continue
            if client_id and args["clientId"] != client_id:
                continue
            events.append(
                {
                    "event_type": "ModelUpdateSubmitted",
                    "experiment_id": args["experimentId"],
                    "round": int(args["round"]),
                    "client_id": args["clientId"],
                    "wallet": args["wallet"],
                    "task_type": args["taskType"],
                    "model_hash": normalize_bytes32_hex(args["modelHash"]),
                    "timestamp": int(args["timestamp"]),
                    "tx_hash": "0x" + log["transactionHash"].hex(),
                    "block_number": int(log["blockNumber"]),
                    "log_index": int(log["logIndex"]),
                }
            )

        # 3. AggregationTriggered
        agg_logs = self.contract.events.AggregationTriggered().get_logs(
            from_block=from_block, to_block=to_block
        )
        for log in agg_logs:
            args = log["args"]
            if experiment_id and args["experimentId"] != experiment_id:
                continue
            if round_num is not None and int(args["round"]) != int(round_num):
                continue
            if client_id:
                continue
            events.append(
                {
                    "event_type": "AggregationTriggered",
                    "experiment_id": args["experimentId"],
                    "round": int(args["round"]),
                    "client_id": None,
                    "wallet": None,
                    "task_type": None,
                    "submission_count": int(args["submissionCount"]),
                    "model_hash": None,
                    "timestamp": int(args["timestamp"]),
                    "tx_hash": "0x" + log["transactionHash"].hex(),
                    "block_number": int(log["blockNumber"]),
                    "log_index": int(log["logIndex"]),
                }
            )

        # 4. GlobalModelStored
        gm_logs = self.contract.events.GlobalModelStored().get_logs(
            from_block=from_block, to_block=to_block
        )
        for log in gm_logs:
            args = log["args"]
            if experiment_id and args["experimentId"] != experiment_id:
                continue
            if round_num is not None and int(args["round"]) != int(round_num):
                continue
            if client_id:
                continue
            events.append(
                {
                    "event_type": "GlobalModelStored",
                    "experiment_id": args["experimentId"],
                    "round": int(args["round"]),
                    "client_id": None,
                    "wallet": None,
                    "task_type": args["taskType"],
                    "model_hash": normalize_bytes32_hex(args["globalModelHash"]),
                    "timestamp": int(args["timestamp"]),
                    "tx_hash": "0x" + log["transactionHash"].hex(),
                    "block_number": int(log["blockNumber"]),
                    "log_index": int(log["logIndex"]),
                }
            )

        events.sort(key=lambda e: (e["block_number"], e["log_index"]))
        return events
