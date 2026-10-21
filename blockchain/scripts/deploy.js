import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import hre from "hardhat";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function upsertEnvKey(envPath, key, value) {
  let content = "";
  if (fs.existsSync(envPath)) {
    content = fs.readFileSync(envPath, "utf8");
  }
  const lines = content.split(/\r?\n/);
  const regex = new RegExp(`^\\s*${key}\\s*=`);
  let updated = false;

  const newLines = lines.map((line) => {
    if (regex.test(line)) {
      updated = true;
      return `${key}=${value}`;
    }
    return line;
  });

  if (!updated) {
    if (newLines.length > 0 && newLines[newLines.length - 1] !== "") {
      newLines.push(`${key}=${value}`);
    } else if (newLines.length > 0) {
      newLines[newLines.length - 1] = `${key}=${value}`;
      newLines.push("");
    } else {
      newLines.push(`${key}=${value}`, "");
    }
  }

  fs.writeFileSync(envPath, newLines.join("\n"), "utf8");
}

async function main() {
  const artifactPath = path.join(
    __dirname,
    "..",
    "artifacts",
    "contracts",
    "FederatedLearning.sol",
    "FederatedLearning.json"
  );

  if (!fs.existsSync(artifactPath)) {
    throw new Error(
      `Compiled artifact not found at ${artifactPath}. Run 'npx hardhat compile' first.`
    );
  }

  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  const conn = await hre.network.connect();

  try {
    const chainIdHex = await conn.provider.request({ method: "eth_chainId" });
    const chainId = parseInt(chainIdHex, 16);
    const accounts = await conn.provider.request({ method: "eth_accounts" });

    if (!accounts || accounts.length === 0) {
      throw new Error(
        `No unlocked accounts returned on network '${conn.networkName}' (chainId=${chainId}).`
      );
    }

    const deployer = accounts[0];
    const txHash = await conn.provider.request({
      method: "eth_sendTransaction",
      params: [
        {
          from: deployer,
          data: artifact.bytecode,
          gas: "0x300000",
        },
      ],
    });

    let receipt = null;
    for (let attempt = 0; attempt < 30; attempt++) {
      receipt = await conn.provider.request({
        method: "eth_getTransactionReceipt",
        params: [txHash],
      });
      if (receipt) break;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }

    if (!receipt || receipt.status !== "0x1" || !receipt.contractAddress) {
      throw new Error(`Deployment transaction ${txHash} failed or did not return a contract address.`);
    }

    const contractAddress = receipt.contractAddress;
    const deployedCode = await conn.provider.request({
      method: "eth_getCode",
      params: [contractAddress, "latest"],
    });

    if (!deployedCode || deployedCode === "0x" || deployedCode.length <= 2) {
      throw new Error(`No contract bytecode found at deployed address ${contractAddress}.`);
    }

    const summary = {
      network: conn.networkName,
      chainId,
      deployer,
      transactionHash: txHash,
      contractAddress,
      bytecodeLengthBytes: (deployedCode.length - 2) / 2,
    };

    console.log("FederatedLearning deployment verified:");
    console.log(JSON.stringify(summary, null, 2));

    const shouldWriteEnv =
      process.env.WRITE_CONTRACT_ADDRESS_TO_ENV === "true" ||
      process.argv.includes("--write-env");

    if (shouldWriteEnv) {
      const envPath = path.resolve(__dirname, "..", "..", ".env");
      upsertEnvKey(envPath, "CONTRACT_ADDRESS", contractAddress);
      console.log(`Updated CONTRACT_ADDRESS in ${envPath}`);
    } else {
      console.log(
        "Skipped writing to .env (set WRITE_CONTRACT_ADDRESS_TO_ENV=true to persist CONTRACT_ADDRESS)."
      );
    }
  } finally {
    await conn.close();
  }
}

main().catch((err) => {
  console.error("Deployment failed:", err.message || err);
  process.exitCode = 1;
});
