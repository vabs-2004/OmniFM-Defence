export default {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    default: {
      type: "edr-simulated",
      chainId: 1337,
    },
    ganache: {
      type: "http",
      url: process.env.GANACHE_URL || "http://127.0.0.1:8545",
      chainId: Number(process.env.GANACHE_CHAIN_ID || 1337),
    },
  },
};
