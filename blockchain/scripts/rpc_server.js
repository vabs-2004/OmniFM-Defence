import http from "node:http";
import hre from "hardhat";

const port = Number(process.env.RPC_PORT || 8545);
const host = process.env.RPC_HOST || "127.0.0.1";

async function startRpcServer() {
  const conn = await hre.network.connect();

  const server = http.createServer((req, res) => {
    if (req.method !== "POST") {
      res.writeHead(405, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Method Not Allowed" }));
      return;
    }

    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });

    req.on("end", async () => {
      let payload;
      try {
        payload = JSON.parse(body);
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "Parse error" },
          })
        );
        return;
      }

      const handleSingle = async (item) => {
        try {
          const result = await conn.provider.request({
            method: item.method,
            params: item.params || [],
          });
          return {
            jsonrpc: "2.0",
            id: item.id ?? null,
            result,
          };
        } catch (err) {
          return {
            jsonrpc: "2.0",
            id: item.id ?? null,
            error: {
              code: err.code || -32000,
              message: err.message || String(err),
              data: err.data,
            },
          };
        }
      };

      const responseData = Array.isArray(payload)
        ? await Promise.all(payload.map(handleSingle))
        : await handleSingle(payload);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(responseData));
    });
  });

  server.listen(port, host, () => {
    console.log(`READY http://${host}:${port} chainId=1337`);
  });

  const shutdown = async () => {
    server.close();
    await conn.close();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

startRpcServer().catch((err) => {
  console.error("Failed to start RPC server:", err);
  process.exit(1);
});
