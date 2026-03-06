import { createBenchLabApiServer } from "../api/benchlab-server";

function readPort(): number {
  const value = Number.parseInt(process.env.PORT ?? "8090", 10);
  if (Number.isNaN(value) || value < 1 || value > 65_535) {
    return 8090;
  }
  return value;
}

const port = readPort();
const server = createBenchLabApiServer();

server.listen(port, "0.0.0.0", () => {
  console.info(`[benchlab-api] listening on 0.0.0.0:${port}`);
});

function shutdown(signal: NodeJS.Signals) {
  console.info(`[benchlab-api] received ${signal}, shutting down`);
  server.close((error) => {
    if (error) {
      console.error("[benchlab-api] shutdown error", error);
      process.exitCode = 1;
      return;
    }
    process.exitCode = 0;
  });
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
