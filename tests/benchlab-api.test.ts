import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createBenchLabApiServer } from "../src/api/benchlab-server";

const serversToClose: ReturnType<typeof createBenchLabApiServer>[] = [];

afterEach(async () => {
  await Promise.all(
    serversToClose.splice(0, serversToClose.length).map((server) => {
      return new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    })
  );
});

async function startServer(
  options: Parameters<typeof createBenchLabApiServer>[0]
) {
  const server = createBenchLabApiServer(options);
  serversToClose.push(server);
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address() as AddressInfo;
  return `http://127.0.0.1:${address.port}`;
}

function createBenchLabFixtureRoot() {
  const repoRoot = mkdtempSync(join(tmpdir(), "benchlab-fixture-"));
  const matrixRoot = join(repoRoot, "experiments", "prompt-bfcl-ralph-matrix");
  mkdirSync(matrixRoot, { recursive: true });

  writeFileSync(
    join(matrixRoot, "models.ollama.local.json"),
    JSON.stringify({ models: [] }, null, 2)
  );
  writeFileSync(
    join(matrixRoot, "models.zero-cost.local.json"),
    JSON.stringify({ models: [] }, null, 2)
  );
  writeFileSync(
    join(matrixRoot, "run_prompt_bfcl_ralph_matrix.py"),
    "print('stub')\n"
  );

  return { matrixRoot, repoRoot };
}

describe("benchlab api", () => {
  it("lists config files and renders the demo", async () => {
    const fixture = createBenchLabFixtureRoot();
    const baseUrl = await startServer({
      benchmarkRoot: "/tmp/bfcl",
      pythonExecutable: "/usr/bin/python3",
      repoRoot: fixture.repoRoot,
    });

    const htmlResponse = await fetch(`${baseUrl}/benchlab`);
    const html = await htmlResponse.text();
    expect(htmlResponse.status).toBe(200);
    expect(html).toContain("BenchLab");

    const configResponse = await fetch(`${baseUrl}/v1/benchlab/configs`);
    const payload = await configResponse.json();
    expect(configResponse.status).toBe(200);
    expect(payload.configs).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "models.ollama.local.json" }),
        expect.objectContaining({ name: "models.zero-cost.local.json" }),
      ])
    );
  });

  it("creates a job and exposes it through jobs and runs endpoints", async () => {
    const fixture = createBenchLabFixtureRoot();
    const baseUrl = await startServer({
      benchmarkRoot: "/tmp/bfcl",
      jobLauncher: ({ stdoutPath, stderrPath }) => {
        writeFileSync(stdoutPath, "job started\n");
        writeFileSync(stderrPath, "");
        const normalizedRuntimeRoot = dirname(stdoutPath);
        writeFileSync(
          join(normalizedRuntimeRoot, "matrix_summary.json"),
          JSON.stringify(
            {
              categories: ["simple_python"],
              cases_per_category: 5,
              preflight_only: false,
              counts: {
                completed: 1,
                improved: 1,
                flat: 0,
                regressed: 0,
                failed: 0,
                preflight_ok: 0,
                unknown: 0,
              },
              models_file: join(fixture.matrixRoot, "models.ollama.local.json"),
            },
            null,
            2
          )
        );
        writeFileSync(
          join(normalizedRuntimeRoot, "matrix_report.md"),
          "# Matrix Report\n\n- improved\n"
        );
        return {
          pid: 4321,
          kill: () => {
            // no-op test launcher
          },
          completion: Promise.resolve(0),
        };
      },
      pythonExecutable: "/usr/bin/python3",
      repoRoot: fixture.repoRoot,
    });

    const createResponse = await fetch(`${baseUrl}/v1/benchlab/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "benchmark",
        modelsFile: "models.ollama.local.json",
        runtimeName: "runtime-test-suite",
        casesPerCategory: 5,
      }),
    });
    const createPayload = await createResponse.json();
    expect(createResponse.status).toBe(202);
    expect(createPayload.job.status).toBe("running");

    await new Promise((resolve) => setTimeout(resolve, 20));

    const jobsResponse = await fetch(`${baseUrl}/v1/benchlab/jobs`);
    const jobsPayload = await jobsResponse.json();
    expect(jobsResponse.status).toBe(200);
    expect(jobsPayload.jobs[0].status).toBe("completed");

    const runsResponse = await fetch(`${baseUrl}/v1/benchlab/runs`);
    const runsPayload = await runsResponse.json();
    expect(runsResponse.status).toBe(200);
    expect(runsPayload.runs[0].name).toBe("runtime-test-suite");
    expect(runsPayload.runs[0].primaryOutcome).toBe("improved");

    const runDetailResponse = await fetch(
      `${baseUrl}/v1/benchlab/runs/runtime-test-suite`
    );
    const runDetailPayload = await runDetailResponse.json();
    expect(runDetailResponse.status).toBe(200);
    expect(runDetailPayload.run.reportMarkdown).toContain("Matrix Report");

    const logsResponse = await fetch(
      `${baseUrl}/v1/benchlab/jobs/${createPayload.job.id}/logs`
    );
    const logsPayload = await logsResponse.json();
    expect(logsResponse.status).toBe(200);
    expect(logsPayload.logs.stdout.text).toContain("job started");
  });

  it("cancels a running job", async () => {
    const fixture = createBenchLabFixtureRoot();
    let wasKilled = false;
    let resolveCompletion: ((value: number) => void) | null = null;
    const baseUrl = await startServer({
      benchmarkRoot: "/tmp/bfcl",
      jobLauncher: ({ stdoutPath, stderrPath }) => {
        writeFileSync(stdoutPath, "job started\n");
        writeFileSync(stderrPath, "");
        return {
          pid: 9876,
          kill: () => {
            wasKilled = true;
            resolveCompletion?.(1);
          },
          completion: new Promise<number>((resolve) => {
            resolveCompletion = resolve;
          }),
        };
      },
      pythonExecutable: "/usr/bin/python3",
      repoRoot: fixture.repoRoot,
    });

    const createResponse = await fetch(`${baseUrl}/v1/benchlab/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "benchmark",
        modelsFile: "models.ollama.local.json",
        runtimeName: "runtime-cancel-me",
      }),
    });
    const createPayload = await createResponse.json();
    const jobId = createPayload.job.id;

    const cancelResponse = await fetch(
      `${baseUrl}/v1/benchlab/jobs/${jobId}/cancel`,
      {
        method: "POST",
      }
    );
    const cancelPayload = await cancelResponse.json();
    expect(cancelResponse.status).toBe(200);
    expect(cancelPayload.job.status).toBe("cancelled");
    expect(wasKilled).toBe(true);

    await new Promise((resolve) => setTimeout(resolve, 20));

    const jobsResponse = await fetch(`${baseUrl}/v1/benchlab/jobs`);
    const jobsPayload = await jobsResponse.json();
    expect(jobsPayload.jobs[0].status).toBe("cancelled");
  });
});
