import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { dirname, extname, join, relative, resolve, sep } from "node:path";
import { renderBenchLabDemoHtml } from "./benchlab-demo";

type Logger = Pick<Console, "error" | "info" | "warn">;

interface JsonObject {
  [key: string]: unknown;
}

type BenchLabJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

interface BenchLabJobRecord {
  command: string[];
  createdAt: string;
  endedAt: string | null;
  errorMessage: string | null;
  exitCode: number | null;
  id: string;
  kill: ((signal?: NodeJS.Signals) => void) | null;
  mode: "benchmark" | "preflight";
  modelsFileName: string;
  modelsFilePath: string;
  pid: number | null;
  runtimeName: string;
  runtimeRoot: string;
  startedAt: string | null;
  status: BenchLabJobStatus;
  stderrPath: string;
  stdoutPath: string;
}

interface BenchLabJobRequest {
  args: string[];
  command: string;
  cwd: string;
  env: NodeJS.ProcessEnv;
  stderrPath: string;
  stdoutPath: string;
}

interface BenchLabLaunchedJob {
  completion: Promise<number>;
  kill: (signal?: NodeJS.Signals) => void;
  pid: number | null;
}

type BenchLabJobLauncher = (request: BenchLabJobRequest) => BenchLabLaunchedJob;

export interface BenchLabApiServerOptions {
  benchmarkRoot?: string;
  jobLauncher?: BenchLabJobLauncher;
  logger?: Logger;
  pythonExecutable?: string;
  repoRoot?: string;
}

interface BenchLabConfigDescriptor {
  isRecommended: boolean;
  name: string;
  path: string;
}

interface BenchLabRuntimeSummary {
  casesPerCategory: number | null;
  categories: string[];
  counts: Record<string, number>;
  modelsFileName: string | null;
  name: string;
  preflightOnly: boolean | null;
  primaryOutcome: string;
  reportMarkdown: string | null;
  runtimeRoot: string;
  updatedAt: string | null;
}

const HTML_CONTENT_TYPE = "text/html; charset=utf-8";
const JSON_CONTENT_TYPE = "application/json; charset=utf-8";
const DEFAULT_BODY_LIMIT_BYTES = 1_048_576;
const DEFAULT_MATRIX_RUNNER_RELATIVE_PATH = join(
  "experiments",
  "prompt-bfcl-ralph-matrix",
  "run_prompt_bfcl_ralph_matrix.py"
);
const DEFAULT_MODELS_FILE_NAME = "models.ollama.local.json";
const DEFAULT_RUNTIME_PREFIX = "runtime-service";

class HttpError extends Error {
  readonly statusCode: number;

  constructor(statusCode: number, message: string) {
    super(message);
    this.name = "HttpError";
    this.statusCode = statusCode;
  }
}

function sendJson(
  response: ServerResponse,
  statusCode: number,
  payload: JsonObject
): void {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", JSON_CONTENT_TYPE);
  response.end(`${JSON.stringify(payload)}\n`);
}

function sendHtml(
  response: ServerResponse,
  statusCode: number,
  html: string
): void {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", HTML_CONTENT_TYPE);
  response.end(html);
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "unknown error";
}

function toHttpError(error: unknown): HttpError {
  if (error instanceof HttpError) {
    return error;
  }
  if (error instanceof SyntaxError) {
    return new HttpError(400, "invalid JSON body");
  }
  return new HttpError(500, toErrorMessage(error));
}

async function readJsonBody(
  request: IncomingMessage,
  maxBytes = DEFAULT_BODY_LIMIT_BYTES
): Promise<unknown> {
  const contentType = request.headers["content-type"];
  if (
    typeof contentType === "string" &&
    !contentType.includes("application/json")
  ) {
    throw new HttpError(415, "content-type must be application/json");
  }

  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    total += buffer.length;
    if (total > maxBytes) {
      throw new HttpError(413, "request body too large");
    }
    chunks.push(buffer);
  }

  if (chunks.length === 0) {
    return {};
  }

  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function parseRequestUrl(rawUrl: string | undefined): URL {
  return new URL(rawUrl ?? "/", "http://127.0.0.1");
}

function nowIso(): string {
  return new Date().toISOString();
}

function toPositiveInt(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const normalized = Math.trunc(value);
  return normalized > 0 ? normalized : fallback;
}

function toOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toOptionalStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter((item) => item.length > 0);
}

function sanitizeRuntimeName(value: string | null): string {
  const candidate = value ?? `${DEFAULT_RUNTIME_PREFIX}-${Date.now()}`;
  const sanitized = candidate
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!sanitized) {
    return `${DEFAULT_RUNTIME_PREFIX}-${Date.now()}`;
  }
  if (!sanitized.startsWith("runtime")) {
    return `runtime-${sanitized}`;
  }
  return sanitized;
}

function safeResolveWithin(parent: string, child: string): string {
  const parentResolved = resolve(parent);
  const candidate = resolve(parentResolved, child);
  if (
    candidate !== parentResolved &&
    !candidate.startsWith(`${parentResolved}${sep}`)
  ) {
    throw new HttpError(400, "path escapes benchmark workspace");
  }
  return candidate;
}

function readTextIfExists(path: string): string | null {
  if (!existsSync(path)) {
    return null;
  }
  return readFileSync(path, "utf8");
}

function readTextTail(path: string, maxBytes: number): string | null {
  const text = readTextIfExists(path);
  if (text === null) {
    return null;
  }
  if (text.length <= maxBytes) {
    return text;
  }
  return text.slice(-maxBytes);
}

function readJsonIfExists(path: string): Record<string, unknown> | null {
  if (!existsSync(path)) {
    return null;
  }
  try {
    const raw = readFileSync(path, "utf8");
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function defaultJobLauncher(request: BenchLabJobRequest): BenchLabLaunchedJob {
  mkdirSync(dirname(request.stdoutPath), { recursive: true });
  mkdirSync(dirname(request.stderrPath), { recursive: true });
  const stdoutStream = createWriteStream(request.stdoutPath, { flags: "w" });
  const stderrStream = createWriteStream(request.stderrPath, { flags: "w" });
  const child = spawn(request.command, request.args, {
    cwd: request.cwd,
    env: request.env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout?.pipe(stdoutStream);
  child.stderr?.pipe(stderrStream);

  const completion = new Promise<number>((resolvePromise, rejectPromise) => {
    child.once("error", (error) => {
      stdoutStream.end();
      stderrStream.end();
      rejectPromise(error);
    });
    child.once("close", (code) => {
      stdoutStream.end();
      stderrStream.end();
      resolvePromise(code ?? 1);
    });
  });

  return {
    pid: child.pid ?? null,
    kill: (signal = "SIGTERM") => {
      child.kill(signal);
    },
    completion,
  };
}

function serializeJob(job: BenchLabJobRecord): JsonObject {
  return {
    id: job.id,
    mode: job.mode,
    modelsFileName: job.modelsFileName,
    modelsFilePath: job.modelsFilePath,
    runtimeName: job.runtimeName,
    runtimeRoot: job.runtimeRoot,
    status: job.status,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    endedAt: job.endedAt,
    exitCode: job.exitCode,
    pid: job.pid,
    command: job.command,
    errorMessage: job.errorMessage,
    stdoutRelativePath: relative(job.runtimeRoot, job.stdoutPath),
    stderrRelativePath: relative(job.runtimeRoot, job.stderrPath),
  };
}

function readJobLogs(
  job: BenchLabJobRecord,
  stream: "stdout" | "stderr" | "both",
  maxBytes: number
): JsonObject {
  const includeStdout = stream === "stdout" || stream === "both";
  const includeStderr = stream === "stderr" || stream === "both";
  return {
    jobId: job.id,
    runtimeName: job.runtimeName,
    stdout: includeStdout
      ? {
          relativePath: relative(job.runtimeRoot, job.stdoutPath),
          text: readTextTail(job.stdoutPath, maxBytes),
        }
      : null,
    stderr: includeStderr
      ? {
          relativePath: relative(job.runtimeRoot, job.stderrPath),
          text: readTextTail(job.stderrPath, maxBytes),
        }
      : null,
  };
}

function attachJobCompletion(options: {
  job: BenchLabJobRecord;
  completion: Promise<number>;
  logger: Logger;
  stderrPath: string;
}): void {
  const { completion, job, logger, stderrPath } = options;
  completion
    .then((exitCode) => {
      job.exitCode = exitCode;
      job.endedAt = nowIso();
      if (job.status === "cancelled") {
        return;
      }
      job.status = exitCode === 0 ? "completed" : "failed";
      if (exitCode !== 0) {
        job.errorMessage =
          readTextIfExists(stderrPath)?.trim() || `exit code ${exitCode}`;
      }
    })
    .catch((error) => {
      job.endedAt = nowIso();
      job.exitCode = 1;
      job.status = "failed";
      job.errorMessage = toErrorMessage(error);
      logger.error("[benchlab] job failed", error);
    })
    .finally(() => {
      job.kill = null;
    });
}

function determinePrimaryOutcome(counts: Record<string, number>): string {
  if ((counts.failed ?? 0) > 0) {
    return "failed";
  }
  if ((counts.improved ?? 0) > 0) {
    return "improved";
  }
  if ((counts.regressed ?? 0) > 0) {
    return "regressed";
  }
  if ((counts.flat ?? 0) > 0) {
    return "flat";
  }
  if ((counts.preflight_ok ?? 0) > 0) {
    return "preflight_ok";
  }
  if ((counts.completed ?? 0) > 0) {
    return "completed";
  }
  return "unknown";
}

function readRuntimeSummary(runtimeRoot: string): BenchLabRuntimeSummary {
  const name = relative(resolve(runtimeRoot, ".."), runtimeRoot) || runtimeRoot;
  const summaryPath = join(runtimeRoot, "matrix_summary.json");
  const reportPath = join(runtimeRoot, "matrix_report.md");
  const summary = readJsonIfExists(summaryPath);
  const reportMarkdown = readTextIfExists(reportPath);
  const summaryCounts =
    summary && typeof summary.counts === "object" && summary.counts !== null
      ? (summary.counts as Record<string, number>)
      : {};

  let updatedAt: string | null = null;
  try {
    const sourcePath = existsSync(summaryPath) ? summaryPath : runtimeRoot;
    updatedAt = statSync(sourcePath).mtime.toISOString();
  } catch {
    updatedAt = null;
  }

  return {
    name,
    runtimeRoot,
    updatedAt,
    counts: summaryCounts,
    primaryOutcome: determinePrimaryOutcome(summaryCounts),
    categories: Array.isArray(summary?.categories)
      ? summary.categories.filter(
          (item): item is string => typeof item === "string"
        )
      : [],
    casesPerCategory:
      typeof summary?.cases_per_category === "number"
        ? summary.cases_per_category
        : null,
    preflightOnly:
      typeof summary?.preflight_only === "boolean"
        ? summary.preflight_only
        : null,
    modelsFileName:
      typeof summary?.models_file === "string"
        ? (summary.models_file.split(sep).pop() ?? summary.models_file)
        : null,
    reportMarkdown,
  };
}

function listRuntimeRoots(matrixRoot: string): string[] {
  if (!existsSync(matrixRoot)) {
    return [];
  }
  return readdirSync(matrixRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith("runtime"))
    .map((entry) => join(matrixRoot, entry.name));
}

function listRuntimeSummaries(matrixRoot: string): BenchLabRuntimeSummary[] {
  return listRuntimeRoots(matrixRoot)
    .map((runtimeRoot) => readRuntimeSummary(runtimeRoot))
    .sort((left, right) => {
      const leftTs = left.updatedAt ? Date.parse(left.updatedAt) : 0;
      const rightTs = right.updatedAt ? Date.parse(right.updatedAt) : 0;
      return rightTs - leftTs;
    });
}

function listConfigFiles(matrixRoot: string): BenchLabConfigDescriptor[] {
  if (!existsSync(matrixRoot)) {
    return [];
  }
  return readdirSync(matrixRoot, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.isFile() &&
        extname(entry.name) === ".json" &&
        entry.name.startsWith("models")
    )
    .map((entry) => ({
      name: entry.name,
      path: join(matrixRoot, entry.name),
      isRecommended: entry.name === DEFAULT_MODELS_FILE_NAME,
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

function parseJobRequestBody(body: unknown): {
  mode: "benchmark" | "preflight";
  modelsFileName: string;
  runtimeName: string;
  modelIds: string[];
  categories: string[];
  casesPerCategory: number;
  numThreads: number;
  maxStepLimit: number;
} {
  if (!body || typeof body !== "object") {
    throw new HttpError(400, "body must be a JSON object");
  }

  const rawMode = toOptionalString((body as JsonObject).mode) ?? "benchmark";
  if (rawMode !== "benchmark" && rawMode !== "preflight") {
    throw new HttpError(400, "mode must be benchmark or preflight");
  }

  const modelsFileName =
    toOptionalString((body as JsonObject).modelsFile) ??
    DEFAULT_MODELS_FILE_NAME;

  return {
    mode: rawMode,
    modelsFileName,
    runtimeName: sanitizeRuntimeName(
      toOptionalString((body as JsonObject).runtimeName)
    ),
    modelIds: toOptionalStringArray((body as JsonObject).modelIds),
    categories: toOptionalStringArray((body as JsonObject).categories),
    casesPerCategory: toPositiveInt(
      (body as JsonObject).casesPerCategory,
      rawMode === "preflight" ? 3 : 5
    ),
    numThreads: toPositiveInt((body as JsonObject).numThreads, 1),
    maxStepLimit: toPositiveInt((body as JsonObject).maxStepLimit, 20),
  };
}

function resolvePythonExecutable(
  benchmarkRoot: string,
  explicitValue: string | undefined
): string {
  if (toOptionalString(explicitValue)) {
    return String(explicitValue);
  }
  const venvPython = join(benchmarkRoot, ".venv311", "bin", "python");
  if (existsSync(venvPython)) {
    return venvPython;
  }
  return "python3";
}

export function createBenchLabApiServer(
  options: BenchLabApiServerOptions = {}
) {
  const logger = options.logger ?? console;
  const repoRoot = resolve(options.repoRoot ?? process.cwd());
  const benchmarkRoot = resolve(
    options.benchmarkRoot ??
      process.env.BFCL_ROOT ??
      "/Users/kim/Downloads/gorilla/berkeley-function-call-leaderboard"
  );
  const matrixRoot = join(repoRoot, "experiments", "prompt-bfcl-ralph-matrix");
  const matrixRunnerPath = join(repoRoot, DEFAULT_MATRIX_RUNNER_RELATIVE_PATH);
  const pythonExecutable = resolvePythonExecutable(
    benchmarkRoot,
    options.pythonExecutable ?? process.env.BENCHLAB_PYTHON_EXECUTABLE
  );
  const launchJob = options.jobLauncher ?? defaultJobLauncher;
  const jobs = new Map<string, BenchLabJobRecord>();

  function resolveConfigFile(name: string): BenchLabConfigDescriptor {
    const configs = listConfigFiles(matrixRoot);
    const match = configs.find((config) => config.name === name);
    if (!match) {
      throw new HttpError(404, `config not found: ${name}`);
    }
    return match;
  }

  function createJob(body: unknown): BenchLabJobRecord {
    const parsed = parseJobRequestBody(body);
    const config = resolveConfigFile(parsed.modelsFileName);
    const runtimeRoot = safeResolveWithin(matrixRoot, parsed.runtimeName);
    if (existsSync(runtimeRoot)) {
      throw new HttpError(409, `runtime already exists: ${parsed.runtimeName}`);
    }

    mkdirSync(runtimeRoot, { recursive: true });
    const command = [
      pythonExecutable,
      matrixRunnerPath,
      "--models-file",
      config.path,
      "--bfcl-root",
      benchmarkRoot,
      "--runtime-root",
      runtimeRoot,
      "--cases-per-category",
      String(parsed.casesPerCategory),
      "--num-threads",
      String(parsed.numThreads),
      "--max-step-limit",
      String(parsed.maxStepLimit),
    ];
    if (parsed.mode === "preflight") {
      command.push("--preflight-only");
    }
    if (parsed.modelIds.length > 0) {
      command.push("--model-ids", parsed.modelIds.join(","));
    }
    if (parsed.categories.length > 0) {
      command.push("--categories", parsed.categories.join(","));
    }

    const stdoutPath = join(runtimeRoot, "service-job.stdout.log");
    const stderrPath = join(runtimeRoot, "service-job.stderr.log");

    const launched = launchJob({
      command: command[0],
      args: command.slice(1),
      cwd: repoRoot,
      env: process.env,
      stdoutPath,
      stderrPath,
    });

    const job: BenchLabJobRecord = {
      id: `job-${randomUUID().slice(0, 8)}`,
      mode: parsed.mode,
      modelsFileName: config.name,
      modelsFilePath: config.path,
      runtimeName: parsed.runtimeName,
      runtimeRoot,
      status: "running",
      createdAt: nowIso(),
      startedAt: nowIso(),
      endedAt: null,
      exitCode: null,
      pid: launched.pid,
      command,
      errorMessage: null,
      stdoutPath,
      stderrPath,
      kill: launched.kill,
    };
    jobs.set(job.id, job);

    attachJobCompletion({
      job,
      completion: launched.completion,
      logger,
      stderrPath,
    });

    writeFileSync(
      join(runtimeRoot, "service-job.meta.json"),
      JSON.stringify(
        {
          jobId: job.id,
          mode: job.mode,
          createdAt: job.createdAt,
          modelsFileName: job.modelsFileName,
          command: job.command,
        },
        null,
        2
      )
    );

    return job;
  }

  function cancelJob(jobId: string): BenchLabJobRecord {
    const job = jobs.get(jobId);
    if (!job) {
      throw new HttpError(404, `job not found: ${jobId}`);
    }
    if (job.status !== "running" || job.kill === null) {
      throw new HttpError(409, `job is not running: ${jobId}`);
    }
    job.kill("SIGTERM");
    job.status = "cancelled";
    job.endedAt = nowIso();
    job.errorMessage = "cancelled by operator";
    return job;
  }

  function handleRootRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (
      request.method !== "GET" ||
      (pathname !== "/" && pathname !== "/benchlab")
    ) {
      return false;
    }
    sendHtml(response, 200, renderBenchLabDemoHtml());
    return true;
  }

  function handleHealthRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (request.method !== "GET" || pathname !== "/health") {
      return false;
    }
    sendJson(response, 200, {
      ok: true,
      benchmarkRoot,
      matrixRoot,
      pythonExecutable,
      repoRoot,
    });
    return true;
  }

  function handleConfigsRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (request.method !== "GET" || pathname !== "/v1/benchlab/configs") {
      return false;
    }
    sendJson(response, 200, {
      configs: listConfigFiles(matrixRoot).map((config) => ({
        isRecommended: config.isRecommended,
        name: config.name,
        relativePath: relative(repoRoot, config.path),
      })),
    });
    return true;
  }

  function handleJobsRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (request.method !== "GET" || pathname !== "/v1/benchlab/jobs") {
      return false;
    }
    const serialized = [...jobs.values()]
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map(serializeJob);
    sendJson(response, 200, { jobs: serialized });
    return true;
  }

  function handleJobLogsRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string,
    requestUrl: URL
  ): boolean {
    if (
      request.method !== "GET" ||
      !pathname.startsWith("/v1/benchlab/jobs/") ||
      !pathname.endsWith("/logs")
    ) {
      return false;
    }

    const jobId = decodeURIComponent(
      pathname.slice("/v1/benchlab/jobs/".length, -"/logs".length)
    );
    const job = jobs.get(jobId);
    if (!job) {
      throw new HttpError(404, `job not found: ${jobId}`);
    }

    const rawStream = requestUrl.searchParams.get("stream");
    const stream =
      rawStream === "stdout" || rawStream === "stderr" ? rawStream : "both";
    const rawMaxBytes = Number.parseInt(
      requestUrl.searchParams.get("maxBytes") ?? "",
      10
    );
    const maxBytes =
      Number.isFinite(rawMaxBytes) && rawMaxBytes > 0
        ? Math.min(rawMaxBytes, 200_000)
        : 20_000;

    sendJson(response, 200, {
      logs: readJobLogs(job, stream, maxBytes),
    });
    return true;
  }

  function handleJobDetailRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (
      request.method !== "GET" ||
      !pathname.startsWith("/v1/benchlab/jobs/") ||
      pathname.endsWith("/logs")
    ) {
      return false;
    }

    const jobId = decodeURIComponent(
      pathname.slice("/v1/benchlab/jobs/".length)
    );
    const job = jobs.get(jobId);
    if (!job) {
      throw new HttpError(404, `job not found: ${jobId}`);
    }

    sendJson(response, 200, { job: serializeJob(job) });
    return true;
  }

  function handleCancelJobRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (
      request.method !== "POST" ||
      !pathname.startsWith("/v1/benchlab/jobs/") ||
      !pathname.endsWith("/cancel")
    ) {
      return false;
    }

    const jobId = decodeURIComponent(
      pathname.slice("/v1/benchlab/jobs/".length, -"/cancel".length)
    );
    const job = cancelJob(jobId);
    sendJson(response, 200, { job: serializeJob(job) });
    return true;
  }

  async function handleCreateJobRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): Promise<boolean> {
    if (request.method !== "POST" || pathname !== "/v1/benchlab/jobs") {
      return false;
    }
    const body = await readJsonBody(request);
    const job = createJob(body);
    sendJson(response, 202, { job: serializeJob(job) });
    return true;
  }

  function handleRunsRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (request.method !== "GET" || pathname !== "/v1/benchlab/runs") {
      return false;
    }
    sendJson(response, 200, {
      runs: listRuntimeSummaries(matrixRoot),
    });
    return true;
  }

  function handleRunDetailRoute(
    request: IncomingMessage,
    response: ServerResponse,
    pathname: string
  ): boolean {
    if (
      request.method !== "GET" ||
      !pathname.startsWith("/v1/benchlab/runs/")
    ) {
      return false;
    }

    const runName = decodeURIComponent(
      pathname.slice("/v1/benchlab/runs/".length)
    );
    const runtimeRoot = safeResolveWithin(matrixRoot, runName);
    if (!existsSync(runtimeRoot)) {
      throw new HttpError(404, `runtime not found: ${runName}`);
    }

    sendJson(response, 200, {
      run: readRuntimeSummary(runtimeRoot),
    });
    return true;
  }

  type BenchLabRouteHandler = (
    request: IncomingMessage,
    response: ServerResponse,
    requestUrl: URL
  ) => boolean | Promise<boolean>;

  const routeHandlers: BenchLabRouteHandler[] = [
    (request, response, requestUrl) =>
      handleRootRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleHealthRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleConfigsRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleJobsRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleJobLogsRoute(request, response, requestUrl.pathname, requestUrl),
    (request, response, requestUrl) =>
      handleJobDetailRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleCancelJobRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleCreateJobRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleRunsRoute(request, response, requestUrl.pathname),
    (request, response, requestUrl) =>
      handleRunDetailRoute(request, response, requestUrl.pathname),
  ];

  return createServer(async (request, response) => {
    const requestUrl = parseRequestUrl(request.url);
    try {
      for (const handler of routeHandlers) {
        if (await handler(request, response, requestUrl)) {
          return;
        }
      }

      sendJson(response, 404, { error: "route not found" });
    } catch (error) {
      const httpError = toHttpError(error);
      logger.error("[benchlab] request failed", httpError.message);
      sendJson(response, httpError.statusCode, {
        error: httpError.message,
      });
    }
  });
}
