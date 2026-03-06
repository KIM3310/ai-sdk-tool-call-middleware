import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const currentDir = dirname(fileURLToPath(import.meta.url));
const templatePath = join(currentDir, "benchlab-demo.html");
const htmlTemplate = readFileSync(templatePath, "utf8");

export function renderBenchLabDemoHtml(): string {
  return htmlTemplate;
}
