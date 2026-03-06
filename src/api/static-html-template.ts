import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export function loadStaticHtmlTemplate(
  moduleUrl: string,
  templateFileName: string
): string {
  const currentDir = dirname(fileURLToPath(moduleUrl));
  const templatePath = join(currentDir, templateFileName);
  return readFileSync(templatePath, "utf8");
}
