import { loadStaticHtmlTemplate } from "./static-html-template";

const htmlTemplate = loadStaticHtmlTemplate(
  import.meta.url,
  "stagepilot-demo.html"
);

export function renderStagePilotDemoHtml(): string {
  return htmlTemplate;
}
