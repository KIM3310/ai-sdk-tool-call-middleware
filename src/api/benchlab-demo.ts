import { loadStaticHtmlTemplate } from "./static-html-template";

const htmlTemplate = loadStaticHtmlTemplate(
  import.meta.url,
  "benchlab-demo.html"
);

export function renderBenchLabDemoHtml(): string {
  return htmlTemplate;
}
