type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | "DOWNLOAD";

export type UiCategory =
  | "matte"
  | "session"
  | "upload"
  | "mode"
  | "detect"
  | "segment"
  | "brush"
  | "sam3"
  | "subject"
  | "layer"
  | "export"
  | "settings"
  | "console";

export interface UiLogEntry {
  id: number;
  source: "ui";
  level: LogLevel;
  category: UiCategory;
  message: string;
  ts: string;
  detail?: string;
}

const subs = new Set<(e: UiLogEntry) => void>();
let seq = 0;

const clock = () => new Date().toTimeString().slice(0, 8);

const PATH_RE = /[A-Za-z]:\\[^\s"']+|\/(?:home|Users|root|tmp|var|opt|srv|workspace|data)\/[^\s"']+/gi;
const SECRET_RE = /(token|bearer|x-local-token)\s*[:=]\s*\S+/gi;
const UUID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
const LONGHEX_RE = /\b[0-9a-f]{32,}\b/gi;
const REQUEST_LINE_RE = /^(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+(\S+)\s+(\d{3})\s+-\s+/;

const REQUEST_LABELS: [RegExp, string][] = [
  [/\/predict$/, "Brush updates"],
  [/\/segment-batch$/, "Segmentation"],
  [/\/detect$/, "Detection"],
  [/\/sam3-(?:prompt|undo|redo|remove-instance)$/, "SAM 3 prompts"],
  [/\/(?:undo|redo)$/, "Undo and redo"],
  [/\/clear-object$/, "Subject cleared"],
  [/\/image$/, "Image uploads"],
  [/\/export-(?:zip|image|matte)$/, "Exports"],
  [/\/matte$/, "Matte updates"],
];

/**
 * Turn a successful request line into a stable group key. The duration is
 * deliberately excluded: two brush strokes are different requests, but they
 * should collapse into one console row.
 */
export function requestSummary(message: string): { key: string; label: string } | null {
  const match = REQUEST_LINE_RE.exec(message);
  if (!match) return null;
  const [, method, rawPath, status] = match;
  if (!/^2\d\d$/.test(status)) return null;

  const path = rawPath.split("?", 1)[0].replace(/\/api\/sessions\/[^/]+/, "/api/sessions/:id");
  const label = REQUEST_LABELS.find(([pattern]) => pattern.test(path))?.[1] ?? `${method} requests`;
  return { key: `${method} ${path} ${status}`, label };
}

export function safeText(value: unknown, max = 300): string {
  let s: string;
  if (value instanceof Error) s = `${value.name}: ${value.message}`;
  else if (typeof value === "string") s = value;
  else {
    try {
      s = JSON.stringify(value) ?? String(value);
    } catch {
      s = String(value);
    }
  }
  s = s
    .replace(PATH_RE, "[path]")
    .replace(SECRET_RE, "[redacted]")
    .replace(UUID_RE, "[id]")
    .replace(LONGHEX_RE, "[hash]");
  return s.length > max ? `${s.slice(0, max)}...` : s;
}

export function emitUi(
  category: UiCategory,
  level: LogLevel,
  message: string,
  detail?: unknown
): void {
  const entry: UiLogEntry = {
    id: ++seq,
    source: "ui",
    level,
    category,
    message: safeText(message),
    ts: clock(),
  };
  if (detail !== undefined) entry.detail = safeText(detail, 200);
  for (const fn of subs) {
    try {
      fn(entry);
    } catch {
    }
  }
}

export const logErr = (category: UiCategory, err: unknown) =>
  emitUi(category, "ERROR", safeText(err));

export function subscribeLogs(fn: (e: UiLogEntry) => void): () => void {
  subs.add(fn);
  return () => {
    subs.delete(fn);
  };
}

export const uiCategoryLabels: Record<UiCategory, string> = {
  matte: "Matte",
  session: "Session",
  upload: "Upload",
  mode: "Mode",
  detect: "Detect",
  segment: "Segment",
  brush: "Brush",
  sam3: "SAM3",
  subject: "Subject",
  layer: "Layer",
  export: "Export",
  settings: "Settings",
  console: "Console",
};
