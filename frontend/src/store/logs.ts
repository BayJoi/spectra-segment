export type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | "DOWNLOAD";

export type UiCategory =
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
const LONGHEX_RE = /\b[0-9a-f]{32,}\b/gi;

function safeText(value: unknown, max = 300): string {
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
  s = s.replace(PATH_RE, "[path]").replace(SECRET_RE, "[redacted]").replace(LONGHEX_RE, "[hash]");
  return s.length > max ? `${s.slice(0, max)}…` : s;
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
