import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { api, BASE } from "@/lib/api";
import { connectSse } from "@/lib/sse";
import { cn } from "@/lib/utils";
import {
  consoleOpenAtom,
  consoleFilterAtom,
  consoleSourceAtom,
  consoleCategoryAtom,
  consoleShowTimeAtom,
  type ConsoleFilterKey,
  type ConsoleSourceKey,
} from "@/store/ui";
import { modelsAtom } from "@/store/session";
import {
  subscribeLogs,
  uiCategoryLabels,
  type UiCategory,
  type UiLogEntry,
} from "@/store/logs";

interface LogEntry {
  seq: number;
  id: string;
  source: "ui" | "backend";
  category?: string;
  level: string;
  message: string;
  hay?: string;
  ts?: string;
  name?: string;
  pct?: number;
  file?: string;
  files_done?: number | null;
  files_total?: number | null;
  phase?: "starting" | "download" | "verify" | "load" | "done";
  done_bytes?: number;
  total_bytes?: number;
  eta?: string;
}

const LEVEL_RANK: Record<string, number> = {
  DEBUG: -1,
  INFO: 0,
  WARNING: 1,
  ERROR: 2,
  CRITICAL: 3,
  DOWNLOAD: -1,
};

const FILTER_OPTIONS: { key: ConsoleFilterKey; label: string; minRank: number }[] = [
  { key: "all", label: "All", minRank: -Infinity },
  { key: "debug", label: "Debug+", minRank: -1 },
  { key: "info", label: "Info+", minRank: 0 },
  { key: "warn", label: "Warn+", minRank: 1 },
  { key: "error", label: "Errors", minRank: 2 },
];


const FILTER_COMBOS: { key: string; filter: ConsoleFilterKey; source: ConsoleSourceKey; label: string }[] = [
  { key: "all", filter: "all", source: "all", label: "Everything" },
  { key: "app", filter: "all", source: "ui", label: "App only" },
  { key: "server", filter: "all", source: "backend", label: "Server only" },
  { key: "warn", filter: "warn", source: "all", label: "Warnings & errors" },
  { key: "error", filter: "error", source: "all", label: "Errors only" },
  { key: "debug", filter: "debug", source: "all", label: "Debug & above" },
];

const UI_CATEGORIES = Object.keys(uiCategoryLabels) as UiCategory[];

const MAX_LOG_ENTRIES = 500;
const MIN_UI_ENTRIES = 150;
const MAX_MESSAGE = 400;
const RENDER_LIMIT = 120;

let localSeq = 0;

const nextId = () => `e${++localSeq}`;

const normalise = (raw: Partial<LogEntry>): LogEntry => {
  const message = String(raw.message ?? "").slice(0, MAX_MESSAGE);
  const seq = ++localSeq;
  const entry: LogEntry = {
    seq,
    id: nextId(),
    source: raw.source ?? "backend",
    level: raw.level ?? "INFO",
    message,
    hay: `${message} ${raw.level ?? ""} ${raw.category ?? ""} ${raw.source ?? "backend"} ${raw.name ?? ""}`.toLowerCase(),
  };
  if (raw.ts !== undefined) entry.ts = raw.ts;
  if (raw.category) entry.category = raw.category;
  for (const k of [
    "name", "pct", "file", "files_done", "files_total",
    "phase", "done_bytes", "total_bytes", "eta",
  ] as const) {
    if (raw[k] !== undefined && raw[k] !== null) entry[k] = raw[k] as never;
  }
  return entry;
};

const toLogEntry = (u: UiLogEntry): LogEntry =>
  normalise({
    source: u.source,
    category: u.category,
    level: u.level,
    message: u.message,
    ts: u.ts,
  });

export function ConsolePanel({ encoding }: { encoding?: boolean }) {
  const [open, setOpen] = useAtom(consoleOpenAtom);
  const [filter, setFilter] = useAtom(consoleFilterAtom);
  const [sourceFilter, setSourceFilter] = useAtom(consoleSourceAtom);
  const [categoryFilter, setCategoryFilter] = useAtom(consoleCategoryAtom);
  const [showTime, setShowTime] = useAtom(consoleShowTimeAtom);
  const [models] = useAtom(modelsAtom);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [search, setSearch] = useState("");
  const [connected, setConnected] = useState(false);
  const [unread, setUnread] = useState(0);
  const [badge, setBadge] = useState<"error" | "warn" | null>(null);
  const [showJump, setShowJump] = useState(false);
  const [copied, setCopied] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [renderLimit, setRenderLimit] = useState(RENDER_LIMIT);

  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const hoverRef = useRef(false);
  const openRef = useRef(open);
  openRef.current = open;
  const badgeRef = useRef(badge);
  badgeRef.current = badge;
  const readyRef = useRef(false);
  const pendingRef = useRef<LogEntry[]>([]);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const trim = useCallback((arr: LogEntry[]): LogEntry[] => {
    if (arr.length <= MAX_LOG_ENTRIES) return arr;
    const uiTail = arr.filter((e) => e.source === "ui").slice(-MIN_UI_ENTRIES);
    if (uiTail.length === 0) return arr.slice(arr.length - MAX_LOG_ENTRIES);
    const reserved = new Set(uiTail);
    const rest = arr
      .filter((e) => !reserved.has(e))
      .slice(-(MAX_LOG_ENTRIES - uiTail.length));
    return [...rest, ...uiTail].sort((a, b) => a.seq - b.seq).slice(-MAX_LOG_ENTRIES);
  }, []);

  const appendEntry = useCallback((entry: LogEntry) => {
    if (!readyRef.current) {
      pendingRef.current.push(entry);
      return;
    }
    setLogs((prev) => trim([...prev, entry]));
    if (entry.level === "ERROR") setBadge("error");
    else if (entry.level === "WARNING" && badgeRef.current !== "error") setBadge("warn");
    else if (entry.level === "CRITICAL") setBadge("error");
    if (!openRef.current || !pinnedRef.current || hoverRef.current) {
      if (!openRef.current) setUnread((u) => u + 1);
    }
  }, [trim]);

  useEffect(() => {
    let cancelled = false;

    api.getLogs(MAX_LOG_ENTRIES).then((res) => {
      if (cancelled) return;
      const entries = (res.entries as LogEntry[]).map(normalise);
      setLogs(() => trim([...entries, ...pendingRef.current]));
      pendingRef.current = [];
      readyRef.current = true;
    }).catch(() => {
      if (cancelled) return;
      readyRef.current = true;
      setLogs((prev) => trim([...prev, ...pendingRef.current]));
      pendingRef.current = [];
    });

    const unsubUi = subscribeLogs((e) => appendEntry(toLogEntry(e)));

    const handle = connectSse(`${BASE}/api/logs/stream`, (data) => {
      try {
        appendEntry(normalise(JSON.parse(data)));
      } catch {
      }
    }, setConnected);

    return () => {
      cancelled = true;
      unsubUi();
      handle.close();
    };
  }, [appendEntry, trim]);

  useEffect(() => {
    if (!open) return;
    const el = scrollRef.current;
    if (!el) return;

    if (pinnedRef.current && !hoverRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [open]);

  useEffect(() => {
    if (open) {
      setUnread(0);
      setBadge(null);
      setRenderLimit(RENDER_LIMIT);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const el = scrollRef.current;
    if (!el) return;
    if (pinnedRef.current && !hoverRef.current) el.scrollTop = el.scrollHeight;
    else if (logs.length > renderLimit) setShowJump(true);
  }, [open, logs, renderLimit]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const t = e.target as Node;
      if (dropdownRef.current && !dropdownRef.current.contains(t)) setFilterOpen(false);
      if (dropdownRef.current && !dropdownRef.current.contains(t)) setFilterOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const progressRows = useMemo(() => {
    interface Row {
      pct: number;
      message: string;
      phase: NonNullable<LogEntry["phase"]>;
      file?: string;
      files_done?: number | null;
      files_total?: number | null;
      done_bytes?: number;
      total_bytes?: number;
      eta?: string;
    }
    if (!open) return [] as [string, Row][];
    const map = new Map<string, Row>();
    for (const l of logs) {
      if (l.level !== "DOWNLOAD" || !l.name) continue;
      const phase = l.phase ?? "download";
      const prev = map.get(l.name);
      const filesTotal = l.files_total ?? prev?.files_total;
      const filesDone = l.files_done ?? prev?.files_done;
      const pct = l.pct ?? prev?.pct ?? 0;
      if (phase === "done" || (pct >= 100 && !filesTotal)) {
        map.delete(l.name);
        continue;
      }
      map.set(l.name, {
        pct,
        message: l.message,
        phase: phase === "starting" && prev?.phase && prev.phase !== "starting" ? prev.phase : phase,
        file: l.file ?? prev?.file,
        files_done: filesDone,
        files_total: filesTotal,
        done_bytes: l.done_bytes ?? prev?.done_bytes,
        total_bytes: l.total_bytes ?? prev?.total_bytes,
        eta: l.eta ?? prev?.eta,
      });
    }
    return [...map.entries()];
  }, [logs, open]);

  const categoryCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const l of logs) {
      if (l.level === "DOWNLOAD") continue;
      const c = l.source === "ui" ? l.category ?? "app" : "backend";
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    return counts;
  }, [logs]);

  const filteredLogs = useMemo(() => {
    const minRank = FILTER_OPTIONS.find((f) => f.key === filter)?.minRank ?? -Infinity;
    const q = search.trim().toLowerCase();
    return logs.filter((e) => {
      if (e.level === "DOWNLOAD") return false;
      if (sourceFilter !== "all" && e.source !== sourceFilter) return false;
      if (categoryFilter !== "all") {
        const cat = e.source === "ui" ? e.category ?? "app" : "backend";
        if (cat !== categoryFilter) return false;
      }
      if ((LEVEL_RANK[e.level] ?? 0) < minRank) return false;
      if (q && !(e.hay ?? e.message.toLowerCase()).includes(q)) return false;
      return true;
    });
  }, [logs, filter, search, sourceFilter, categoryFilter]);

  const visibleLogs = useMemo(
    () => (filteredLogs.length > renderLimit ? filteredLogs.slice(-renderLimit) : filteredLogs),
    [filteredLogs, renderLimit]
  );

  const hiddenCount = filteredLogs.length - visibleLogs.length;
  const errorCount = filteredLogs.filter((e) => e.level === "ERROR" || e.level === "CRITICAL").length;
  const warnCount = filteredLogs.filter((e) => e.level === "WARNING").length;
  const filterLabel =
    categoryFilter !== "all"
      ? (uiCategoryLabels[categoryFilter as UiCategory] ?? categoryFilter)
      : sourceFilter !== "all"
        ? (sourceFilter === "ui" ? "App" : "Server")
        : (FILTER_OPTIONS.find((f) => f.key === filter)?.label ?? "All");

  const activeCategories = useMemo(() => {
    return UI_CATEGORIES
      .map((c) => [c, categoryCounts.get(c) ?? 0] as [UiCategory, number])
      .filter(([, n]) => n > 0)
      .sort((a, b) => b[1] - a[1]);
  }, [categoryCounts]);

  const clearLogs = useCallback(() => {
    setLogs([]);
    setUnread(0);
    setBadge(null);
    setShowJump(false);
    setRenderLimit(RENDER_LIMIT);
  }, []);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    hoverRef.current = false;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setShowJump(false);
  }, []);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    pinnedRef.current = atBottom;
    setShowJump(!atBottom);
  }, []);

  const handleCopy = useCallback(async () => {
    const lines = filteredLogs.map((l) => {
      const src = l.source === "ui" ? `[${l.category ?? "app"}]` : "[server]";
      return `[${l.ts ?? "--:--:--"}] [${l.level}] ${src} ${l.message}`;
    });
    if (progressRows.length > 0) {
      lines.push("", "[downloads]");
      for (const [name, p] of progressRows) {
        const files = p.files_total && p.files_total > 1 ? ` files ${p.files_done ?? 0}/${p.files_total}` : "";
        lines.push(`  ${name}: ${p.pct}%${files}${p.eta ? ` ETA ${p.eta}` : ""}`);
      }
    }
    const text = lines.join("\n");
    let ok = false;
    if (navigator.clipboard?.writeText && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch {
        ok = false;
      }
    }
    if (!ok) {
      let ta: HTMLTextAreaElement | null = null;
      try {
        ta = document.createElement("textarea");
        ta.value = text;
        ta.readOnly = true;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand("copy");
      } catch {
        ok = false;
      } finally {
        if (ta) ta.remove();
      }
    }
    setCopied(ok);
    if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    copiedTimerRef.current = setTimeout(() => setCopied(false), 1500);
  }, [filteredLogs, progressRows]);

  const downloadedModels = models.filter((m) => m.downloaded).length;

  return (
    <div ref={panelRef} className="relative z-[85]">
      <button
        onClick={() => setOpen(!open)}
        aria-label="Console"
        aria-expanded={open}
        aria-controls="console-panel"
        className={cn(
          "relative inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans transition-all duration-200 cursor-pointer select-none",
          "border hover:scale-105 active:scale-95 grain-bg",
          open
            ? "bg-orange-500/15 border-orange-500/30 text-orange-300 shadow-[0_0_12px_rgba(184,92,42,0.15)] grain-bg-strong"
            : encoding
              ? "bg-neutral-800/70 border-orange-500/30 text-orange-300 grain-bg-strong"
              : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700 hover:bg-neutral-800/50"
        )}
        title="Console"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <polyline points="4,17 10,11 4,5" /><line x1="12" y1="19" x2="20" y2="19" />
        </svg>
        Logs
        {!open && unread > 0 && (
          <span
            className="absolute -top-1.5 -right-1.5 min-w-[14px] h-[14px] px-0.5 flex items-center justify-center rounded-full bg-orange-500 text-[8px] font-bold text-white tabular-nums"
            aria-label={`${unread} new log entries`}
          >
            {unread > 99 ? "99+" : unread}
          </span>
        )}
        {!open && unread === 0 && badge && (
          <span
            className={cn(
              "absolute -top-1 -right-1 w-2 h-2 rounded-full",
              badge === "error" ? "bg-red-500 animate-pulse" : "bg-yellow-400"
            )}
            aria-label={badge === "error" ? "Errors in the console" : "Warnings in the console"}
          />
        )}
        {!open && !badge && !connected && (
          <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-neutral-600" aria-label="Console disconnected" />
        )}
      </button>

      {open && (
        <div
          id="console-panel"
          role="region"
          aria-label="Console output"
          className="absolute top-full right-0 mt-3 w-[380px] bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-xl shadow-2xl shadow-black/60 backdrop-blur-xl overflow-hidden animate-drop-in z-50 grain-bg"
        >
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-neutral-800/60">
            <span className="text-[10px] text-neutral-500 uppercase tracking-wider font-sans">
              Console
              <span className={cn("inline-block w-1.5 h-1.5 rounded-full ml-1.5 align-middle", connected ? "bg-green-400/70" : "bg-neutral-600")} />
            </span>
            <div className="flex items-center gap-1">
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search..."
                aria-label="Search logs"
                className="w-[92px] h-5 px-1.5 bg-neutral-900/70 border border-neutral-800/60 rounded text-[10px] font-mono text-neutral-300 placeholder:text-neutral-600 outline-none focus:border-neutral-600 transition-colors"
              />
              <div ref={dropdownRef} className="relative">
                <button
                  onClick={() => setFilterOpen(!filterOpen)}
                  aria-label="Filter logs"
                  aria-expanded={filterOpen}
                  className="flex items-center gap-1 text-[10px] font-sans bg-transparent border border-neutral-800/60 rounded px-1.5 py-0.5 text-neutral-400 cursor-pointer outline-none hover:border-neutral-700 hover:scale-105 active:scale-95 transition-all duration-200"
                >
                  {filterLabel}
                  <svg
                    width="8" height="8" viewBox="0 0 24 24"
                    fill="none" stroke="currentColor" strokeWidth="2"
                    strokeLinecap="round" strokeLinejoin="round"
                    className={cn("transition-transform", filterOpen && "rotate-180")}
                  >
                    <polyline points="6,9 12,15 18,9" />
                  </svg>
                </button>
                {filterOpen && (
                  <div
                    role="listbox"
                    aria-label="Log filter"
                    className="animate-drop-in absolute right-0 top-full mt-1 w-[186px] bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl z-50 overflow-hidden grain-bg"
                  >
                    <div className="max-h-[264px] overflow-y-auto custom-scrollbar py-1">
                    {FILTER_COMBOS.map((o) => (
                      <button
                        key={o.key}
                        role="option"
                        aria-selected={filter === o.filter && sourceFilter === o.source}
                        onClick={() => {
                          setFilter(o.filter);
                          setSourceFilter(o.source);
                          setFilterOpen(false);
                        }}
                        className={cn(
                          "block w-full text-left text-[11px] font-sans px-3 py-1.5 transition-colors cursor-pointer",
                          filter === o.filter && sourceFilter === o.source
                            ? "text-orange-400"
                            : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
                        )}
                      >
                        {o.label}
                      </button>
                    ))}
                    {activeCategories.length > 0 && (
                      <>
                        <div className="px-3 pt-2 pb-1 text-[9px] font-medium uppercase tracking-widest text-neutral-600 font-sans select-none">
                          Category
                        </div>
                        {activeCategories.map(([c, n]) => (
                          <button
                            key={c}
                            role="option"
                            aria-selected={categoryFilter === c}
                            onClick={() => { setCategoryFilter(categoryFilter === c ? "all" : c); setFilterOpen(false); }}
                            className={cn(
                              "flex items-center justify-between w-full text-left text-[11px] font-sans px-3 py-1.5 transition-colors cursor-pointer",
                              categoryFilter === c
                                ? "text-orange-400"
                                : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
                            )}
                          >
                            <span>{uiCategoryLabels[c]}</span>
                            <span className="text-neutral-600 tabular-nums">{n}</span>
                          </button>
                        ))}
                      </>
                    )}
                    {categoryFilter !== "all" && (
                      <>
                        <div className="border-t border-neutral-800 my-1" />
                        <button
                          onClick={() => { setCategoryFilter("all"); setFilterOpen(false); }}
                          className="block w-full text-left text-[11px] font-sans px-3 py-1.5 text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/60 transition-colors cursor-pointer"
                        >
                          Clear category
                        </button>
                      </>
                    )}
                    </div>
                  </div>
                )}
              </div>
              <button
                onClick={() => setShowTime(!showTime)}
                title="Toggle timestamps"
                aria-label="Toggle timestamps"
                aria-pressed={showTime}
                className={cn(
                  "flex items-center text-[10px] font-mono px-1.5 py-0.5 rounded border transition-all duration-150 cursor-pointer hover:scale-105 active:scale-95",
                  showTime
                    ? "text-orange-400 border-orange-500/30 bg-orange-500/10"
                    : "text-neutral-500 border-transparent hover:text-neutral-300"
                )}
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="9" /><polyline points="12,7 12,12 15.5,13.5" />
                </svg>
              </button>
              <button
                onClick={clearLogs}
                title="Clear the console"
                aria-label="Clear console"
                className="flex items-center text-[10px] font-mono px-1.5 py-0.5 rounded border border-transparent text-neutral-500 hover:text-red-400 hover:bg-red-500/10 transition-all duration-150 cursor-pointer"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M18 6 6 18" /><path d="m6 6 12 12" />
                </svg>
              </button>
              <button
                onClick={handleCopy}
                aria-label="Copy logs"
                className={cn(
                  "text-[10px] font-mono px-1.5 py-0.5 rounded transition-all duration-150 cursor-pointer hover:scale-105 active:scale-95",
                  copied
                    ? "text-green-400 bg-green-400/10"
                    : "text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/60"
                )}
                role="status"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 px-3 py-[3px] border-b border-neutral-800/60 text-[9px] font-mono text-neutral-600">
            <span className="tabular-nums">
              {visibleLogs.length}{filteredLogs.length > visibleLogs.length ? ` +${hiddenCount}` : ""} shown
            </span>
            {errorCount > 0 && <span className="text-red-400/80 tabular-nums">· {errorCount} err</span>}
            {warnCount > 0 && <span className="text-yellow-400/70 tabular-nums">· {warnCount} warn</span>}
            {categoryFilter !== "all" && (
              <button
                onClick={() => setCategoryFilter("all")}
                className="text-orange-400 hover:underline cursor-pointer"
              >
                {uiCategoryLabels[categoryFilter as UiCategory] ?? categoryFilter} ×
              </button>
            )}
            {sourceFilter !== "all" && (
              <button
                onClick={() => setSourceFilter("all")}
                className="text-orange-400 hover:underline cursor-pointer"
              >
                {sourceFilter === "ui" ? "App" : "Server"} ×
              </button>
            )}
          </div>

          {models.length > 0 && (
            <div className="border-b border-neutral-800/60">
              <button
                onClick={() => setModelsOpen(!modelsOpen)}
                className="flex items-center gap-1.5 w-full px-3 py-1.5 text-[10px] font-mono cursor-pointer hover:bg-neutral-900/40 transition-colors"
              >
                <svg
                  width="8" height="8" viewBox="0 0 24 24"
                  fill="none" stroke="currentColor" strokeWidth="2"
                  strokeLinecap="round" strokeLinejoin="round"
                  className={cn("transition-transform text-neutral-500", modelsOpen && "rotate-90")}
                >
                  <polyline points="9,18 15,12 9,6" />
                </svg>
                <span className="text-neutral-600 tracking-wider uppercase font-sans">Models</span>
                <span className="ml-auto text-neutral-600">
                  {downloadedModels} of {models.length} downloaded
                </span>
              </button>
              {modelsOpen && (
                <div className="px-3 pb-2 space-y-1">
                  {models.map((m) => (
                    <div key={m.name} className="flex items-center gap-2 text-[10px] font-mono">
                      <span className={cn(
                        "w-1.5 h-1.5 rounded-full shrink-0",
                        m.loaded ? "bg-orange-400" : m.downloaded ? "bg-green-400" : "bg-neutral-600"
                      )} />
                      <span className={cn(
                        m.loaded ? "text-orange-300" : m.downloaded ? "text-green-400/80" : "text-neutral-500"
                      )}>
                        {m.display_name}
                      </span>
                      {m.loaded && (
                        <span className="text-[8px] text-orange-400/60 uppercase tracking-wider ml-auto">Active</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {progressRows.length > 0 && (
            <div className="border-b border-neutral-800/60 px-3 py-2 space-y-2">
              {progressRows.map(([name, p]) => {
                const gb = 1024 ** 3;
                const mb = 1024 ** 2;
                const sizeText =
                  p.total_bytes && p.total_bytes >= gb
                    ? `${((p.done_bytes ?? 0) / gb).toFixed(1)} / ${(p.total_bytes / gb).toFixed(1)} GB`
                    : p.total_bytes
                      ? `${Math.round((p.done_bytes ?? 0) / mb)} / ${Math.round(p.total_bytes / mb)} MB`
                      : null;
                const chip = p.phase === "verify" ? "Verifying" : p.phase === "starting" ? "Starting" : "Downloading";
                const hot = p.pct > 90;
                const files = p.files_total && p.files_total > 1 ? `${p.files_done ?? 0}/${p.files_total}` : null;
                return (
                  <div key={name}>
                    <div className="flex items-center justify-between text-[9px] font-mono text-orange-300/80 mb-0.5">
                      <span className="truncate mr-2">{name}</span>
                      <span className="flex items-center gap-1.5 shrink-0">
                        <span className="text-orange-400">{chip}</span>
                        <span className="tabular-nums">{p.pct}%</span>
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-neutral-800 overflow-hidden">
                      <div
                        className={cn(
                          "h-full rounded-full transition-[width] duration-300",
                          hot ? "bg-orange-400" : "bg-gradient-to-r from-orange-500 to-orange-400"
                        )}
                        style={{ width: `${p.pct}%` }}
                      />
                    </div>
                    {(files || sizeText || p.eta) && (
                      <div className="flex items-center justify-between text-[8px] font-mono text-neutral-500 mt-0.5">
                        <span className="truncate">{[sizeText, p.file].filter(Boolean).join(" · ")}</span>
                        <span className="shrink-0 ml-2 flex items-center gap-1.5">
                          {files && <span>{files}</span>}
                          {p.eta && <span>ETA {p.eta}</span>}
                        </span>
                      </div>
                    )}
                    {files && (
                      <div className="h-0.5 rounded-full bg-neutral-800 overflow-hidden mt-0.5">
                        <div
                          className="h-full bg-orange-500/70 transition-[width] duration-300"
                          style={{ width: `${(((p.files_done ?? 0) / p.files_total!) * 100).toFixed(0)}%` }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          <div className="relative">
            <div
              ref={scrollRef}
              onScroll={handleScroll}
              onMouseEnter={() => { hoverRef.current = true; }}
              onMouseLeave={() => { hoverRef.current = false; }}
              className="overflow-y-auto custom-scrollbar px-3 py-2 h-64 font-mono text-[11px] leading-relaxed"
              role="log"
              aria-live="polite"
              aria-relevant="additions"
            >
              {filteredLogs.length === 0 ? (
                <p className="text-neutral-600 text-center py-4">
                  {logs.length === 0 ? "No logs yet" : "No matches — clear the search or filters"}
                </p>
              ) : (
                visibleLogs.map((entry) => (
                  <div key={entry.id} className="flex items-start gap-2 py-[3px]">
                    {showTime && entry.ts && (
                      <span className="text-neutral-600 tabular-nums shrink-0">{entry.ts}</span>
                    )}
                    <span
                      className={cn(
                        "w-1.5 h-1.5 rounded-full mt-[5px] shrink-0",
                        entry.level === "ERROR" || entry.level === "CRITICAL"
                          ? "bg-red-400"
                          : entry.level === "WARNING"
                            ? "bg-yellow-400"
                            : entry.source === "ui"
                              ? "bg-orange-500/70"
                              : "bg-neutral-600"
                      )}
                      aria-hidden
                    />
                    <span className={cn(
                      "break-all",
                      entry.level === "ERROR" || entry.level === "CRITICAL"
                        ? "text-red-300/90"
                        : entry.level === "WARNING"
                          ? "text-yellow-200/80"
                          : entry.source === "ui"
                            ? "text-neutral-200"
                            : "text-neutral-400"
                    )}>
                      {entry.source === "ui" && entry.category && (
                        <span className="text-neutral-600">{uiCategoryLabels[entry.category as UiCategory] ?? entry.category} · </span>
                      )}
                      {entry.message}
                    </span>
                  </div>
                ))
              )}
            </div>
            {hiddenCount > 0 && !showJump && (
              <button
                onClick={() => setRenderLimit((n) => n + 200)}
                className="absolute bottom-2 left-3 z-10 h-6 px-2.5 rounded-lg text-[10px] font-sans border border-neutral-700 bg-neutral-800 text-neutral-300 hover:bg-neutral-700 transition-colors cursor-pointer"
              >
                Show {hiddenCount} older
              </button>
            )}
            {showJump && (
              <button
                onClick={jumpToLatest}
                className="absolute bottom-2 right-3 z-10 h-6 px-2.5 rounded-lg text-[10px] font-sans font-medium border border-orange-400/40 bg-orange-500 hover:bg-orange-400 text-white shadow-[0_0_12px_rgba(184,92,42,0.35)] grain-bg grain-bg-strong transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer select-none flex items-center gap-1"
              >
                <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="12" y1="5" x2="12" y2="19" /><polyline points="19,12 12,19 5,12" />
                </svg>
                Latest
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
