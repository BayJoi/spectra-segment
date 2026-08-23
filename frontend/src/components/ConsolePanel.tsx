import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { api, BASE } from "@/lib/api";
import { connectSse } from "@/lib/sse";
import { cn } from "@/lib/utils";
import {
  consoleOpenAtom,
  consoleFilterAtom,
  consoleShowTimeAtom,
  consoleScrollRatioAtom,
  type ConsoleFilterKey,
} from "@/store/ui";
import { modelsAtom } from "@/store/session";

interface LogEntry {
  level: string;
  message: string;
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

const LEVEL_DOT: Record<string, string> = {
  INFO: "bg-neutral-500",
  WARNING: "bg-yellow-400",
  ERROR: "bg-red-400",
  DOWNLOAD: "bg-orange-400",
};

const MAX_LOG_ENTRIES = 500;

const LEVEL_RANK: Record<string, number> = {
  INFO: 0,
  WARNING: 1,
  ERROR: 2,
  DOWNLOAD: -1,
};

const FILTER_OPTIONS: { key: ConsoleFilterKey; label: string; minRank: number }[] = [
  { key: "all", label: "All", minRank: -Infinity },
  { key: "info", label: "Info+", minRank: 0 },
  { key: "warn", label: "Warn+", minRank: 1 },
  { key: "error", label: "Errors", minRank: 2 },
];

export function ConsolePanel({ encoding }: { encoding?: boolean }) {
  const [open, setOpen] = useAtom(consoleOpenAtom);
  const [filter, setFilter] = useAtom(consoleFilterAtom);
  const [showTime, setShowTime] = useAtom(consoleShowTimeAtom);
  const [savedScrollRatio, setSavedScrollRatio] = useAtom(consoleScrollRatioAtom);
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

  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const hoverRef = useRef(false);
  const lastDownloadTsRef = useRef(0);
  const openRef = useRef(open);
  openRef.current = open;
  const badgeRef = useRef(badge);
  badgeRef.current = badge;
  const readyRef = useRef(false);
  const pendingRef = useRef<LogEntry[]>([]);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const prevOpenRef = useRef(false);

  const appendEntry = useCallback((entry: LogEntry) => {
    if (!readyRef.current) {
      pendingRef.current.push(entry);
      return;
    }
    setLogs((prev) => {
      const next = [...prev, entry];
      return next.length > MAX_LOG_ENTRIES ? next.slice(next.length - MAX_LOG_ENTRIES) : next;
    });
    if (entry.level === "DOWNLOAD") lastDownloadTsRef.current = Date.now();
    if (!openRef.current || !pinnedRef.current || hoverRef.current) {
      if (entry.level === "ERROR") setBadge("error");
      else if (entry.level === "WARNING" && badgeRef.current !== "error") setBadge("warn");
      if (!openRef.current) setUnread((u) => u + 1);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    api.getLogs(300).then((res) => {
      if (cancelled) return;
      const entries = (res.entries as LogEntry[]).map((e) => ({
        level: e.level,
        message: e.message,
        ts: e.ts,
        name: e.name,
        pct: e.pct,
      }));
      setLogs(entries.slice(-MAX_LOG_ENTRIES));
      for (const e of pendingRef.current) appendEntry(e);
      pendingRef.current = [];
      readyRef.current = true;
    }).catch(() => {
      readyRef.current = true;
      for (const e of pendingRef.current) appendEntry(e);
      pendingRef.current = [];
    });

    const handle = connectSse(`${BASE}/api/logs/stream`, (data) => {
      try {
        appendEntry(JSON.parse(data));
      } catch {}
    }, setConnected);

    return () => {
      cancelled = true;
      handle.close();
    };
  }, [appendEntry]);

  useEffect(() => {
    if (!open) return;
    const el = scrollRef.current;
    if (!el) return;
    const forceFollow = Date.now() - lastDownloadTsRef.current < 1500;
    if ((pinnedRef.current && !hoverRef.current) || forceFollow) {
      el.scrollTop = el.scrollHeight;
      pinnedRef.current = true;
      setShowJump(false);
    } else {
      setShowJump(true);
    }
  }, [logs, open]);

  useEffect(() => {
    if (open && !prevOpenRef.current) {
      setUnread(0);
      setBadge(null);
      const el = scrollRef.current;
      if (el) {
        const raf = requestAnimationFrame(() => {
          const max = el.scrollHeight - el.clientHeight;
          if (savedScrollRatio < 0.97 && max > 0) {
            el.scrollTop = savedScrollRatio * max;
            pinnedRef.current = false;
            setShowJump(true);
          } else {
            el.scrollTop = el.scrollHeight;
            pinnedRef.current = true;
            setShowJump(false);
          }
        });
        prevOpenRef.current = true;
        return () => cancelAnimationFrame(raf);
      }
    } else if (!open && prevOpenRef.current) {
      prevOpenRef.current = false;
    }
  }, [open, savedScrollRatio]);

  useEffect(() => {
    if (open) return;
    const el = scrollRef.current;
    if (!el) return;
    const max = el.scrollHeight - el.clientHeight;
    setSavedScrollRatio(max > 0 ? el.scrollTop / max : 1);
  }, [open, setSavedScrollRatio]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const atBottom = scrollHeight - scrollTop - clientHeight < 30;
    pinnedRef.current = atBottom;
    setShowJump(!atBottom);
  }, []);

  useEffect(() => {
    if (!filterOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setFilterOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [filterOpen]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, setOpen]);

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    };
  }, []);

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
      const filesIncomplete = !!l.files_total && (l.files_done ?? 0) < l.files_total;
      if (phase === "done" || ((l.pct ?? 0) >= 100 && !filesIncomplete)) {
        map.delete(l.name);
        continue;
      }
      const prev = map.get(l.name);
      map.set(l.name, {
        pct: l.pct ?? prev?.pct ?? 0,
        message: l.message,
        phase: phase === "starting" && prev?.phase && prev.phase !== "starting" ? prev.phase : phase,
        file: l.file ?? prev?.file,
        files_done: l.files_done ?? prev?.files_done,
        files_total: l.files_total ?? prev?.files_total,
        done_bytes: l.done_bytes ?? prev?.done_bytes,
        total_bytes: l.total_bytes ?? prev?.total_bytes,
        eta: l.eta ?? prev?.eta,
      });
    }
    return [...map.entries()];
  }, [logs, open]);

  const filteredLogs = useMemo(() => {
    const minRank = FILTER_OPTIONS.find((f) => f.key === filter)?.minRank ?? -Infinity;
    const q = search.trim().toLowerCase();
    return logs.filter((e) => {
      if (e.level === "DOWNLOAD") return false;
      const rank = LEVEL_RANK[e.level] ?? 0;
      if (rank < minRank) return false;
      if (q && !e.message.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [logs, filter, search]);

  const jumpToLatest = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    hoverRef.current = false;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setShowJump(false);
  }, []);

  const handleCopy = useCallback(async () => {
    const lines = filteredLogs.map((l) => `[${l.ts ?? "--:--:--"}] [${l.level}] ${l.message}`);
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
          <span className="absolute -top-1.5 -right-1.5 min-w-[14px] h-[14px] px-0.5 flex items-center justify-center rounded-full bg-orange-500 text-[8px] font-bold text-white tabular-nums">
            {unread > 99 ? "99+" : unread}
          </span>
        )}
        {!open && unread === 0 && badge && (
          <span
            className={cn(
              "absolute -top-1 -right-1 w-2 h-2 rounded-full",
              badge === "error" ? "bg-red-500 animate-pulse" : "bg-yellow-400"
            )}
          />
        )}
        {!open && !badge && !connected && (
          <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-neutral-600" />
        )}
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-3 w-[380px] bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-xl shadow-2xl shadow-black/60 backdrop-blur-xl overflow-hidden animate-drop-in z-50 grain-bg">
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-neutral-800/60">
            <span className="text-[10px] text-neutral-500 uppercase tracking-wider font-sans">
              Console
              <span className={cn("inline-block w-1.5 h-1.5 rounded-full ml-1.5 align-middle", connected ? "bg-green-400/70" : "bg-neutral-600")} />
            </span>
            <div className="flex items-center gap-2">
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search..."
                className="w-[90px] h-5 px-1.5 bg-neutral-900/70 border border-neutral-800/60 rounded text-[10px] font-mono text-neutral-300 placeholder:text-neutral-600 outline-none focus:border-neutral-600 transition-colors"
              />
              <button
                onClick={() => setShowTime(!showTime)}
                title="Toggle timestamps"
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
              <div ref={dropdownRef} className="relative">
                <button
                  onClick={() => setFilterOpen(!filterOpen)}
                  className="flex items-center gap-1 text-[10px] font-sans bg-transparent border border-neutral-800/60 rounded px-1.5 py-0.5 text-neutral-400 cursor-pointer outline-none hover:border-neutral-700 hover:scale-105 active:scale-95 transition-all duration-200"
                >
                  {FILTER_OPTIONS.find((f) => f.key === filter)?.label}
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
                  <div className="absolute top-full right-0 mt-1 bg-neutral-900 border border-neutral-700 rounded-lg shadow-xl overflow-hidden z-50 min-w-[90px]">
                    {FILTER_OPTIONS.map((f) => (
                      <button
                        key={f.key}
                        onClick={() => { setFilter(f.key); setFilterOpen(false); }}
                        className={cn(
                          "block w-full text-left text-[11px] font-sans px-3 py-1.5 transition-colors cursor-pointer",
                          filter === f.key
                            ? "text-orange-400 bg-orange-400/10"
                            : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
                        )}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button
                onClick={handleCopy}
                className={cn(
                  "text-[10px] font-mono px-1.5 py-0.5 rounded transition-all duration-150 cursor-pointer hover:scale-105 active:scale-95",
                  copied
                    ? "text-green-400 bg-green-400/10"
                    : "text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800/60"
                )}
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
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
                  {downloadedModels}/{models.length}
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
            >
              {filteredLogs.length === 0 ? (
                <p className="text-neutral-600 text-center py-4">No logs yet</p>
              ) : (
                filteredLogs.map((entry, i) => (
                  <div key={i} className="flex items-start gap-2 py-[3px] hover:bg-neutral-900/40 rounded px-1 -mx-1">
                    {showTime && entry.ts && (
                      <span className="text-neutral-600 tabular-nums shrink-0">{entry.ts}</span>
                    )}
                    <span className={cn("w-1.5 h-1.5 rounded-full mt-[5px] shrink-0", LEVEL_DOT[entry.level] || "bg-neutral-600")} />
                    <span className="text-neutral-300 break-all">{entry.message}</span>
                  </div>
                ))
              )}
            </div>
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
