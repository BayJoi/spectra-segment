import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, BASE } from "@/lib/api";
import { cn } from "@/lib/utils";

interface LogEntry {
  level: string;
  message: string;
}

interface ModelStatus {
  name: string;
  display_name: string;
  downloaded: boolean;
  loaded: boolean;
}

const LEVEL_DOT: Record<string, string> = {
  INFO: "bg-neutral-500",
  WARNING: "bg-yellow-400",
  ERROR: "bg-red-400",
  DOWNLOAD: "bg-blue-400",
};

const MAX_LOG_ENTRIES = 500;

const LEVEL_RANK: Record<string, number> = {
  INFO: 0,
  WARNING: 1,
  ERROR: 2,
  DOWNLOAD: -1,
};

type FilterKey = "all" | "info" | "warn" | "error";

const FILTER_OPTIONS: { key: FilterKey; label: string; minRank: number }[] = [
  { key: "all", label: "All", minRank: -Infinity },
  { key: "info", label: "Info+", minRank: 0 },
  { key: "warn", label: "Warn+", minRank: 1 },
  { key: "error", label: "Errors", minRank: 2 },
];

export function ConsolePanel({ encoding }: { encoding?: boolean }) {
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);
  const [copied, setCopied] = useState(false);
  const [filter, setFilter] = useState<FilterKey>("all");
  const [filterOpen, setFilterOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [modelStatuses, setModelStatuses] = useState<ModelStatus[]>([]);
  const [modelsOpen, setModelsOpen] = useState(false);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const evtRef = useRef<EventSource | null>(null);
  const modelEvtRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!open) {
      if (evtRef.current) {
        evtRef.current.close();
        evtRef.current = null;
      }
      return;
    }

    api.getLogs(300).then((res) => {
      setLogs(
        res.entries.map((e: { level: string; message: string }) => ({
          level: e.level,
          message: e.message,
        }))
      );
    }).catch(() => {});

    const evtSource = new EventSource(`${BASE}/api/logs/stream`);
    evtRef.current = evtSource;

    evtSource.onmessage = (ev) => {
      try {
        const entry: LogEntry = JSON.parse(ev.data);
        setLogs((prev) => {
          const next = [...prev, entry];
          return next.length > MAX_LOG_ENTRIES ? next.slice(next.length - MAX_LOG_ENTRIES) : next;
        });
      } catch {}
    };

    evtSource.onerror = () => {
      evtSource.close();
      evtRef.current = null;
    };

    return () => {
      evtSource.close();
      evtRef.current = null;
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      if (modelEvtRef.current) {
        modelEvtRef.current.close();
        modelEvtRef.current = null;
      }
      return;
    }

    api.getModels().then((res) => {
      setModelStatuses(res.models);
    }).catch(() => {});

    const evtSource = new EventSource(`${BASE}/api/models/stream`);
    modelEvtRef.current = evtSource;

    evtSource.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.models) {
          setModelStatuses(data.models);
        }
      } catch {}
    };

    evtSource.onerror = () => {
      evtSource.close();
      modelEvtRef.current = null;
    };

    return () => {
      evtSource.close();
      modelEvtRef.current = null;
      setModelStatuses([]);
    };
  }, [open]);

  useEffect(() => {
    if (isAtBottomRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    isAtBottomRef.current = scrollHeight - scrollTop - clientHeight < 30;
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
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    };
  }, []);

  const filteredLogs = useMemo(() => {
    const minRank = FILTER_OPTIONS.find((f) => f.key === filter)?.minRank ?? -Infinity;
    return logs.filter((e) => {
      const rank = LEVEL_RANK[e.level] ?? 0;
      return rank >= minRank;
    });
  }, [logs, filter]);

  const handleCopy = useCallback(async () => {
    const text = filteredLogs.map((l) => `[${l.level}] ${l.message}`).join("\n");
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
  }, [filteredLogs]);

  return (
    <div className="relative z-[85]">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans transition-all duration-200 cursor-pointer select-none",
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
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-3 w-[380px] bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-xl shadow-2xl shadow-black/60 backdrop-blur-xl overflow-hidden animate-drop-in z-50 grain-bg">
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-neutral-800/60">
            <span className="text-[10px] text-neutral-500 uppercase tracking-wider font-sans">Console</span>
            <div className="flex items-center gap-2">
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
                {copied ? "Copied!" : "Copy logs"}
              </button>
            </div>
          </div>
          {modelStatuses.length > 0 && (
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
                  {modelStatuses.filter((m) => m.downloaded).length}/{modelStatuses.length}
                </span>
              </button>
              {modelsOpen && (
                <div className="px-3 pb-2 space-y-1">
                  {modelStatuses.map((m) => (
                    <div
                      key={m.name}
                      className="flex items-center gap-2 text-[10px] font-mono"
                    >
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
          <div
            ref={scrollRef}
            onScroll={handleScroll}
            className="overflow-y-auto custom-scrollbar px-3 py-2 h-64 font-mono text-[11px] leading-relaxed"
          >
            {filteredLogs.length === 0 ? (
              <p className="text-neutral-600 text-center py-4">No logs yet</p>
            ) : (
              filteredLogs.map((entry, i) => (
                <div key={i} className="flex items-start gap-2 py-[3px] hover:bg-neutral-900/40 rounded px-1 -mx-1">
                  <span className={cn("w-1.5 h-1.5 rounded-full mt-[5px] shrink-0", LEVEL_DOT[entry.level] || "bg-neutral-600")} />
                  <span className="text-neutral-300 break-all">{entry.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
