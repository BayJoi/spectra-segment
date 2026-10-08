import { useCallback, useEffect, useRef, useState } from "react";
import { useAtom } from "jotai";
import { useSam3 } from "@/hooks/useSam3";
import { sam3PromptInputAtom, sam3InstancesAtom, sam3KeepLoadedAtom, sam3EncodeDimAtom, type Sam3EncodeDim } from "@/store/sam3";
import { fitToViewAtom, zoomInAtom, zoomOutAtom, showTransparentAtom, hideBboxesAtom, exportOpenAtom } from "@/store/ui";
import { LayersPanel } from "@/components/LayersPanel";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Tooltip } from "@/components/ui/Tooltip";

const ENCODE_OPTIONS: { value: Sam3EncodeDim; label: string }[] = [
  { value: 512, label: "Fast" },
  { value: 1024, label: "Balanced" },
  { value: 1500, label: "High" },
];

export function Sam3PromptBar() {
  const { prompt, undo, redo, canUndo, canRedo, prompting, removeInstance, error, setError } = useSam3();
  const [input, setInput] = useAtom(sam3PromptInputAtom);
  const [sam3Instances] = useAtom(sam3InstancesAtom);
  const [keepLoaded, setKeepLoaded] = useAtom(sam3KeepLoadedAtom);
  const [encodeDim, setEncodeDim] = useAtom(sam3EncodeDimAtom);
  const [showTransparent] = useAtom(showTransparentAtom);
  const [hideBboxes, setHideBboxes] = useAtom(hideBboxesAtom);
  const [fitToView] = useAtom(fitToViewAtom);
  const [zoomIn] = useAtom(zoomInAtom);
  const [zoomOut] = useAtom(zoomOutAtom);
  const [, setExportOpen] = useAtom(exportOpenAtom);
  const layersToggleRef = useRef<HTMLDivElement>(null);
  const optionsRef = useRef<HTMLDivElement>(null);
  const [layersOpen, setLayersOpen] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const [keepPending, setKeepPending] = useState(false);
  const [reencoding, setReencoding] = useState(false);

  useEffect(() => {
    if (!optionsOpen) return;
    const handler = (e: MouseEvent) => {
      if (optionsRef.current && !optionsRef.current.contains(e.target as Node)) {
        setOptionsOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [optionsOpen]);

  const handleSubmit = useCallback(() => {
    if (prompting || !input.trim()) return;
    const text = input;
    setInput("");
    prompt(text);
  }, [prompt, input, setInput, prompting]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const applySettings = useCallback(
    async (partial: { keep_loaded?: boolean; encode_dim?: number }) => {
      const reencodes = partial.encode_dim !== undefined;
      const setBusy = reencodes ? setReencoding : setKeepPending;
      setBusy(true);
      try {
        return await api.setSam3Settings(partial);
      } catch (err) {
        console.error("Failed to update SAM3 settings:", err);
        return null;
      } finally {
        setBusy(false);
      }
    },
    []
  );

  const toggleKeepLoaded = useCallback(async () => {
    const result = await applySettings({ keep_loaded: !keepLoaded });
    if (result) setKeepLoaded(result.keep_loaded);
  }, [keepLoaded, applySettings, setKeepLoaded]);

  const changeEncodeDim = useCallback(
    async (dim: Sam3EncodeDim) => {
      if (dim === encodeDim) return;
      const result = await applySettings({ encode_dim: dim });
      if (result) setEncodeDim(result.encode_dim as Sam3EncodeDim);
    },
    [encodeDim, applySettings, setEncodeDim]
  );

  const iconBtnCls =
    "inline-flex items-center justify-center w-7 h-7 rounded-lg text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100";

  return (
    <div className="absolute bottom-5 inset-x-0 z-20 pointer-events-none flex flex-col items-center gap-2">
      {layersOpen && (
        <div className="pointer-events-auto animate-drop-in">
          <LayersPanel
            open={layersOpen}
            onClose={() => setLayersOpen(false)}
            ignoredRef={layersToggleRef}
            onRemoveSam3Instance={removeInstance}
          />
        </div>
      )}

      {error && (
        <div className="pointer-events-auto flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-950/40 border border-red-900/50 text-red-300 text-[11px] font-sans shadow-lg shadow-black/40">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
            <circle cx="12" cy="12" r="10" /><path d="M12 8v4" /><path d="M12 16h.01" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      <div className="pointer-events-auto flex items-center gap-2 px-3 py-2 bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-2xl shadow-2xl shadow-black/80 backdrop-blur-xl grain-bg w-max max-w-[calc(100vw-2rem)]">
        <Tooltip tip="SAM3 mode — segment objects by text prompt" className="shrink-0">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-neutral-900/50 border border-neutral-800/40">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-orange-400"><path d="M12 2a10 10 0 1 0 10 10h-4a6 6 0 1 1-6-6V2z"/><path d="m9 12 2 2 4-4"/></svg>
            <span className="text-[11px] font-medium font-sans text-orange-400">SAM3</span>
          </div>
        </Tooltip>

        <div className="w-px h-7 bg-neutral-800/60 shrink-0" />

        <Tooltip tip="Enter a text prompt to segment every matching object (Enter)" className="shrink-0">
          <input
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              if (error) setError(null);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Describe what to segment…"
            className="w-56 sm:w-72 px-2.5 py-1.5 bg-neutral-900/80 border border-neutral-800/60 rounded-lg text-xs text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-orange-500/50 focus:shadow-[0_0_12px_rgba(184,92,42,0.1)] transition-all duration-200 font-sans"
          />
        </Tooltip>
        <button
          onClick={handleSubmit}
          disabled={prompting || !input.trim()}
          className="inline-flex items-center justify-center gap-1.5 h-7 px-3.5 rounded-lg text-[11px] font-semibold font-sans transition-all duration-200 cursor-pointer select-none bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 grain-bg grain-bg-strong"
        >
          {prompting ? (
            <>
              <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
              Segmenting…
            </>
          ) : (
            "Segment"
          )}
        </button>

        <div className="w-px h-7 bg-neutral-800/60 shrink-0" />

        <div className="flex items-center gap-0.5 shrink-0">
          <Tooltip tip="Undo (Ctrl+Z)">
            <button
              type="button"
              onClick={undo}
              disabled={!canUndo || prompting}
              className={iconBtnCls}
              aria-label="Undo"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 7v6h6" />
                <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
              </svg>
            </button>
          </Tooltip>
          <Tooltip tip="Redo (Ctrl+Y)">
            <button
              type="button"
              onClick={redo}
              disabled={!canRedo || prompting}
              className={iconBtnCls}
              aria-label="Redo"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 7v6h-6" />
                <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" />
              </svg>
            </button>
          </Tooltip>
        </div>

        <div ref={layersToggleRef} className="flex items-center gap-0.5 shrink-0">
          <Tooltip tip="SAM3 segments panel — view and remove individual segments">
            <button
              type="button"
              onClick={() => setLayersOpen(!layersOpen)}
              aria-label="Toggle segments panel"
              className={cn(iconBtnCls, layersOpen && "bg-neutral-800/50 text-neutral-200")}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
              <span className="text-[10px] font-mono tabular-nums">{sam3Instances.length}</span>
            </button>
          </Tooltip>
        </div>

        <Tooltip tip={sam3Instances.length > 0 ? "Export segment masks as a zip" : "Segment something first"}>
          <button
            type="button"
            onClick={() => setExportOpen(true)}
            disabled={sam3Instances.length === 0}
            aria-label="Export segments"
            className={cn(
              "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans border transition-all duration-200 select-none shrink-0",
              sam3Instances.length === 0
                ? "bg-neutral-900/50 border-neutral-800/50 text-neutral-600 cursor-not-allowed"
                : "bg-neutral-900/50 border-neutral-800/50 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700 hover:scale-105 active:scale-95 cursor-pointer"
            )}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Export
          </button>
        </Tooltip>

        {showTransparent && sam3Instances.length > 0 && (
          <Tooltip tip={hideBboxes ? "Segment boxes hidden in transparent mode — click to show" : "Hide all segment boxes in transparent mode"} className="shrink-0">
            <button
              type="button"
              onClick={() => setHideBboxes(!hideBboxes)}
              className={cn(
                "h-7 px-2 rounded-lg text-[10px] font-medium font-mono border transition-all duration-200 cursor-pointer whitespace-nowrap hover:scale-105 active:scale-95",
                hideBboxes
                  ? "bg-orange-500/15 border-orange-500/40 text-orange-300 hover:bg-orange-500/25"
                  : "bg-neutral-900/80 border-neutral-800/60 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700"
              )}
            >
              {hideBboxes ? "Boxes off" : "Boxes"}
            </button>
          </Tooltip>
        )}

        <div className="relative shrink-0" ref={optionsRef}>
          <Tooltip tip="Performance & encoding options">
            <button
              type="button"
              onClick={() => setOptionsOpen(!optionsOpen)}
              aria-label="Performance and encoding options"
              className={cn(iconBtnCls, optionsOpen && "bg-neutral-800/50 text-neutral-200")}
            >
              {reencoding ? (
                <svg className="animate-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
              )}
            </button>
          </Tooltip>

          {optionsOpen && (
            <div className="absolute bottom-full mb-2 right-0 z-50 w-72 p-3 rounded-2xl bg-[#0a0a0a]/95 border border-neutral-800/80 shadow-2xl shadow-black/80 backdrop-blur-xl grain-bg animate-drop-in">
              <p className="text-[10px] uppercase tracking-widest text-neutral-500 font-mono mb-2">Performance</p>
              <button
                type="button"
                role="switch"
                aria-checked={keepLoaded}
                onClick={toggleKeepLoaded}
                disabled={keepPending}
                className={cn(
                  "w-full flex items-center justify-between gap-2 h-8 px-2.5 rounded-lg text-[11px] font-medium font-sans border transition-all duration-200 cursor-pointer select-none",
                  keepLoaded
                    ? "bg-orange-500/15 border-orange-500/40 text-orange-300 hover:bg-orange-500/25"
                    : "bg-neutral-900/80 border-neutral-700/80 text-neutral-300 hover:border-neutral-500 hover:text-neutral-100"
                )}
              >
                <span>Keep model loaded</span>
                <span
                  className={cn(
                    "relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors duration-200",
                    keepLoaded ? "bg-orange-500" : "bg-neutral-700"
                  )}
                >
                  <span
                    className={cn(
                      "inline-block h-3 w-3 rounded-full bg-white shadow transition-transform duration-200",
                      keepLoaded ? "translate-x-3.5" : "translate-x-0.5"
                    )}
                  />
                </span>
              </button>
              <p className="text-[9px] text-neutral-600 mt-1.5 leading-relaxed">
                Keeps the model in memory between prompts, avoiding the slow reload that happens after ~25s of idle time.
              </p>

              <p className="text-[10px] uppercase tracking-widest text-neutral-500 font-mono mt-3 mb-2">Encoding quality</p>
              <div className="flex items-center gap-1 p-1 rounded-lg bg-neutral-900/50 border border-neutral-800/60">
                {ENCODE_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => changeEncodeDim(opt.value)}
                    disabled={reencoding}
                    className={cn(
                      "flex-1 h-7 px-2 rounded-md text-[10px] font-medium font-mono transition-all duration-200 cursor-pointer select-none",
                      encodeDim === opt.value
                        ? "bg-gradient-to-r from-orange-500 to-orange-600 text-white grain-bg grain-bg-strong"
                        : "text-neutral-500 hover:text-neutral-200"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="text-[9px] text-neutral-600 mt-1.5 leading-relaxed">
                Higher = more accurate masks but a slower image encode. Changing the quality re-encodes the image.
              </p>

              {reencoding && (
                <div className="flex items-center gap-1.5 mt-2.5 px-2.5 py-1.5 rounded-lg bg-neutral-900/60 border border-orange-500/25 text-orange-300/90">
                  <svg className="animate-spin shrink-0" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                  <span className="text-[9px] font-mono">
                    Re-encoding image… this can take up to a minute
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="w-px h-7 bg-neutral-800/60 shrink-0" />

        <div className="flex items-center gap-0.5">
          <Tooltip tip="Zoom in">
            <button onClick={() => zoomIn?.()} aria-label="Zoom in" className={iconBtnCls}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
          </Tooltip>
          <Tooltip tip="Fit image to view">
            <button onClick={() => fitToView?.()} aria-label="Fit image to view" className={iconBtnCls}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
            </button>
          </Tooltip>
          <Tooltip tip="Zoom out">
            <button onClick={() => zoomOut?.()} aria-label="Zoom out" className={iconBtnCls}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
