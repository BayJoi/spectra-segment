import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { Button } from "@/components/ui/Button";
import { NumberInput } from "@/components/ui/NumberInput";
import { SelectNative } from "@/components/ui/Select";
import { Tooltip } from "@/components/ui/Tooltip";
import { TierIcon } from "@/components/ui/TierIcon";
import { MatteLayersPanel } from "@/components/MatteLayersPanel";
import { useMatte } from "@/hooks/useMatte";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { MatteGuide } from "@/lib/matte";
import { emitUi } from "@/store/logs";
import { sessionIdAtom } from "@/store/session";
import { fitToViewAtom, pushToast, zoomInAtom, zoomOutAtom } from "@/store/ui";
import {
  matteBusyAtom,
  matteEdgeFeatherAtom,
  matteEdgeShiftAtom,
  matteExportOpenAtom,
  matteGuideAtom,
  matteHistoryBusyAtom,
  matteInfoAtom,
  matteInvertAtom,
  matteLayersAtom,
  matteModelAtom,
  matteModelsAtom,
} from "@/store/matte";

const GUIDES: { key: MatteGuide; label: string; tip: string }[] = [
  { key: "auto", label: "Auto", tip: "Find the main subject with no prompt" },
  { key: "stroke", label: "Stroke", tip: "Paint the subject to cut it out" },
];

const iconButton =
  "w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-neutral-500 transition-colors cursor-pointer";

const divider = <div className="h-6 w-px shrink-0 bg-neutral-800/70" />;

export function MatteBar() {
  const [sessionId] = useAtom(sessionIdAtom);
  const [models, setModels] = useAtom(matteModelsAtom);
  const [model, setModel] = useAtom(matteModelAtom);
  const [guide, setGuide] = useAtom(matteGuideAtom);
  const [busy] = useAtom(matteBusyAtom);
  const [historyBusy] = useAtom(matteHistoryBusyAtom);
  const [info] = useAtom(matteInfoAtom);
  const [layers] = useAtom(matteLayersAtom);
  const [edgeShift, setEdgeShift] = useAtom(matteEdgeShiftAtom);
  const [edgeFeather, setEdgeFeather] = useAtom(matteEdgeFeatherAtom);
  const [invert, setInvert] = useAtom(matteInvertAtom);
  const [, setExportOpen] = useAtom(matteExportOpenAtom);
  const [fitToView] = useAtom(fitToViewAtom);
  const [zoomIn] = useAtom(zoomInAtom);
  const [zoomOut] = useAtom(zoomOutAtom);
  const {
    run,
    reset,
    undo,
    redo,
    addLayer,
    setLayerModel,
    activeLayer,
    canUndo,
    canRedo,
  } = useMatte();
  const [layersOpen, setLayersOpen] = useState(false);
  const layerButtonRef = useRef<HTMLDivElement>(null);

  const refreshModels = useCallback(() => {
    api
      .listMatteModels()
      .then((res) => setModels(res.models))
      .catch((err) =>
        emitUi(
          "matte",
          "ERROR",
          `Could not refresh matte models: ${err instanceof Error ? err.message : "unknown error"}`
        )
      );
  }, [setModels]);

  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  useEffect(() => {
    if (guide === "text" || guide === "box") setGuide("auto");
  }, [guide, setGuide]);

  const visibleSubjects = layers.filter((layer) => layer.visible).length;
  const visibleCutouts = layers.filter((layer) => layer.visible && layer.rawAlpha).length;
  const readyCutouts = layers.filter((layer) => layer.rawAlpha).length;
  const selectedModel = models.find((item) => item.name === model);

  const options = useMemo(
    () =>
      models.map((item) => ({
        value: item.name,
        label: `${item.downloaded ? "\u2713 " : ""}${item.display_name}`,
        icon: <TierIcon tier={item.tier} />,
        hint: `${item.perf} \u00b7 ${item.weight_mb} MB download`,
        loaded: item.loaded,
      })),
    [models]
  );

  const handleModel = useCallback(
    (name: string) => {
      setModel(name);
      if (!activeLayer) return;
      setLayerModel(activeLayer.id, name);
      if (activeLayer.rawAlpha) {
        void run({ guide, model: name }).then((res) => {
          if (res) refreshModels();
        });
      }
    },
    [setModel, activeLayer, setLayerModel, run, guide, refreshModels]
  );

  const handleGuide = useCallback(
    (next: MatteGuide) => {
      setGuide(next);
      setLayersOpen(false);
    },
    [setGuide]
  );

  const runCurrent = useCallback(async () => {
    if (guide === "stroke" && activeLayer && activeLayer.strokes.points.length === 0) {
      pushToast("Paint the subject first");
      return;
    }
    const res =
      guide === "stroke" && activeLayer
        ? await run({
            guide: "stroke",
            points: activeLayer.strokes.points,
            labels: activeLayer.strokes.labels,
          })
        : await run({ guide: "auto" });
    if (res) refreshModels();
  }, [guide, activeLayer, run, refreshModels]);

  const clearMatte = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.clearMatte(sessionId);
    } catch {
      pushToast("Could not clear the matte");
    }
    reset();
    setLayersOpen(false);
    emitUi("matte", "INFO", "Matte layers cleared");
  }, [sessionId, reset]);

  const handleAddLayer = useCallback(() => {
    addLayer();
    setLayersOpen(true);
  }, [addLayer]);

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2">
      <MatteLayersPanel
        open={layersOpen}
        onClose={() => setLayersOpen(false)}
        ignoredRef={layerButtonRef}
      />

      <div className="flex flex-col gap-1.5 px-2 py-1.5 bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-2xl shadow-2xl shadow-black/80 backdrop-blur-xl grain-bg">
        <div className="flex items-center gap-1.5">
          <Tooltip tip="Matte mode - soft alpha cutout" className="shrink-0">
            <div className="flex h-7 items-center gap-1.5 px-2 rounded-lg bg-orange-500/10 border border-orange-500/25">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-orange-300">
                <path d="M12 3v18" />
                <path d="M4 8.5A8.5 8.5 0 0 1 12 3v18a8.5 8.5 0 0 1-8-8.5Z" />
              </svg>
              <span className="text-[11px] font-medium font-sans text-orange-200">Matte</span>
            </div>
          </Tooltip>

          {divider}

          <SelectNative
            value={model}
            onChange={handleModel}
            options={options}
            placeholder="Choose a matte model..."
            className={cn("w-[178px]", busy && "pointer-events-none")}
            disabled={busy}
          />

          <div className="flex h-7 items-center gap-0.5 p-0.5 rounded-lg border border-neutral-800/60 bg-neutral-900/50 shrink-0">
            {GUIDES.map((item) => (
              <Tooltip key={item.key} tip={item.tip}>
                <button
                  onClick={() => handleGuide(item.key)}
                  className={cn(
                    "inline-flex h-6 items-center gap-1.5 px-2.5 rounded-md text-[11px] font-medium font-sans transition-colors cursor-pointer whitespace-nowrap",
                    guide === item.key
                      ? "bg-orange-500/20 text-orange-100"
                      : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/60"
                  )}
                >
                  {item.key === "auto" ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                      <path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08" />
                      <path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z" />
                    </svg>
                  )}
                  {item.label}
                </button>
              </Tooltip>
            ))}
          </div>

          {divider}

          <div ref={layerButtonRef} className="shrink-0">
            <Tooltip tip="Switch, hide or remove subjects">
              <button
                onClick={() => setLayersOpen((open) => !open)}
                aria-expanded={layersOpen}
                className={cn(
                  "flex h-7 items-center gap-1.5 rounded-lg border pl-1.5 pr-2 text-[11px] font-sans transition-colors cursor-pointer",
                  layersOpen
                    ? "bg-orange-500/15 border-orange-500/30 text-orange-200"
                    : "bg-neutral-900/70 border-neutral-800/60 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700"
                )}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: activeLayer?.color ?? "#d47534" }}
                />
                <span className="max-w-[96px] truncate">{activeLayer?.name ?? "Main subject"}</span>
                <span className="text-neutral-600 tabular-nums">{visibleSubjects}/{layers.length}</span>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className={cn("transition-transform", layersOpen && "rotate-180")}>
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>
            </Tooltip>
          </div>

          <Tooltip tip="Add another subject">
            <button
              onClick={handleAddLayer}
              aria-label="Add matte layer"
              className={cn(iconButton, "border border-neutral-800/60")}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M12 5v14" />
                <path d="M5 12h14" />
              </svg>
            </button>
          </Tooltip>

          <div className="flex items-center gap-0.5 shrink-0">
            <Tooltip tip="Undo the last matte step (Ctrl+Z)">
              <button onClick={() => void undo()} disabled={!canUndo || historyBusy} aria-label="Undo matte step" className={iconButton}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 7v6h6" />
                  <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
                </svg>
              </button>
            </Tooltip>
            <Tooltip tip="Redo the last matte step (Ctrl+Y)">
              <button onClick={() => void redo()} disabled={!canRedo || historyBusy} aria-label="Redo matte step" className={iconButton}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 7v6h-6" />
                  <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" />
                </svg>
              </button>
            </Tooltip>
          </div>

          {divider}

          <Tooltip tip={guide === "stroke" ? "Cut out the painted subject" : "Find and cut out the main subject"}>
            <Button
              variant="accent"
              size="sm"
              onClick={() => void runCurrent()}
              disabled={
                busy || (guide === "stroke" && (activeLayer?.strokes.points.length ?? 0) === 0)
              }
              className="shrink-0"
            >
              {busy ? (
                <span className="flex gap-0.5">
                  <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" />
                  <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "150ms" }} />
                  <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "300ms" }} />
                </span>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="shrink-0">
                  <path d="M5 3v4M3 5h4" />
                  <path d="M19 17v4M17 19h4" />
                  <path d="m9 11 4 4 8-8" />
                </svg>
              )}
              <span>
                {busy
                  ? selectedModel && !selectedModel.downloaded
                    ? "Downloading..."
                    : "Matting..."
                  : activeLayer?.rawAlpha
                    ? "Re-run"
                    : "Cut out"}
              </span>
            </Button>
          </Tooltip>

          {info && (
            <span className="text-[10px] text-neutral-600 font-sans tabular-nums shrink-0">
              {info.ms} ms
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <Tooltip tip="Grow or shrink the matte edge">
            <div className="flex h-7 items-center gap-1.5 rounded-lg border border-neutral-800/60 bg-neutral-900/50 px-2">
              <span className="text-[10px] text-neutral-500 font-sans">Edge</span>
              <input
                type="range"
                min={-50}
                max={50}
                value={edgeShift}
                aria-label="Edge shift"
                onChange={(event) => setEdgeShift(Number(event.target.value))}
                className="w-14 h-1 cursor-pointer"
                style={{ "--pct": `${((edgeShift + 50) / 100) * 100}%` } as React.CSSProperties}
              />
              <NumberInput value={edgeShift} min={-50} max={50} onChange={setEdgeShift} ariaLabel="Edge shift value" />
            </div>
          </Tooltip>

          <Tooltip tip="Soften the matte edge">
            <div className="flex h-7 items-center gap-1.5 rounded-lg border border-neutral-800/60 bg-neutral-900/50 px-2">
              <span className="text-[10px] text-neutral-500 font-sans">Feather</span>
              <input
                type="range"
                min={0}
                max={20}
                value={edgeFeather}
                aria-label="Edge feather"
                onChange={(event) => setEdgeFeather(Number(event.target.value))}
                className="w-14 h-1 cursor-pointer"
                style={{ "--pct": `${(edgeFeather / 20) * 100}%` } as React.CSSProperties}
              />
              <NumberInput value={edgeFeather} min={0} max={20} onChange={setEdgeFeather} ariaLabel="Edge feather value" />
            </div>
          </Tooltip>

          <Tooltip tip="Swap subject and background">
            <button
              onClick={() => setInvert(!invert)}
              aria-label="Invert matte"
              aria-pressed={invert}
              className={cn(
                "h-7 px-2 rounded-lg text-[10px] font-medium font-sans border transition-colors cursor-pointer whitespace-nowrap shrink-0",
                invert
                  ? "bg-orange-500/15 border-orange-500/40 text-orange-200"
                  : "bg-neutral-900/80 border-neutral-800/60 text-neutral-500 hover:text-neutral-300"
              )}
            >
              Invert
            </button>
          </Tooltip>

          {divider}

          <div className="flex items-center gap-0.5 shrink-0">
            <Tooltip tip="Discard every matte layer and start over">
              <button
                onClick={() => void clearMatte()}
                disabled={readyCutouts === 0}
                aria-label="Clear matte"
                className={cn(iconButton, "hover:text-red-400")}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M3 6h18" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                  <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
            </Tooltip>
            <Tooltip tip="Export the cutout">
              <Button variant="ghost" size="sm" onClick={() => setExportOpen(true)} disabled={visibleCutouts === 0} className="shrink-0">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7,10 12,15 17,10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Export
              </Button>
            </Tooltip>
          </div>

          {divider}

          <div className="flex items-center gap-0.5">
            <Tooltip tip="Zoom in">
              <button onClick={() => zoomIn?.()} aria-label="Zoom in" className={iconButton}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  <line x1="11" y1="8" x2="11" y2="14" />
                  <line x1="8" y1="11" x2="14" y2="11" />
                </svg>
              </button>
            </Tooltip>
            <Tooltip tip="Fit image to view">
              <button onClick={() => fitToView?.()} aria-label="Fit image to view" className={iconButton}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                </svg>
              </button>
            </Tooltip>
            <Tooltip tip="Zoom out">
              <button onClick={() => zoomOut?.()} aria-label="Zoom out" className={iconButton}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  <line x1="8" y1="11" x2="14" y2="11" />
                </svg>
              </button>
            </Tooltip>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#0a0a0a]/90 border border-neutral-800/60 backdrop-blur-sm grain-bg">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 text-orange-300/70">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
        <span className="text-[10px] font-sans text-neutral-500 whitespace-nowrap">
          Auto finds the main subject. Add a subject to cut out another.
        </span>
      </div>
    </div>
  );
}
