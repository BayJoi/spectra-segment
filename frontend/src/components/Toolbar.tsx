import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { Button } from "@/components/ui/Button";
import { SelectNative } from "@/components/ui/Select";
import { Tooltip } from "@/components/ui/Tooltip";
import { LayersPanel } from "@/components/LayersPanel";
import { SubjectsPanel } from "@/components/SubjectsPanel";
import { TierIcon } from "@/components/ui/TierIcon";
import { useDetection } from "@/hooks/useDetection";
import { useSession } from "@/hooks/useSession";
import { useLayers } from "@/hooks/useLayers";
import { masksAtom, perDetectionMasksAtom, brushObjectsAtom, activeObjectIdAtom, objectMasksAtom } from "@/store/session";
import { exportOpenAtom, fitToViewAtom, zoomInAtom, zoomOutAtom, modeLockAtom, showTransparentAtom, hideBboxesAtom } from "@/store/ui";
import { api } from "@/lib/api";
import type { PackedMask } from "@/lib/mask";
import { cn } from "@/lib/utils";

export function Toolbar() {
  const { sessionId, undo, redo, canUndo, canRedo, clearObjectHistory } = useSession();
  const {
    detectors,
    selectedDetector,
    setSelectedDetector,
    detectorLoading,
    detectQuery,
    setDetectQuery,
    detections,
    isDetecting,
    yoloeMasksEnabled,
    setYoloeMasksEnabled,
    featherRadius,
    setFeatherRadius,
    detect,
    loadedDetector,
    brushSize,
    setBrushSize,
  } = useDetection();
  const { layers, selectedLayers, clearAllLayers, addLayer } = useLayers();
  const [masks, setMasks] = useAtom(masksAtom);
  const [perDetectionMasks, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [brushObjects, setBrushObjects] = useAtom(brushObjectsAtom);
  const [activeObjectId, setActiveObjectId] = useAtom(activeObjectIdAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setExportOpen] = useAtom(exportOpenAtom);
  const [fitToView] = useAtom(fitToViewAtom);
  const [zoomIn] = useAtom(zoomInAtom);
  const [zoomOut] = useAtom(zoomOutAtom);
  const [modeLock] = useAtom(modeLockAtom);
  const [showTransparent] = useAtom(showTransparentAtom);
  const [hideBboxes, setHideBboxes] = useAtom(hideBboxesAtom);

  const [layersOpen, setLayersOpen] = useState(false);
  const [subjectsOpen, setSubjectsOpen] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const layersToggleRef = useRef<HTMLDivElement>(null);
  const subjectsToggleRef = useRef<HTMLDivElement>(null);
  const queryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (modeLock === "detect") queryInputRef.current?.focus();
  }, [modeLock]);

  const isDetectLocked = modeLock === "detect";
  const isBrushLocked = modeLock === "brush";
  const showLayers = isDetectLocked;
  const hasExportableMasks = Object.keys(perDetectionMasks).length > 0 || masks.length > 0;
  const hasRightGroup = showLayers || hasExportableMasks;

  useEffect(() => {
    if (modeLock !== "detect") setLayersOpen(false);
    if (modeLock !== "brush") setSubjectsOpen(false);
  }, [modeLock]);

  const handleDetect = useCallback(async () => {
    if (!sessionId) return;
    await detect(sessionId);
  }, [sessionId, detect]);

  const handleSegmentAll = useCallback(async () => {
    if (!sessionId || !detections.length) return;
    setIsSegmenting(true);
    try {
      const bboxes = detections.map((d) => d.bbox);
      const res = await api.segmentBatch(sessionId, bboxes);
      if (res?.masks) {
        setPerDetectionMasks((prev) => {
          const next = { ...prev };
          res.masks.forEach((mask: PackedMask, idx: number) => {
            next[idx] = mask;
          });
          return next;
        });
        detections.forEach((det: { label: string; score: number }, idx: number) => {
          if (!layers.some((l) => l.detectionIndex === idx)) {
            addLayer({
              type: "detection",
              label: det.label,
              detectionIndex: idx,
              preview: null,
              objectId: 0,
            });
          }
        });
      }
    } catch (err) {
      console.error("Batch segmentation failed:", err);
    } finally {
      setIsSegmenting(false);
    }
  }, [sessionId, detections, layers, setPerDetectionMasks, addLayer]);

  const activeSubjectIdx = Math.max(0, brushObjects.indexOf(activeObjectId));

  const switchSubject = useCallback(
    (dir: number) => {
      if (brushObjects.length <= 1) return;
      const idx = Math.max(0, brushObjects.indexOf(activeObjectId));
      const nextIdx = (idx + dir + brushObjects.length) % brushObjects.length;
      setActiveObjectId(brushObjects[nextIdx]);
    },
    [brushObjects, activeObjectId, setActiveObjectId]
  );

  const addSubject = useCallback(() => {
    const newId = Math.max(...brushObjects, -1) + 1;
    setBrushObjects((prev) => [...prev, newId]);
    setActiveObjectId(newId);
  }, [brushObjects, setBrushObjects, setActiveObjectId]);

  const deleteSubject = useCallback(() => {
    if (brushObjects.length <= 1) return;
    const idx = Math.max(0, brushObjects.indexOf(activeObjectId));
    const oid = brushObjects[idx];
    const next = brushObjects.filter((x) => x !== oid);
    setBrushObjects(next);
    setActiveObjectId(next[Math.min(idx, next.length - 1)]);
    clearObjectHistory(oid);
  }, [brushObjects, activeObjectId, setBrushObjects, setActiveObjectId, clearObjectHistory]);

  const clearAllBrushObjects = useCallback(() => {
    brushObjects.forEach((oid) => clearObjectHistory(oid));
    setBrushObjects([0]);
    setActiveObjectId(0);
    setObjectMasks({});
    setMasks([]);
  }, [brushObjects, clearObjectHistory, setBrushObjects, setActiveObjectId, setObjectMasks, setMasks]);

  const detectorOptions = useMemo(() => {
    const groups: Record<string, typeof detectors> = {};
    for (const d of detectors) {
      const section = d.type === "yoloe" ? "YOLOE" : d.display_name.includes("Florence") ? "Florence-2" : "Grounding DINO";
      if (!groups[section]) groups[section] = [];
      groups[section].push(d);
    }
    const result: { value: string; label: string; icon?: React.ReactNode; hint?: string; section?: string; disabled?: boolean; loaded?: boolean }[] = [];
    for (const [section, models] of Object.entries(groups)) {
      for (const d of models) {
        result.push({
          value: d.name,
          label: d.downloaded ? `\u2713 ${d.display_name}` : d.display_name,
          icon: <TierIcon tier={d.tier || "medium"} />,
          hint: d.perf,
          section,
          disabled: d.name === selectedDetector,
          loaded: d.name === loadedDetector,
        });
      }
    }
    return result;
  }, [detectors, selectedDetector, loadedDetector]);

  const { isYoloeSelected, isDetectorDownloaded } = useMemo(() => {
    const selectedDetectorInfo = detectors.find((d) => d.name === selectedDetector);
    return {
      isYoloeSelected: selectedDetectorInfo?.type === "yoloe",
      isDetectorDownloaded: selectedDetectorInfo?.downloaded ?? true,
    };
  }, [detectors, selectedDetector]);

  return (
    <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2">
      {isDetectLocked && <LayersPanel open={layersOpen} onClose={() => setLayersOpen(false)} ignoredRef={layersToggleRef} />}
      {isBrushLocked && (
        <SubjectsPanel open={subjectsOpen} onClose={() => setSubjectsOpen(false)} ignoredRef={subjectsToggleRef} />
      )}

      <div className="flex items-center gap-2 px-3 py-2 bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-2xl shadow-2xl shadow-black/80 backdrop-blur-xl grain-bg w-max">
        {modeLock && (
          <>
            <Tooltip tip={isBrushLocked ? "Brush mode — paint strokes to refine masks" : "Detect mode — find objects by text query"} className="shrink-0">
              <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-neutral-900/50 border border-neutral-800/40">
                {isBrushLocked ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-orange-400"><path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08"/><path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z"/></svg>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-green-400"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                )}
                <span className="text-[11px] font-medium font-sans text-neutral-400">
                  {isBrushLocked ? "Brush" : "Detect"}
                </span>
              </div>
            </Tooltip>
            <div className="w-px h-7 bg-neutral-800/60 shrink-0" />
          </>
        )}

        {isDetectLocked ? (
          <div className="flex items-center gap-2 animate-fade-in shrink-0">
            <Tooltip tip="Enter a text query to find objects (e.g. 'person', 'red car', 'eyes')" className="shrink-0">
              <input
                ref={queryInputRef}
                type="text"
                value={detectQuery}
                onChange={(e) => setDetectQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleDetect(); }}
                placeholder="Describe what to find..."
                className="w-44 px-2.5 py-1.5 bg-neutral-900/80 border border-neutral-800/60 rounded-lg text-xs text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-orange-500/50 focus:shadow-[0_0_12px_rgba(184,92,42,0.1)] transition-all duration-200 font-sans"
              />
            </Tooltip>
            <div className="relative shrink-0">
              <SelectNative
                value={selectedDetector}
                onChange={setSelectedDetector}
                options={detectorOptions}
                placeholder="Choose a detector..."
                className={cn("w-[190px]", detectorLoading && "pointer-events-none")}
                disabled={detectorLoading}
              />
              {detectorLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#0d0d0d] rounded-lg backdrop-blur-sm overflow-hidden z-10 pointer-events-none grain-bg">
                  <div className="shimmer-overlay" />
                  <span className="text-[10px] text-orange-400 font-mono animate-pulse relative z-10 whitespace-nowrap">
                    {isDetectorDownloaded ? "Loading..." : "Downloading..."}
                  </span>
                </div>
              )}
            </div>
            {isYoloeSelected && (
              <Tooltip tip={yoloeMasksEnabled ? "YOLOE masks ON — use detector's own masks (faster, skips SAM)" : "YOLOE masks OFF — use SAM for higher quality masks"} className="shrink-0">
                <button
                  onClick={() => setYoloeMasksEnabled(!yoloeMasksEnabled)}
                  className={cn(
                    "h-7 px-2 rounded-lg text-[10px] font-medium font-mono border transition-all duration-200 cursor-pointer whitespace-nowrap hover:scale-105 active:scale-95",
                    yoloeMasksEnabled
                      ? "bg-cyan-500/15 border-cyan-500/40 text-cyan-300 hover:bg-cyan-500/25"
                      : "bg-neutral-900/80 border-neutral-800/60 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700"
                  )}
                >
                  YOLOE
                </button>
              </Tooltip>
            )}
            <Tooltip tip="Run detection on query (Enter)">
              <Button
                variant="accent"
                size="sm"
                onClick={handleDetect}
                disabled={!detectQuery.trim() || isDetecting || detectorLoading}
                className="shrink-0"
              >
                {isDetecting ? (
                  <span className="flex gap-0.5">
                    <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" />
                    <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "150ms" }} />
                    <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "300ms" }} />
                  </span>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="shrink-0"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                )}
                <span>{isDetecting ? "Detecting..." : "Detect"}</span>
              </Button>
            </Tooltip>
            {detections.length > 0 && (
              <>
            <div className="w-px h-7 bg-neutral-800/60 shrink-0" />
                <Tooltip tip="Run SAM on all detected bboxes at once">
                  <Button size="sm" onClick={handleSegmentAll} disabled={isSegmenting} className="shrink-0 bg-green-500/15 border border-green-500/30 text-green-400 hover:bg-green-500/25">
                    {isSegmenting ? (
                      <span className="flex gap-0.5">
                        <span className="w-1 h-1 rounded-full bg-green-400 animate-pulse" />
                        <span className="w-1 h-1 rounded-full bg-green-400 animate-pulse" style={{ animationDelay: "150ms" }} />
                        <span className="w-1 h-1 rounded-full bg-green-400 animate-pulse" style={{ animationDelay: "300ms" }} />
                      </span>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="M12 2a10 10 0 1 0 10 10h-4a6 6 0 1 1-6-6V2z"/></svg>
                    )}
                    <span>{isSegmenting ? "..." : "All"}</span>
                  </Button>
                </Tooltip>
                <span className="inline-flex items-center gap-1 h-7 rounded-lg bg-green-500/10 border border-green-500/25 px-2 text-[10px] font-medium text-green-400 font-mono tabular-nums whitespace-nowrap animate-fade-in shrink-0">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  {detections.length} {detections.length === 1 ? "result" : "results"}
                </span>
                {showTransparent && (
                  <Tooltip tip={hideBboxes ? "Detection boxes hidden in transparent mode — click to show" : "Hide all detection boxes in transparent mode"} className="shrink-0">
                    <button
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
              </>
            )}
          </div>
        ) : isBrushLocked ? (
          <div className="flex items-center gap-2 animate-fade-in shrink-0">
            <Tooltip tip="Undo last stroke (Ctrl+Z)">
              <Button variant="ghost" size="sm" onClick={undo} disabled={!canUndo} aria-label="Undo last stroke" className="shrink-0">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/></svg>
              </Button>
            </Tooltip>
            <Tooltip tip="Redo last stroke (Ctrl+Y)">
              <Button variant="ghost" size="sm" onClick={redo} disabled={!canRedo} aria-label="Redo last stroke" className="shrink-0">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13"/></svg>
              </Button>
            </Tooltip>
            <div className="w-px h-5 bg-neutral-800/60 mx-0.5 shrink-0" />
            <Tooltip tip="Subjects — switch, add, delete" className="shrink-0">
              <div ref={subjectsToggleRef} className="flex items-center gap-0.5 shrink-0">
                <button
                  onClick={() => switchSubject(-1)}
                  disabled={brushObjects.length <= 1}
                  aria-label="Previous subject"
                  className="w-5 h-5 rounded-md flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M15 18l-6-6 6-6"/></svg>
                </button>
                <button
                  onClick={() => setSubjectsOpen(!subjectsOpen)}
                  aria-label="Open subjects menu"
                  className="inline-flex items-center justify-center gap-0.5 text-[10px] text-neutral-400 hover:text-orange-300 font-mono tabular-nums whitespace-nowrap min-w-[36px] py-1 rounded-md hover:bg-orange-500/10 hover:scale-105 active:scale-95 transition-all duration-200 cursor-pointer"
                >
                  Subj {activeSubjectIdx + 1}
                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0"><path d="m6 9 6 6 6-6"/></svg>
                </button>
                <button
                  onClick={() => switchSubject(1)}
                  disabled={brushObjects.length <= 1}
                  aria-label="Next subject"
                  className="w-5 h-5 rounded-md flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M9 18l6-6-6-6"/></svg>
                </button>
                <button
                  onClick={addSubject}
                  aria-label="New subject"
                  className="w-5 h-5 rounded-md flex items-center justify-center text-orange-400 hover:text-orange-300 hover:bg-orange-500/10 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
                </button>
                <button
                  onClick={deleteSubject}
                  disabled={brushObjects.length <= 1}
                  aria-label="Delete active subject"
                  className="w-5 h-5 rounded-md flex items-center justify-center text-neutral-500 hover:text-red-300 hover:bg-red-500/10 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                </button>
              </div>
            </Tooltip>
            <div className="w-px h-5 bg-neutral-800/60 mx-0.5 shrink-0" />
            <Tooltip tip="Brush size — adjust stroke width" className="mr-2">
              <div className="flex items-center gap-1.5 shrink-0">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-500"><circle cx="12" cy="12" r="3"/></svg>
                <input
                  type="range"
                  min={2}
                  max={80}
                  value={brushSize}
                  aria-label="Brush size"
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-20 h-1 cursor-pointer"
                  style={{ "--pct": `${((brushSize - 2) / 78) * 100}%` } as React.CSSProperties}
                />
                <span className="text-[10px] text-neutral-500 font-mono tabular-nums min-w-[20px] text-right">{brushSize}</span>
              </div>
            </Tooltip>
            <div className="w-px h-5 bg-neutral-800/60 mx-0.5 shrink-0" />
            <Tooltip tip="Feather radius — soften mask edges for smoother compositing" className="mr-2">
              <div className="flex items-center gap-1.5 shrink-0">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-500"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/><line x1="16" y1="8" x2="2" y2="22"/></svg>
                <input
                  type="range"
                  min={0}
                  max={20}
                  value={featherRadius}
                  aria-label="Feather radius"
                  onChange={(e) => setFeatherRadius(Number(e.target.value))}
                  className="w-16 h-1 cursor-pointer"
                  style={{ "--pct": `${(featherRadius / 20) * 100}%` } as React.CSSProperties}
                />
                <span className="text-[10px] text-neutral-500 font-mono tabular-nums min-w-[20px] text-right">{featherRadius}</span>
              </div>
            </Tooltip>
            {(masks.length > 0 || Object.keys(perDetectionMasks).length > 0) && (
              <>
            <div className="w-px h-5 bg-neutral-800/60 mx-0.5 shrink-0" />
                <Tooltip tip="Clear all masks">
                  <Button variant="ghost" size="sm" onClick={() => {
                    clearAllBrushObjects();
                    setPerDetectionMasks({});
                  }} aria-label="Clear all masks" className="shrink-0 text-neutral-500 hover:text-red-400">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                  </Button>
                </Tooltip>
              </>
            )}
          </div>
        ) : null}

        {isDetectLocked && (
          <>
            <div className="w-px h-7 bg-neutral-800/60 shrink-0" />
            <Tooltip tip="Feather radius — soften mask edges for smoother compositing" className="mr-2">
              <div className="flex items-center gap-1.5 shrink-0">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-500"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/><line x1="16" y1="8" x2="2" y2="22"/></svg>
                <input
                  type="range"
                  min={0}
                  max={20}
                  value={featherRadius}
                  aria-label="Feather radius"
                  onChange={(e) => setFeatherRadius(Number(e.target.value))}
                  className="w-16 h-1 cursor-pointer"
                  style={{ "--pct": `${(featherRadius / 20) * 100}%` } as React.CSSProperties}
                />
                <span className="text-[10px] text-neutral-500 font-mono tabular-nums min-w-[16px] text-right">{featherRadius}</span>
              </div>
            </Tooltip>
          </>
        )}

        {hasRightGroup && <div className="w-px h-7 bg-neutral-800/60 shrink-0" />}

        <div className="flex items-center gap-2 shrink-0">
          {showLayers && (
            <div ref={layersToggleRef} className="flex items-center gap-0.5 shrink-0">
              <Tooltip tip="Detections panel — view and manage detected objects">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setLayersOpen(!layersOpen)}
                  aria-label="Toggle layers panel"
                  className={cn("shrink-0", selectedLayers.size > 0 && "bg-neutral-800/50 text-neutral-200")}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>
                  <span className="text-[10px] font-mono tabular-nums">{layers.length}</span>
                </Button>
              </Tooltip>
                <Tooltip tip="Clear all detection layers">
                  <Button variant="ghost" size="sm" onClick={() => {
                    clearAllLayers();
                    clearAllBrushObjects();
                  }} aria-label="Clear all detection layers" className="shrink-0 text-neutral-500 hover:text-red-400">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5"><polyline points="3,6 5,6 21,6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </Button>
              </Tooltip>
            </div>
          )}
          {hasExportableMasks && (
            <Tooltip tip="Export masks as a zip with layer folders">
              <Button variant="ghost" size="sm" onClick={() => setExportOpen(true)} className="shrink-0">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Export
              </Button>
            </Tooltip>
          )}
        </div>

        {hasRightGroup && <div className="w-px h-7 bg-neutral-800/60 shrink-0" />}

        <div className="flex items-center gap-0.5">
          <Tooltip tip="Zoom in">
            <button
              onClick={() => zoomIn?.()}
              aria-label="Zoom in"
              className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
          </Tooltip>
          <Tooltip tip="Fit image to view">
            <button
              onClick={() => fitToView?.()}
              aria-label="Fit image to view"
              className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
            </button>
          </Tooltip>
          <Tooltip tip="Zoom out">
            <button
              onClick={() => zoomOut?.()}
              aria-label="Zoom out"
              className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
            </button>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}
