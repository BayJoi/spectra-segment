import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { Button } from "@/components/ui/Button";
import { NumberInput } from "@/components/ui/NumberInput";
import { SelectNative } from "@/components/ui/Select";
import { Tooltip } from "@/components/ui/Tooltip";
import { LayersPanel } from "@/components/LayersPanel";
import { SubjectsPanel } from "@/components/SubjectsPanel";
import { TierIcon } from "@/components/ui/TierIcon";
import { BoxesToggle } from "@/components/ui/BoxesToggle";
import { useDetection } from "@/hooks/useDetection";
import { isSegmentingAtom } from "@/store/detection";
import { useSession } from "@/hooks/useSession";
import { useLayers } from "@/hooks/useLayers";
import {
  masksAtom,
  perDetectionMasksAtom,
  brushObjectsAtom,
  activeObjectIdAtom,
  objectMasksAtom,
  subjectMetaAtom,
  objectHistoryAtom,
  subjectColor,
} from "@/store/session";
import {
  exportOpenAtom,
  fitToViewAtom,
  zoomInAtom,
  zoomOutAtom,
  modeLockAtom,
  showTransparentAtom,
  hideBboxesAtom,
  pushToast,
} from "@/store/ui";
import { api } from "@/lib/api";
import { emitUi, logErr } from "@/store/logs";
import type { PackedMask } from "@/lib/mask";
import { cn } from "@/lib/utils";

const iconButton =
  "w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-neutral-500 transition-colors cursor-pointer";

const divider = <div className="h-6 w-px shrink-0 bg-neutral-800/70" />;

export function Toolbar() {
  const { sessionId, undo, redo, canUndo, canRedo, clearObjectHistory, forgetSubject } = useSession();
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
  } = useDetection();
  const { layers, selectedLayers, clearAllLayers, addLayer, syncDetectionLayers, appendDetectionLayers } = useLayers();
  const [masks, setMasks] = useAtom(masksAtom);
  const [perDetectionMasks, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [brushObjects, setBrushObjects] = useAtom(brushObjectsAtom);
  const [activeObjectId, setActiveObjectId] = useAtom(activeObjectIdAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setSubjectMeta] = useAtom(subjectMetaAtom);
  const [subjectMeta] = useAtom(subjectMetaAtom);
  const [objectHistory] = useAtom(objectHistoryAtom);
  const [, setExportOpen] = useAtom(exportOpenAtom);
  const [fitToView] = useAtom(fitToViewAtom);
  const [zoomIn] = useAtom(zoomInAtom);
  const [zoomOut] = useAtom(zoomOutAtom);
  const [modeLock] = useAtom(modeLockAtom);
  const [showTransparent] = useAtom(showTransparentAtom);
  const [hideBboxes, setHideBboxes] = useAtom(hideBboxesAtom);

  const [layersOpen, setLayersOpen] = useState(false);
  const [subjectsOpen, setSubjectsOpen] = useState(false);
  const [isSegmenting, setIsSegmenting] = useAtom(isSegmentingAtom);
  const layersToggleRef = useRef<HTMLDivElement>(null);
  const subjectsToggleRef = useRef<HTMLDivElement>(null);
  const queryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (modeLock === "detect") queryInputRef.current?.focus();
  }, [modeLock]);

  const isDetectLocked = modeLock === "detect";
  const isBrushLocked = modeLock === "brush";
  const hasExportableMasks = Object.keys(perDetectionMasks).length > 0 || masks.length > 0;

  useEffect(() => {
    if (modeLock !== "detect") setLayersOpen(false);
    if (modeLock !== "brush") setSubjectsOpen(false);
  }, [modeLock]);

  const handleDetect = useCallback(async () => {
    if (!sessionId) return;
    const offset = detections.length;
    const res = await detect(sessionId);
    if (res && res.detections.length) {
      if (offset === 0) {
        syncDetectionLayers(res.detections);
      } else {
        appendDetectionLayers(res.detections, offset);
      }
    }
  }, [sessionId, detect, detections.length, syncDetectionLayers, appendDetectionLayers]);

  const handleSegmentAll = useCallback(async () => {
    if (!sessionId || !detections.length) return;
    setIsSegmenting(true);
    try {
      const bboxes = detections.map((d) => d.bbox);
      const t = Date.now();
      const res = await api.segmentBatch(sessionId, bboxes);
      const made = res?.masks?.filter((m) => m).length ?? 0;
      emitUi(
        "segment",
        "INFO",
        `Segmented ${made} of ${bboxes.length} detection(s) in ${Date.now() - t}ms`
      );
      if (res?.masks) {
        const next: Record<number, PackedMask> = {};
        res.masks.forEach((mask: PackedMask | null, idx: number) => {
          if (mask) next[idx] = mask;
        });
        setPerDetectionMasks(next);
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
      logErr("segment", err);
      pushToast("Batch segmentation failed");
    } finally {
      setIsSegmenting(false);
    }
  }, [sessionId, detections, layers, setPerDetectionMasks, addLayer]);

  const activeMeta = subjectMeta[activeObjectId];
  const activeLabel = activeMeta?.name ?? `Subject ${activeObjectId + 1}`;
  const activeColor = activeMeta?.color ?? subjectColor(activeObjectId);
  const activeStrokes = objectHistory[activeObjectId]?.undo ?? 0;

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
    const newId = Math.max(-1, ...brushObjects) + 1;
    setBrushObjects((prev) => [...prev, newId]);
    setActiveObjectId(newId);
    setSubjectMeta((prev) => ({
      ...prev,
      [newId]: { id: newId, name: `Subject ${newId + 1}`, color: subjectColor(newId) },
    }));
    emitUi("subject", "INFO", "New subject added");
  }, [brushObjects, setBrushObjects, setActiveObjectId, setSubjectMeta]);

  const deleteSubject = useCallback(() => {
    if (brushObjects.length <= 1) {
      emitUi("subject", "WARNING", "Cannot delete - at least one subject must remain");
      return;
    }
    const idx = Math.max(0, brushObjects.indexOf(activeObjectId));
    const oid = brushObjects[idx];
    const next = brushObjects.filter((x) => x !== oid);
    setBrushObjects(next);
    setActiveObjectId(next[Math.min(idx, next.length - 1)]);
    setSubjectMeta((prev) => {
      const nextMeta = { ...prev };
      delete nextMeta[oid];
      return nextMeta;
    });
    forgetSubject(oid);
    clearObjectHistory(oid);
    emitUi("subject", "INFO", `Subject ${oid + 1} deleted`);
  }, [
    brushObjects,
    activeObjectId,
    setBrushObjects,
    setActiveObjectId,
    setSubjectMeta,
    clearObjectHistory,
    forgetSubject,
  ]);

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
    const result: {
      value: string;
      label: string;
      icon?: React.ReactNode;
      hint?: string;
      section?: string;
      disabled?: boolean;
      loaded?: boolean;
    }[] = [];
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

  const modeChip = modeLock ? (
    <div className="flex h-7 items-center gap-1.5 rounded-lg border border-neutral-800/60 bg-neutral-900/60 px-2 shrink-0">
      {isBrushLocked ? (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-orange-400">
          <path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08" />
          <path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z" />
        </svg>
      ) : (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="text-orange-400">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
      )}
      <span className="text-[11px] font-medium font-sans text-neutral-300">
        {isBrushLocked ? "Brush" : "Detect"}
      </span>
    </div>
  ) : null;

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex flex-col items-center gap-2">
      {isDetectLocked && (
        <LayersPanel open={layersOpen} onClose={() => setLayersOpen(false)} ignoredRef={layersToggleRef} />
      )}
      {isBrushLocked && (
        <SubjectsPanel open={subjectsOpen} onClose={() => setSubjectsOpen(false)} ignoredRef={subjectsToggleRef} />
      )}

      <div className="flex flex-col gap-1.5 px-2 py-1.5 bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-2xl shadow-2xl shadow-black/80 backdrop-blur-xl grain-bg">
        {isDetectLocked && (
          <>
            <div className="flex items-center gap-1.5">
              {modeChip}
              {divider}
              <Tooltip tip="Enter a text query to find objects (e.g. 'person', 'red car')" className="shrink-0">
                <input
                  ref={queryInputRef}
                  type="text"
                  value={detectQuery}
                  onChange={(e) => setDetectQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleDetect();
                  }}
                  placeholder="Describe what to find..."
                  className="h-7 w-44 px-2.5 bg-neutral-900/80 border border-neutral-800/60 rounded-lg text-xs text-neutral-200 placeholder-neutral-600 focus:outline-none focus:border-orange-500/50 focus:shadow-[0_0_12px_rgba(184,92,42,0.1)] transition-all duration-200 font-sans"
                />
              </Tooltip>
              <div className="relative shrink-0">
                <SelectNative
                  value={selectedDetector}
                  onChange={setSelectedDetector}
                  options={detectorOptions}
                  placeholder="Choose a detector..."
                  className={cn("w-[184px]", detectorLoading && "pointer-events-none")}
                  disabled={detectorLoading}
                />
                {detectorLoading && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center overflow-hidden rounded-lg bg-[#0d0d0d] backdrop-blur-sm pointer-events-none grain-bg">
                    <div className="shimmer-overlay" />
                    <span className="relative z-10 whitespace-nowrap text-[10px] text-orange-400 font-sans animate-pulse">
                      {isDetectorDownloaded ? "Loading..." : "Downloading..."}
                    </span>
                  </div>
                )}
              </div>
              {isYoloeSelected && (
                <Tooltip
                  tip={
                    yoloeMasksEnabled
                      ? "YOLOE masks ON - use detector masks, skip SAM"
                      : "YOLOE masks OFF - use SAM for higher quality masks"
                  }
                  className="shrink-0"
                >
                  <button
                    onClick={() => setYoloeMasksEnabled(!yoloeMasksEnabled)}
                    className={cn(
                      "h-7 px-2 rounded-lg text-[10px] font-medium font-sans border transition-colors cursor-pointer whitespace-nowrap",
                      yoloeMasksEnabled
                        ? "bg-orange-500/15 border-orange-500/40 text-orange-300"
                        : "bg-neutral-900/80 border-neutral-800/60 text-neutral-500 hover:text-neutral-300"
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
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="shrink-0">
                      <circle cx="11" cy="11" r="8" />
                      <path d="m21 21-4.3-4.3" />
                    </svg>
                  )}
                  <span>{isDetecting ? "Detecting..." : "Detect"}</span>
                </Button>
              </Tooltip>
            </div>

            <div className="flex items-center gap-1.5">
              <Tooltip tip="Soften detection mask edges">
                <div className="flex h-7 items-center gap-1.5 rounded-lg border border-neutral-800/60 bg-neutral-900/50 px-2">
                  <span className="text-[10px] text-neutral-500 font-sans">Feather</span>
                  <input
                    type="range"
                    min={0}
                    max={20}
                    value={featherRadius}
                    aria-label="Feather radius"
                    onChange={(e) => setFeatherRadius(Number(e.target.value))}
                    className="w-14 h-1 cursor-pointer"
                    style={{ "--pct": `${(featherRadius / 20) * 100}%` } as React.CSSProperties}
                  />
                  <NumberInput value={featherRadius} min={0} max={20} onChange={setFeatherRadius} ariaLabel="Feather radius value" />
                </div>
              </Tooltip>

              {detections.length > 0 && (
                <>
                  <span className="inline-flex h-7 items-center gap-1 rounded-lg bg-orange-500/10 border border-orange-500/25 px-2 text-[10px] font-medium text-orange-300 font-sans tabular-nums whitespace-nowrap shrink-0">
                    <span className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
                    {detections.length} {detections.length === 1 ? "result" : "results"}
                  </span>
                  <Tooltip tip="Run SAM on all detected boxes">
                    <Button
                      size="sm"
                      onClick={handleSegmentAll}
                      disabled={isSegmenting}
                      className="shrink-0 bg-orange-500/15 border border-orange-500/30 text-orange-300 hover:bg-orange-500/25"
                    >
                      {isSegmenting ? (
                        <span className="flex gap-0.5">
                          <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" />
                          <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "150ms" }} />
                          <span className="w-1 h-1 rounded-full bg-orange-400 animate-pulse" style={{ animationDelay: "300ms" }} />
                        </span>
                      ) : (
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                          <path d="M12 2a10 10 0 1 0 10 10h-4a6 6 0 1 1-6-6V2z" />
                        </svg>
                      )}
                      <span>{isSegmenting ? "..." : "All"}</span>
                    </Button>
                  </Tooltip>
                  {showTransparent && (
                    <BoxesToggle
                      hidden={hideBboxes}
                      onToggle={() => setHideBboxes(!hideBboxes)}
                      tip={
                        hideBboxes
                          ? "Detection boxes hidden in transparent mode - click to show"
                          : "Hide all detection boxes in transparent mode"
                      }
                    />
                  )}
                </>
              )}

              {divider}

              <div ref={layersToggleRef} className="flex items-center gap-0.5 shrink-0">
                <Tooltip tip={`Detections - ${detections.length} found. Click one to segment it.`}>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setLayersOpen(!layersOpen)}
                    aria-label="Toggle detections panel"
                    title={`${detections.length} detection${detections.length === 1 ? "" : "s"}`}
                    className={cn("shrink-0", selectedLayers.size > 0 && "bg-neutral-800/50 text-neutral-200")}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5">
                      <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" />
                      <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65" />
                      <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65" />
                    </svg>
                    <span className="text-[10px] font-sans tabular-nums">{layers.length}</span>
                  </Button>
                </Tooltip>
                <Tooltip tip="Clear all detection layers">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      clearAllLayers();
                      clearAllBrushObjects();
                    }}
                    aria-label="Clear all detection layers"
                    className="shrink-0 text-neutral-500 hover:text-red-400"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5">
                      <polyline points="3,6 5,6 21,6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </Button>
                </Tooltip>
              </div>

              {hasExportableMasks && (
                <Tooltip tip="Export masks as a zip with layer folders">
                  <Button variant="ghost" size="sm" onClick={() => setExportOpen(true)} className="shrink-0">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7,10 12,15 17,10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Export
                  </Button>
                </Tooltip>
              )}

              {divider}

              <div className="flex items-center gap-0.5">
                <Tooltip tip="Zoom in">
                  <button onClick={() => zoomIn?.()} aria-label="Zoom in" className={iconButton}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="11" y1="8" x2="11" y2="14" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                </Tooltip>
                <Tooltip tip="Fit image to view">
                  <button onClick={() => fitToView?.()} aria-label="Fit image to view" className={iconButton}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                    </svg>
                  </button>
                </Tooltip>
                <Tooltip tip="Zoom out">
                  <button onClick={() => zoomOut?.()} aria-label="Zoom out" className={iconButton}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                </Tooltip>
              </div>
            </div>
          </>
        )}

        {isBrushLocked && (
          <div className="flex items-center gap-1.5">
            {modeChip}
            {divider}
            <div className="flex items-center gap-0.5 shrink-0">
              <Tooltip tip="Undo last stroke (Ctrl+Z)">
                <Button variant="ghost" size="sm" onClick={undo} disabled={!canUndo} aria-label="Undo last stroke" className="shrink-0">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                    <path d="M3 7v6h6" />
                    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13" />
                  </svg>
                </Button>
              </Tooltip>
              <Tooltip tip="Redo last stroke (Ctrl+Y)">
                <Button variant="ghost" size="sm" onClick={redo} disabled={!canRedo} aria-label="Redo last stroke" className="shrink-0">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                    <path d="M21 7v6h-6" />
                    <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13" />
                  </svg>
                </Button>
              </Tooltip>
            </div>

            {divider}

            <div ref={subjectsToggleRef} className="flex items-center gap-0.5 shrink-0">
              <button
                onClick={() => switchSubject(-1)}
                disabled={brushObjects.length <= 1}
                aria-label="Previous subject"
                className="w-5 h-5 rounded-md flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M15 18l-6-6 6-6" />
                </svg>
              </button>
              <button
                onClick={() => setSubjectsOpen(!subjectsOpen)}
                aria-label="Open subjects menu"
                title={`${activeLabel} - ${activeStrokes} stroke${activeStrokes === 1 ? "" : "s"}`}
                className="inline-flex h-7 min-w-[52px] items-center justify-center gap-1 rounded-md px-1 text-[10px] text-neutral-400 hover:text-orange-300 font-sans tabular-nums whitespace-nowrap hover:bg-orange-500/10 transition-colors cursor-pointer"
              >
                <span
                  className="w-2 h-2 rounded-sm flex-shrink-0 ring-1 ring-inset ring-white/10"
                  style={{ backgroundColor: activeColor }}
                  aria-hidden
                />
                {activeLabel.replace("Subject ", "S")}
                <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>
              <button
                onClick={() => switchSubject(1)}
                disabled={brushObjects.length <= 1}
                aria-label="Next subject"
                className="w-5 h-5 rounded-md flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/50 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M9 18l6-6-6-6" />
                </svg>
              </button>
              <button
                onClick={addSubject}
                aria-label="New subject"
                className="w-6 h-6 rounded-md flex items-center justify-center text-orange-400 hover:text-orange-300 hover:bg-orange-500/10 transition-colors cursor-pointer"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M12 5v14" />
                  <path d="M5 12h14" />
                </svg>
              </button>
              <button
                onClick={deleteSubject}
                disabled={brushObjects.length <= 1}
                aria-label="Delete active subject"
                className="w-6 h-6 rounded-md flex items-center justify-center text-neutral-500 hover:text-red-300 hover:bg-red-500/10 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18" />
                  <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                  <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                </svg>
              </button>
            </div>

            {divider}

            <Tooltip tip="Soften mask edges">
              <div className="flex h-7 items-center gap-1.5 rounded-lg border border-neutral-800/60 bg-neutral-900/50 px-2">
                <span className="text-[10px] text-neutral-500 font-sans">Feather</span>
                <input
                  type="range"
                  min={0}
                  max={20}
                  value={featherRadius}
                  aria-label="Feather radius"
                  onChange={(e) => setFeatherRadius(Number(e.target.value))}
                  className="w-14 h-1 cursor-pointer"
                  style={{ "--pct": `${(featherRadius / 20) * 100}%` } as React.CSSProperties}
                />
                <NumberInput value={featherRadius} min={0} max={20} onChange={setFeatherRadius} ariaLabel="Feather radius value" />
              </div>
            </Tooltip>

            {hasExportableMasks && (
              <>
                {divider}
                <Tooltip tip="Clear all masks">
                  <button
                    onClick={() => {
                      clearAllBrushObjects();
                      setPerDetectionMasks({});
                    }}
                    aria-label="Clear all masks"
                    className={cn(iconButton, "text-neutral-500 hover:text-red-400")}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
                      <path d="M3 6h18" />
                      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                    </svg>
                  </button>
                </Tooltip>
                <Tooltip tip="Export masks">
                  <Button variant="ghost" size="sm" onClick={() => setExportOpen(true)} className="shrink-0">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 w-3.5 h-3.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7,10 12,15 17,10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Export
                  </Button>
                </Tooltip>
              </>
            )}

            {divider}

            <div className="flex items-center gap-0.5">
              <Tooltip tip="Zoom in">
                <button onClick={() => zoomIn?.()} aria-label="Zoom in" className={iconButton}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    <line x1="11" y1="8" x2="11" y2="14" />
                    <line x1="8" y1="11" x2="14" y2="11" />
                  </svg>
                </button>
              </Tooltip>
              <Tooltip tip="Fit image to view">
                <button onClick={() => fitToView?.()} aria-label="Fit image to view" className={iconButton}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                  </svg>
                </button>
              </Tooltip>
              <Tooltip tip="Zoom out">
                <button onClick={() => zoomOut?.()} aria-label="Zoom out" className={iconButton}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    <line x1="8" y1="11" x2="14" y2="11" />
                  </svg>
                </button>
              </Tooltip>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
