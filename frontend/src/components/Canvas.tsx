import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { useSession } from "@/hooks/useSession";
import { useDetection } from "@/hooks/useDetection";
import { useLayers } from "@/hooks/useLayers";
import { useCanvasRenderer } from "@/hooks/useCanvasRenderer";
import { usePanZoom } from "@/hooks/usePanZoom";
import { useBrushDrawing } from "@/hooks/useBrushDrawing";
import { masksAtom, perDetectionMasksAtom, activeObjectIdAtom } from "@/store/session";
import { layersAtom, selectedLayersAtom } from "@/store/layers";
import { sam3InstancesAtom, selectedSam3InstanceAtom } from "@/store/sam3";
import { showTransparentAtom, hideBboxesAtom, fitToViewAtom, zoomInAtom, zoomOutAtom } from "@/store/ui";
import { api, decodeDetectionMask } from "@/lib/api";
import type { PackedMask } from "@/lib/mask";
import { buildStrokePrompt } from "@/lib/strokePrompt";

type CursorMode = "crosshair" | "pointer" | "grab" | "grabbing" | "not-allowed";

const CURSOR_STYLES: Record<CursorMode, string> = {
  crosshair: "crosshair",
  pointer: "pointer",
  grab: "grab",
  grabbing: "grabbing",
  "not-allowed": "not-allowed",
};

interface CanvasProps {
  interactive?: boolean;
}

const STROKE_COLLECT_SPACING = 2;

export function Canvas({ interactive = true }: CanvasProps) {
  const { sessionId, imageUrl, predict } = useSession();
  const { detectMode, toolMode, brushSize, detections, setDetections, selectedDetection, setSelectedDetection, yoloeMasksEnabled, featherRadius } = useDetection();
  const { addLayer } = useLayers();
  const [masks] = useAtom(masksAtom);
  const [activeObjectId] = useAtom(activeObjectIdAtom);
  const activeObjectIdRef = useRef(activeObjectId);
  activeObjectIdRef.current = activeObjectId;
  const [perDetectionMasks, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [layers, setLayers] = useAtom(layersAtom);
  const [, setSelectedLayers] = useAtom(selectedLayersAtom);
  const [showTransparent] = useAtom(showTransparentAtom);
  const [hideBboxes] = useAtom(hideBboxesAtom);
  const [sam3Instances] = useAtom(sam3InstancesAtom);
  const [selectedSam3Instance] = useAtom(selectedSam3InstanceAtom);
  const [, setFitToView] = useAtom(fitToViewAtom);
  const [, setZoomIn] = useAtom(zoomInAtom);
  const [, setZoomOut] = useAtom(zoomOutAtom);

  const containerRef = useRef<HTMLDivElement>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement>(null);
  const drawCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const detectCanvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });

  const rafIdRef = useRef<number | null>(null);
  const dirtyRef = useRef(false);
  const strokeIdRef = useRef(0);

  const detectModeRef = useRef(detectMode);
  detectModeRef.current = detectMode;

  const [cursorMode, setCursorMode] = useState<CursorMode>("crosshair");

  const allMasks = useMemo(
    () => [...Object.values(perDetectionMasks), ...masks],
    [perDetectionMasks, masks]
  );
  const masksRef = useRef(allMasks);
  masksRef.current = allMasks;
  const detectionsRef = useRef(detections);
  detectionsRef.current = detections;
  const selectedDetectionRef = useRef(selectedDetection);
  selectedDetectionRef.current = selectedDetection;
  const showTransparentRef = useRef(showTransparent);
  showTransparentRef.current = showTransparent;
  const hideBboxesRef = useRef(hideBboxes);
  hideBboxesRef.current = hideBboxes;
  const featherRadiusRef = useRef(featherRadius);
  featherRadiusRef.current = featherRadius;
  const sam3InstancesRef = useRef(sam3Instances);
  sam3InstancesRef.current = sam3Instances;
  const selectedSam3InstanceRef = useRef(selectedSam3Instance);
  selectedSam3InstanceRef.current = selectedSam3Instance;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const perDetectionMasksRef = useRef(perDetectionMasks);
  perDetectionMasksRef.current = perDetectionMasks;

  const canvasRefs = useMemo(
      () => ({ containerRef, imageCanvasRef, drawCanvasRef, maskCanvasRef, detectCanvasRef, imageRef }),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      []
  );
  const rendererState = useMemo(
    () => ({ masksRef, detectionsRef, selectedDetectionRef, showTransparentRef, hideBboxesRef, featherRadiusRef, sam3InstancesRef, selectedSam3InstanceRef }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const { computeState, redraw, invalidateCache } = useCanvasRenderer(
    canvasRefs,
    rendererState,
    zoomRef,
    panRef
  );

  const scheduleRedraw = useCallback(() => {
    dirtyRef.current = true;
    if (rafIdRef.current === null) {
      rafIdRef.current = requestAnimationFrame(() => {
        rafIdRef.current = null;
        if (dirtyRef.current) {
          dirtyRef.current = false;
          redraw();
        }
      });
    }
  }, [redraw]);

  const { fitToView, zoomIn, zoomOut, isPanningRef, panStartRef } = usePanZoom(
    zoomRef,
    panRef,
    scheduleRedraw
  );

  const {
    isDrawingRef,
    strokePointsRef,
    activeStrokeModeRef,
    drawBrushPreview,
    canvasToImage,
    isInsideImage,
    startProcessingRing,
    stopProcessingRing,
  } = useBrushDrawing(
    { drawCanvasRef, imageRef },
    zoomRef,
    panRef,
    computeState,
    brushSize
  );

  useEffect(() => {
    setFitToView(() => fitToView);
    setZoomIn(() => zoomIn);
    setZoomOut(() => zoomOut);
    return () => {
      setFitToView(null);
      setZoomIn(null);
      setZoomOut(null);
    };
  }, [fitToView, zoomIn, zoomOut, setFitToView, setZoomIn, setZoomOut]);

  useEffect(() => {
    if (!imageUrl) return;
    let active = true;
    const img = new Image();
    img.onload = () => {
      if (!active) return;
      imageRef.current = img;
      zoomRef.current = 1;
      panRef.current = { x: 0, y: 0 };
      invalidateCache();
      requestAnimationFrame(() => redraw());
    };
    img.src = imageUrl;
    return () => {
      active = false;
    };
  }, [imageUrl, redraw, invalidateCache]);

  useEffect(() => {
    if (!imageRef.current) return;
    invalidateCache();
    requestAnimationFrame(() => redraw());
  }, [masks, perDetectionMasks, detections, selectedDetection, showTransparent, hideBboxes, featherRadius, sam3Instances, selectedSam3Instance, redraw, invalidateCache]);

  useEffect(() => {
    const removeDetectionAt = (idx: number) => {
      setPerDetectionMasks((prev) => {
        const next: Record<number, PackedMask> = {};
        for (const [k, v] of Object.entries(prev)) {
          const ki = Number(k);
          if (ki < idx) next[ki] = v;
          else if (ki > idx) next[ki - 1] = v;
        }
        return next;
      });
      setDetections((prev) => prev.filter((_, i) => i !== idx));
      setSelectedDetection((prev) => {
        if (prev === null) return null;
        if (prev > idx) return prev - 1;
        return null;
      });
      setLayers((prev) => prev
        .filter((l) => l.detectionIndex !== idx)
        .map((l) => l.detectionIndex !== undefined && l.detectionIndex > idx
          ? { ...l, detectionIndex: l.detectionIndex - 1 }
          : l
        )
      );
      setSelectedLayers(new Set());
    };
    const handler = (e: KeyboardEvent) => {
      if (!interactive) return;
      if (e.key === "Delete" || e.key === "Backspace") {
        if (e.repeat) return;
        const target = e.target as HTMLElement;
        if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return;
        e.preventDefault();
        const cur = selectedDetectionRef.current;
        if (cur !== null) {
          removeDetectionAt(cur);
          return;
        }
        if (detectMode && detectionsRef.current.length > 0) {
          removeDetectionAt(detectionsRef.current.length - 1);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setPerDetectionMasks, setDetections, setSelectedDetection, setLayers, setSelectedLayers, interactive, detectMode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const obs = new ResizeObserver(() => scheduleRedraw());
    obs.observe(container);
    return () => obs.disconnect();
  }, [scheduleRedraw]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleWheel = (e: WheelEvent) => {
      if (!imageRef.current) return;
      e.preventDefault();

      const img = imageRef.current;
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const iw = img.naturalWidth;
      const ih = img.naturalHeight;
      const baseScale = Math.min(cw / iw, ch / ih, 1);

      const oldZoom = zoomRef.current;
      const oldScale = baseScale * oldZoom;
      const oldCenterX = Math.floor((cw - Math.floor(iw * oldScale)) / 2);
      const oldCenterY = Math.floor((ch - Math.floor(ih * oldScale)) / 2);
      const oldOffsetX = oldCenterX + panRef.current.x;
      const oldOffsetY = oldCenterY + panRef.current.y;

      const rect = container.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      const imgX = (cx - oldOffsetX) / oldScale;
      const imgY = (cy - oldOffsetY) / oldScale;

      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.1, Math.min(50, oldZoom * delta));
      const newScale = baseScale * newZoom;

      const newCenterX = Math.floor((cw - Math.floor(iw * newScale)) / 2);
      const newCenterY = Math.floor((ch - Math.floor(ih * newScale)) / 2);
      panRef.current = {
        x: Math.round(cx - imgX * newScale - newCenterX),
        y: Math.round(cy - imgY * newScale - newCenterY),
      };

      zoomRef.current = newZoom;
      scheduleRedraw();
    };

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, [scheduleRedraw]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handlePanStart = (e: MouseEvent) => {
      if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        e.preventDefault();
        if (isDrawingRef.current) {
          isDrawingRef.current = false;
          strokePointsRef.current = [];
          const canvas = drawCanvasRef.current;
          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        }
        isPanningRef.current = true;
        panStartRef.current = { x: e.clientX - panRef.current.x, y: e.clientY - panRef.current.y };
        setCursorMode("grabbing");
      }
    };

    const handlePanMove = (e: MouseEvent) => {
      if (!isPanningRef.current) return;
      panRef.current = {
        x: e.clientX - panStartRef.current.x,
        y: e.clientY - panStartRef.current.y,
      };
      scheduleRedraw();
    };

    const handlePanEnd = () => {
      if (!isPanningRef.current) return;
      isPanningRef.current = false;
      setCursorMode(detectModeRef.current ? "pointer" : interactive ? "crosshair" : "grab");
    };

    container.addEventListener("mousedown", handlePanStart);
    window.addEventListener("mousemove", handlePanMove);
    window.addEventListener("mouseup", handlePanEnd);
    return () => {
      container.removeEventListener("mousedown", handlePanStart);
      window.removeEventListener("mousemove", handlePanMove);
      window.removeEventListener("mouseup", handlePanEnd);
    };
  }, [scheduleRedraw, isPanningRef, panStartRef, isDrawingRef, strokePointsRef, drawCanvasRef, interactive]);

  const getPointerPos = useCallback(
    (e: React.PointerEvent | React.MouseEvent): { x: number; y: number } => {
      const rect = containerRef.current!.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    },
    []
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      const active = document.activeElement as HTMLElement | null;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) active.blur();
      if (!interactive || !imageRef.current || isPanningRef.current || detectMode) return;
      if (e.button === 1 || e.shiftKey) return;
      const pos = getPointerPos(e);
      if (!isInsideImage(pos)) {
        setCursorMode("not-allowed");
        return;
      }
      e.preventDefault();
      isDrawingRef.current = true;
      strokeIdRef.current += 1;
      activeStrokeModeRef.current = e.button === 2 ? "negative" : toolMode;
      const imgPt = canvasToImage(pos.x, pos.y);
      strokePointsRef.current = [imgPt];
      drawBrushPreview([imgPt]);
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [canvasToImage, getPointerPos, drawBrushPreview, toolMode, detectMode, isInsideImage, isPanningRef, isDrawingRef, strokePointsRef, interactive]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 2 && !isDrawingRef.current && !isPanningRef.current && imageRef.current && !detectMode && interactive) {
        const pos = getPointerPos(e);
        if (!isInsideImage(pos)) return;
        e.preventDefault();
        isDrawingRef.current = true;
        strokeIdRef.current += 1;
        activeStrokeModeRef.current = "negative";
        const imgPt = canvasToImage(pos.x, pos.y);
        strokePointsRef.current = [imgPt];
        drawBrushPreview([imgPt]);
      }
    },
    [canvasToImage, getPointerPos, drawBrushPreview, detectMode, isInsideImage, isDrawingRef, isPanningRef]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!isDrawingRef.current && !isPanningRef.current && imageRef.current) {
        const pos = getPointerPos(e);
        if (!interactive) {
          setCursorMode(isInsideImage(pos) ? "grab" : "not-allowed");
        } else if (detectMode) {
          setCursorMode("pointer");
        } else if (isInsideImage(pos)) {
          setCursorMode("crosshair");
        } else {
          setCursorMode("not-allowed");
        }
        return;
      }
      if (!isDrawingRef.current) return;
      const pos = getPointerPos(e);
      if (!isInsideImage(pos)) return;
      const imgPt = canvasToImage(pos.x, pos.y);

      const prev = strokePointsRef.current[strokePointsRef.current.length - 1];
      const dist = Math.hypot(imgPt.x - prev.x, imgPt.y - prev.y);
      if (dist >= STROKE_COLLECT_SPACING) {
        strokePointsRef.current.push(imgPt);
      }
      drawBrushPreview(strokePointsRef.current);
    },
    [canvasToImage, getPointerPos, drawBrushPreview, isInsideImage, detectMode, isDrawingRef, isPanningRef, strokePointsRef, interactive]
  );

  const handlePointerUp = useCallback(async () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    const strokeId = strokeIdRef.current;

    const points = strokePointsRef.current;
    if (points.length === 0) return;

    const img = imageRef.current;
    const imgW = img ? img.naturalWidth : 0;
    const imgH = img ? img.naturalHeight : 0;

    const prompt = buildStrokePrompt(points, brushSize, activeStrokeModeRef.current, imgW, imgH);
    const sampled = prompt.points;
    const labels = prompt.labels;
    if (sampled.length === 0) return;

    const canvas = drawCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    strokePointsRef.current = [];

    const cx = sampled.reduce((s, p) => s + p[0], 0) / sampled.length;
    const cy = sampled.reduce((s, p) => s + p[1], 0) / sampled.length;

    const showRingTimer = setTimeout(() => startProcessingRing(cx, cy), 150);

    const res = await predict({ object_id: activeObjectIdRef.current, points: sampled, labels });
    clearTimeout(showRingTimer);
    if (strokeIdRef.current === strokeId) stopProcessingRing();
    if (!res) return;
  }, [predict, brushSize, startProcessingRing, stopProcessingRing, isDrawingRef, strokePointsRef, activeStrokeModeRef, drawCanvasRef, imageRef]);

  const handleDetectClick = useCallback(
    async (e: React.MouseEvent) => {
      const active = document.activeElement as HTMLElement | null;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) active.blur();
      if (!interactive || !detectMode || !detections.length) return;

      const pos = getPointerPos(e);
      const canvasState = computeState(zoomRef.current, panRef.current.x, panRef.current.y);

      for (let i = detections.length - 1; i >= 0; i--) {
        const [x1, y1, x2, y2] = detections[i].bbox;
        const bx = x1 * canvasState.scale + canvasState.offsetX;
        const by = y1 * canvasState.scale + canvasState.offsetY;
        const bw = (x2 - x1) * canvasState.scale;
        const bh = (y2 - y1) * canvasState.scale;

        if (pos.x >= bx && pos.x <= bx + bw && pos.y >= by && pos.y <= by + bh) {
          const hasMask = i in perDetectionMasksRef.current;
          if (selectedDetection === i) {
            setSelectedDetection(null);
            setSelectedLayers(new Set());
            return;
          }
          setSelectedDetection(i);
          setSelectedLayers((prev) => {
            const layerId = layers.find((l) => l.detectionIndex === i)?.id;
            if (!layerId) return prev;
            const next = new Set(prev);
            next.clear();
            next.add(layerId);
            return next;
          });

          if (hasMask) return;

          const det = detections[i];

          if (yoloeMasksEnabled && det.mask) {
            try {
              const decoded = await decodeDetectionMask(det.mask);
              setPerDetectionMasks((prev) => ({ ...prev, [i]: decoded }));
              setDetections((prev) => prev.map((d, di) => (di === i && d.mask ? { ...d, mask: null } : d)));
            } catch (err) {
              console.error("Failed to decode detection mask, falling back to SAM:", err);
              const res = await api.segmentBatch(sessionIdRef.current!, [[...det.bbox]]);
              if (res?.masks?.length) {
                setPerDetectionMasks((prev) => ({ ...prev, [i]: res.masks[0] }));
              }
            }
          } else {
            const res = await api.segmentBatch(sessionIdRef.current!, [[...det.bbox]]);
            if (res?.masks?.length) {
              setPerDetectionMasks((prev) => ({ ...prev, [i]: res.masks[0] }));
            }
          }

          if (!layers.some((l) => l.detectionIndex === i)) {
            addLayer({
              type: "detection",
              label: det.label,
              detectionIndex: i,
              preview: null,
              objectId: 0,
            });
          }
          return;
        }
      }
      setSelectedDetection(null);
    },
    [detectMode, detections, selectedDetection, yoloeMasksEnabled, getPointerPos, computeState, setSelectedDetection, setPerDetectionMasks, addLayer, setSelectedLayers, layers, interactive]
  );

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 overflow-hidden bg-[#0a0a0a]"
      onContextMenu={(e) => e.preventDefault()}
    >
      <canvas ref={imageCanvasRef} className="absolute inset-0 w-full h-full max-w-none" />
      <canvas ref={maskCanvasRef} className="absolute inset-0 w-full h-full max-w-none pointer-events-none" />
      <canvas ref={detectCanvasRef} className="absolute inset-0 w-full h-full max-w-none pointer-events-none" />
      <canvas
        ref={drawCanvasRef}
        className="absolute inset-0 w-full h-full max-w-none"
        style={{ cursor: CURSOR_STYLES[cursorMode] }}
        onMouseDown={handleMouseDown}
        onClick={handleDetectClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      />
    </div>
  );
}
