import { useCallback, useEffect, useRef } from "react";
import type { StrokePoint } from "@/lib/types";

interface BrushDrawingRefs {
  drawCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  imageRef: React.RefObject<HTMLImageElement | null>;
}

interface ComputeStateFn {
  (zoom: number, panX: number, panY: number): {
    scale: number;
    offsetX: number;
    offsetY: number;
  };
}

export function useBrushDrawing(
  refs: BrushDrawingRefs,
  zoomRef: React.MutableRefObject<number>,
  panRef: React.MutableRefObject<{ x: number; y: number }>,
  computeState: ComputeStateFn,
  brushSize: number,
  activeSubjectColor: string = "#f97316"
) {
  const isDrawingRef = useRef(false);
  const strokePointsRef = useRef<StrokePoint[]>([]);
  const activeStrokeModeRef = useRef<"positive" | "negative">("positive");
  const processingRingRef = useRef<{ x: number; y: number; startTime: number } | null>(null);
  const ringAnimFrameRef = useRef<number | null>(null);

  const drawBrushPreview = useCallback(
    (points: StrokePoint[]) => {
      const canvas = refs.drawCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (points.length === 0) return;

      const state = computeState(zoomRef.current, panRef.current.x, panRef.current.y);
      const m = activeStrokeModeRef.current;
      ctx.beginPath();
      ctx.strokeStyle = m === "positive" ? activeSubjectColor : "#ff4444";
      ctx.lineWidth = Math.max(1, brushSize * state.scale);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalAlpha = 0.6;

      for (let i = 0; i < points.length; i++) {
        const px = points[i].x * state.scale + state.offsetX;
        const py = points[i].y * state.scale + state.offsetY;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    },
    [refs.drawCanvasRef, brushSize, computeState, zoomRef, panRef]
  );

  const canvasToImage = useCallback(
    (cx: number, cy: number): StrokePoint => {
      const state = computeState(zoomRef.current, panRef.current.x, panRef.current.y);
      const img = refs.imageRef.current;
      const raw = {
        x: (cx - state.offsetX) / state.scale,
        y: (cy - state.offsetY) / state.scale,
      };
      if (img) {
        raw.x = Math.max(0, Math.min(img.naturalWidth, raw.x));
        raw.y = Math.max(0, Math.min(img.naturalHeight, raw.y));
      }
      return raw;
    },
    [computeState, refs.imageRef, zoomRef, panRef]
  );

  const isInsideImage = useCallback(
    (pos: { x: number; y: number }): boolean => {
      const state = computeState(zoomRef.current, panRef.current.x, panRef.current.y);
      const img = refs.imageRef.current;
      if (!img) return false;
      const imgX = (pos.x - state.offsetX) / state.scale;
      const imgY = (pos.y - state.offsetY) / state.scale;
      return imgX >= 0 && imgX <= img.naturalWidth && imgY >= 0 && imgY <= img.naturalHeight;
    },
    [computeState, refs.imageRef, zoomRef, panRef]
  );

  const startProcessingRing = useCallback((imageX: number, imageY: number) => {
    processingRingRef.current = { x: imageX, y: imageY, startTime: performance.now() };
    const drawRing = () => {
      const ring = processingRingRef.current;
      if (!ring) return;
      const canvas = refs.drawCanvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const state = computeState(zoomRef.current, panRef.current.x, panRef.current.y);
      const px = ring.x * state.scale + state.offsetX;
      const py = ring.y * state.scale + state.offsetY;
      const elapsed = (performance.now() - ring.startTime) / 1000;
      const pulse = Math.sin(elapsed * 6) * 0.4 + 0.6;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.arc(px, py, 14 + Math.sin(elapsed * 8) * 3, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(212, 117, 52, ${0.5 * pulse})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(212, 117, 52, 0.8)`;
      ctx.fill();

      ringAnimFrameRef.current = requestAnimationFrame(drawRing);
    };
    ringAnimFrameRef.current = requestAnimationFrame(drawRing);
  }, [computeState, refs.drawCanvasRef, zoomRef, panRef]);

  const stopProcessingRing = useCallback(() => {
    processingRingRef.current = null;
    if (ringAnimFrameRef.current !== null) {
      cancelAnimationFrame(ringAnimFrameRef.current);
      ringAnimFrameRef.current = null;
    }
    const canvas = refs.drawCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [refs.drawCanvasRef]);

  useEffect(() => {
    return () => {
      if (ringAnimFrameRef.current !== null) {
        cancelAnimationFrame(ringAnimFrameRef.current);
        ringAnimFrameRef.current = null;
      }
    };
  }, []);

  return {
    isDrawingRef,
    strokePointsRef,
    activeStrokeModeRef,
    drawBrushPreview,
    canvasToImage,
    isInsideImage,
    startProcessingRing,
    stopProcessingRing,
  };
}
