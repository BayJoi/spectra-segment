import { useCallback, useRef } from "react";

export function usePanZoom(
  zoomRef: React.MutableRefObject<number>,
  panRef: React.MutableRefObject<{ x: number; y: number }>,
  scheduleRedraw: () => void
) {
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0 });

  const fitToView = useCallback(() => {
    zoomRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    scheduleRedraw();
  }, [scheduleRedraw, zoomRef, panRef]);

  const zoomIn = useCallback(() => {
    zoomRef.current = Math.min(50, zoomRef.current * 1.25);
    scheduleRedraw();
  }, [scheduleRedraw, zoomRef]);

  const zoomOut = useCallback(() => {
    zoomRef.current = Math.max(0.1, zoomRef.current * 0.8);
    scheduleRedraw();
  }, [scheduleRedraw, zoomRef]);

  return { fitToView, zoomIn, zoomOut, isPanningRef, panStartRef };
}
