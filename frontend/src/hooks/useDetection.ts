import type { PackedMask } from "@/lib/mask";
import { useCallback } from "react";
import { useAtom } from "jotai";
import {
  detectModeAtom,
  toolModeAtom,
  brushSizeAtom,
  featherRadiusAtom,
  detectorsAtom,
  selectedDetectorAtom,
  loadedDetectorAtom,
  detectorLoadingAtom,
  detectQueryAtom,
  detectionsAtom,
  selectedDetectionAtom,
  isDetectingAtom,
  yoloeMasksEnabledAtom,
} from "@/store/detection";
import { api } from "@/lib/api";
import { emitUi, logErr } from "@/store/logs";
import { perDetectionMasksAtom } from "@/store/session";
import { pushToast } from "@/store/ui";

export function useDetection() {
  const [detectMode] = useAtom(detectModeAtom);
  const [toolMode] = useAtom(toolModeAtom);
  const [brushSize, setBrushSize] = useAtom(brushSizeAtom);
  const [detectors] = useAtom(detectorsAtom);
  const [selectedDetector, setSelectedDetector] = useAtom(selectedDetectorAtom);
  const [loadedDetector, setLoadedDetector] = useAtom(loadedDetectorAtom);
  const [detectorLoading, setDetectorLoading] = useAtom(detectorLoadingAtom);
  const [detectQuery, setDetectQuery] = useAtom(detectQueryAtom);
  const [detections, setDetections] = useAtom(detectionsAtom);
  const [selectedDetection, setSelectedDetection] = useAtom(selectedDetectionAtom);
  const [isDetecting, setIsDetecting] = useAtom(isDetectingAtom);
  const [yoloeMasksEnabled, setYoloeMasksEnabled] = useAtom(yoloeMasksEnabledAtom);
  const [featherRadius, setFeatherRadius] = useAtom(featherRadiusAtom);
  const [, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);

  const detect = useCallback(
    async (sessionId: string) => {
      if (!detectQuery.trim()) return;
      const target = selectedDetector || "grounding-dino-tiny";
      if (!selectedDetector) setSelectedDetector(target);
      setIsDetecting(true);
      try {
        if (loadedDetector !== target) {
          setDetectorLoading(true);
          try {
            await api.loadDetector(target);
            setLoadedDetector(target);
          } finally {
            setDetectorLoading(false);
          }
        }
        const t = Date.now();
        const res = await api.detect(sessionId, {
          query: detectQuery,
          max_detections: 20,
          use_yoloe_masks: yoloeMasksEnabled,
        });
        emitUi(
          "detect",
          "INFO",
          `Found ${res.detections.length} for "${detectQuery}" with ${target} in ${Date.now() - t}ms`
        );
        const base = detections.length;
        setDetections((prev) => [...prev, ...res.detections]);
        // Detections append, so masks for indices that still exist are kept and
        // the rest are dropped rather than left pointing at nothing.
        setPerDetectionMasks((prev) => {
          const next: Record<number, PackedMask> = {};
          for (const [k, v] of Object.entries(prev)) {
            const idx = Number(k);
            if (idx < base) next[idx] = v;
          }
          return next;
        });
        setSelectedDetection(null);
        return { detections: res.detections, baseOffset: base };
      } catch (err) {
        logErr("detect", err);
        pushToast("Detection failed — check the console for details");
        setLoadedDetector(null);
        return null;
      } finally {
        setIsDetecting(false);
      }
    },
    [
      detectQuery,
      selectedDetector,
      yoloeMasksEnabled,
      loadedDetector,
      detections.length,
      setSelectedDetector,
      setLoadedDetector,
      setDetectorLoading,
      setDetections,
      setSelectedDetection,
      setIsDetecting,
      setPerDetectionMasks,
    ]
  );

  return {
    detectMode,
    toolMode,
    brushSize,
    setBrushSize,
    detectors,
    selectedDetector,
    setSelectedDetector,
    detectorLoading,
    detectQuery,
    setDetectQuery,
    detections,
    setDetections,
    selectedDetection,
    setSelectedDetection,
    isDetecting,
    yoloeMasksEnabled,
    setYoloeMasksEnabled,
    featherRadius,
    setFeatherRadius,
    loadedDetector,
    detect,
  };
}
