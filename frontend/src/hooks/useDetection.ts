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
import { perDetectionMasksAtom } from "@/store/session";

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
      setDetections([]);
      setPerDetectionMasks({});
      setSelectedDetection(null);
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
        const res = await api.detect(sessionId, {
          query: detectQuery,
          max_detections: 20,
          use_yoloe_masks: yoloeMasksEnabled,
        });
        setDetections(res.detections);
        return res.detections;
      } catch (err) {
        console.error("Detection failed:", err);
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
