import { useCallback } from "react";
import { useAtom } from "jotai";
import {
  layersAtom,
  selectedLayersAtom,
  layerIdCounterAtom,
  type Layer,
} from "@/store/layers";
import { masksAtom, objectMasksAtom, perDetectionMasksAtom } from "@/store/session";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import type { PackedMask } from "@/lib/mask";

export function useLayers() {
  const [layers, setLayers] = useAtom(layersAtom);
  const [selectedLayers, setSelectedLayers] = useAtom(selectedLayersAtom);
  const [layerIdCounter, setLayerIdCounter] = useAtom(layerIdCounterAtom);
  const [, setMasks] = useAtom(masksAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [, setDetections] = useAtom(detectionsAtom);
  const [, setSelectedDetection] = useAtom(selectedDetectionAtom);

  const addLayer = useCallback(
    (layer: Omit<Layer, "id" | "createdAt">) => {
      const id = `layer-${layerIdCounter + 1}`;
      setLayerIdCounter((c) => c + 1);
      setLayers((prev) => [
        ...prev,
        { ...layer, id, createdAt: Date.now() },
      ]);
      return id;
    },
    [layerIdCounter, setLayerIdCounter, setLayers]
  );

  const removeLayer = useCallback(
    (id: string) => {
      const removedLayer = layers.find((l) => l.id === id);
      setSelectedLayers((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      if (removedLayer?.detectionIndex !== undefined) {
        const idx = removedLayer.detectionIndex;
        setLayers((prev) => prev
          .filter((l) => l.id !== id)
          .map((l) => l.detectionIndex !== undefined && l.detectionIndex > idx
            ? { ...l, detectionIndex: l.detectionIndex - 1 }
            : l
          )
        );
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
          if (prev === idx) return null;
          if (prev !== null && prev > idx) return prev - 1;
          return prev;
        });
      } else {
        setLayers((prev) => prev.filter((l) => l.id !== id));
        const remainingBrush = layers.some((l) => l.type === "brush" && l.id !== id);
        if (!remainingBrush) {
          setMasks([]);
          setObjectMasks({});
        }
      }
    },
    [layers, setLayers, setSelectedLayers, setMasks, setObjectMasks, setPerDetectionMasks, setDetections, setSelectedDetection]
  );

  const clearAllLayers = useCallback(() => {
    setLayers([]);
    setSelectedLayers(new Set<string>());
    setMasks([]);
    setObjectMasks({});
    setPerDetectionMasks({});
    setDetections([]);
    setSelectedDetection(null);
  }, [setLayers, setSelectedLayers, setMasks, setObjectMasks, setPerDetectionMasks, setDetections, setSelectedDetection]);

  const selectLayer = useCallback(
    (id: string, ctrlKey: boolean) => {
      setSelectedLayers((prev) => {
        const next = new Set(prev);
        if (ctrlKey) {
          if (next.has(id)) next.delete(id);
          else next.add(id);
        } else {
          next.clear();
          next.add(id);
        }
        return next;
      });
    },
    [setSelectedLayers]
  );

  return {
    layers,
    selectedLayers,
    addLayer,
    removeLayer,
    clearAllLayers,
    selectLayer,
  };
}
