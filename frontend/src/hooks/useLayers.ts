import { useCallback } from "react";
import { useAtom } from "jotai";
import {
  layersAtom,
  selectedLayersAtom,
  layerIdCounterAtom,
  type Layer,
} from "@/store/layers";
import { masksAtom, objectMasksAtom, perDetectionMasksAtom, objectUndoCountsAtom, objectRedoCountsAtom, sessionIdAtom } from "@/store/session";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import type { PackedMask } from "@/lib/mask";
import { api } from "@/lib/api";

export function useLayers() {
  const [layers, setLayers] = useAtom(layersAtom);
  const [selectedLayers, setSelectedLayers] = useAtom(selectedLayersAtom);
  const [layerIdCounter, setLayerIdCounter] = useAtom(layerIdCounterAtom);
  const [, setMasks] = useAtom(masksAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [, setDetections] = useAtom(detectionsAtom);
  const [, setSelectedDetection] = useAtom(selectedDetectionAtom);
  const [sessionId] = useAtom(sessionIdAtom);
  const [, setObjectUndoCounts] = useAtom(objectUndoCountsAtom);
  const [, setObjectRedoCounts] = useAtom(objectRedoCountsAtom);

  const clearServerBrushObjects = useCallback(
    (oids: number[]) => {
      if (!sessionId || oids.length === 0) return;
      oids.forEach((oid) => {
        api.clearObject(sessionId, oid).catch(() => {});
      });
      setObjectUndoCounts((prev) => {
        const next = { ...prev };
        oids.forEach((oid) => delete next[oid]);
        return next;
      });
      setObjectRedoCounts((prev) => {
        const next = { ...prev };
        oids.forEach((oid) => delete next[oid]);
        return next;
      });
    },
    [sessionId, setObjectUndoCounts, setObjectRedoCounts]
  );

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
          clearServerBrushObjects(
            layers.filter((l) => l.type === "brush").map((l) => l.objectId)
          );
        }
      }
    },
    [layers, setLayers, setSelectedLayers, setMasks, setObjectMasks, setPerDetectionMasks, setDetections, setSelectedDetection, clearServerBrushObjects]
  );

  const clearAllLayers = useCallback(() => {
    clearServerBrushObjects(
      layers.filter((l) => l.type === "brush").map((l) => l.objectId)
    );
    setLayers([]);
    setSelectedLayers(new Set<string>());
    setMasks([]);
    setObjectMasks({});
    setPerDetectionMasks({});
    setDetections([]);
    setSelectedDetection(null);
  }, [layers, setLayers, setSelectedLayers, setMasks, setObjectMasks, setPerDetectionMasks, setDetections, setSelectedDetection, clearServerBrushObjects]);

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

  const syncDetectionLayers = useCallback(
    (dets: { label: string }[]) => {
      const base = layers.filter((l) => l.type !== "detection");
      let counter = layerIdCounter;
      const detLayers: Layer[] = dets.map((d, i) => ({
        id: `layer-${++counter}`,
        type: "detection",
        label: `#${i + 1} ${d.label}`,
        preview: null,
        objectId: 0,
        detectionIndex: i,
        createdAt: Date.now(),
      }));
      setLayerIdCounter(counter);
      setLayers([...base, ...detLayers]);
    },
    [layers, layerIdCounter, setLayers, setLayerIdCounter]
  );

  return {
    layers,
    selectedLayers,
    addLayer,
    removeLayer,
    clearAllLayers,
    selectLayer,
    syncDetectionLayers,
  };
}
