import { useCallback } from "react";
import { useAtom } from "jotai";
import {
  layersAtom,
  selectedLayersAtom,
  nextLayerId,
  type Layer,
} from "@/store/layers";
import { masksAtom, objectMasksAtom, perDetectionMasksAtom, sessionIdAtom } from "@/store/session";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import {
  removeDetectionMask,
  removeDetectionIndexFromLayers,
  shiftSelectedDetection,
} from "@/lib/detectionEdit";
import { api } from "@/lib/api";
import { emitUi, logErr } from "@/store/logs";
import { useSession } from "./useSession";

export function useLayers() {
  const [layers, setLayers] = useAtom(layersAtom);
  const [selectedLayers, setSelectedLayers] = useAtom(selectedLayersAtom);
  const [, setMasks] = useAtom(masksAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [, setDetections] = useAtom(detectionsAtom);
  const [, setSelectedDetection] = useAtom(selectedDetectionAtom);
  const [sessionId] = useAtom(sessionIdAtom);

  const { applyHistory } = useSession();

  const clearServerBrushObjects = useCallback(
    (oids: number[]) => {
      if (!sessionId || oids.length === 0) return;
      emitUi("layer", "INFO", `Cleared ${oids.length} subject(s) on the server`);
      oids.forEach((oid) => {
        api
          .clearObject(sessionId, oid)
          .then((res) => applyHistory(res.objectHistory))
          .catch((err) => logErr("subject", err));
      });
    },
    [sessionId, applyHistory]
  );

  const addLayer = useCallback(
    (layer: Omit<Layer, "id" | "createdAt">) => {
      const id = nextLayerId(layers);
      setLayers((prev) => [...prev, { ...layer, id, createdAt: Date.now() }]);
      return id;
    },
    [layers, setLayers]
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
        setLayers((prev) => removeDetectionIndexFromLayers(prev, idx));
        setPerDetectionMasks((prev) => removeDetectionMask(prev, idx));
        setDetections((prev) => prev.filter((_, i) => i !== idx));
        setSelectedDetection((prev) => shiftSelectedDetection(prev, idx));
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
      const detLayers: Layer[] = dets.map((d, i) => ({
        id: nextLayerId(layers),
        type: "detection",
        label: `#${i + 1} ${d.label}`,
        preview: null,
        objectId: 0,
        detectionIndex: i,
        createdAt: Date.now(),
      }));
      setLayers([...base, ...detLayers]);
    },
    [layers, setLayers]
  );

  const appendDetectionLayers = useCallback(
    (dets: { label: string }[], offset: number) => {
      const base = layers.filter((l) => l.type !== "detection");
      const existing = layers.filter((l) => l.type === "detection");
      const nextOffset = existing.length === 0 ? 0 : Math.max(offset, 0);
      const detLayers: Layer[] = [];
      for (let i = 0; i < dets.length; i++) {
        detLayers.push({
          id: nextLayerId(layers),
          type: "detection",
          label: `#${nextOffset + i + 1} ${dets[i].label}`,
          preview: null,
          objectId: 0,
          detectionIndex: nextOffset + i,
          createdAt: Date.now(),
        });
      }
      setLayers([...base, ...existing, ...detLayers]);
    },
    [layers, setLayers]
  );

  return {
    layers,
    selectedLayers,
    addLayer,
    removeLayer,
    clearAllLayers,
    selectLayer,
    syncDetectionLayers,
    appendDetectionLayers,
  };
}
