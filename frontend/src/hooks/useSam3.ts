import { useCallback, useRef, useState } from "react";
import { useAtom } from "jotai";
import { sessionIdAtom, masksAtom } from "@/store/session";
import {
  sam3PromptsAtom,
  sam3RedoStackAtom,
  sam3PromptingAtom,
  sam3InstancesAtom,
  selectedSam3InstanceAtom,
  type Sam3Prompt,
  type Sam3Instance,
} from "@/store/sam3";
import { layersAtom, layerIdCounterAtom, type Layer } from "@/store/layers";
import { api } from "@/lib/api";
import type { PackedMask } from "@/lib/mask";

const instanceKey = (pi: number, ii: number) => `${pi}:${ii}`;

export function useSam3() {
  const [sessionId] = useAtom(sessionIdAtom);
  const [, setMasks] = useAtom(masksAtom);
  const [prompts, setPrompts] = useAtom(sam3PromptsAtom);
  const [redoStack, setRedoStack] = useAtom(sam3RedoStackAtom);
  const [prompting, setPrompting] = useAtom(sam3PromptingAtom);
  const [, setInstances] = useAtom(sam3InstancesAtom);
  const [selectedInstance, setSelectedInstance] = useAtom(selectedSam3InstanceAtom);
  const [layers, setLayers] = useAtom(layersAtom);
  const [layerIdCounter, setLayerIdCounter] = useAtom(layerIdCounterAtom);
  const layerIdCounterRef = useRef(layerIdCounter);
  layerIdCounterRef.current = layerIdCounter;
  const [error, setError] = useState<string | null>(null);

  const extractError = (err: unknown): string => {
    const msg = err instanceof Error ? err.message : String(err);
    const m = msg.match(/"detail"\s*:\s*"([^"]*)"/);
    return m ? m[1] : msg;
  };

  const reconcileLayers = useCallback(
    (insts: Sam3Instance[]) => {
      const base = layers.filter((l) => l.type !== "sam3");
      const existing = new Map<string, Layer>();
      layers.forEach((l) => {
        if (l.type === "sam3" && l.promptIndex !== undefined && l.instanceIndex !== undefined) {
          existing.set(instanceKey(l.promptIndex, l.instanceIndex), l);
        }
      });
      let counter = layerIdCounterRef.current;
      const sam3Layers: Layer[] = insts.map((inst) => {
        const key = instanceKey(inst.promptIndex, inst.instanceIndex);
        const prevLayer = existing.get(key);
        if (prevLayer) {
          return {
            ...prevLayer,
            label: `#${inst.instanceIndex + 1} ${inst.text}`,
          };
        }
        counter += 1;
        return {
          id: `layer-${counter}`,
          type: "sam3",
          label: `#${inst.instanceIndex + 1} ${inst.text}`,
          preview: null,
          objectId: 0,
          promptIndex: inst.promptIndex,
          instanceIndex: inst.instanceIndex,
          createdAt: Date.now(),
        };
      });
      if (counter !== layerIdCounterRef.current) {
        layerIdCounterRef.current = counter;
        setLayerIdCounter(counter);
      }
      setLayers([...base, ...sam3Layers]);
    },
    [layers, setLayers, setLayerIdCounter]
  );

  const syncFromPrompts = useCallback(
    (list: Sam3Prompt[]) => {
      const allMasks: PackedMask[] = [];
      const instances: Sam3Instance[] = [];
      list.forEach((p, pi) => {
        (p.masks || []).forEach((m, ii) => {
          allMasks.push(m);
          instances.push({
            promptIndex: pi,
            instanceIndex: ii,
            text: p.text,
            bbox: p.bboxes?.[ii] ?? [],
            score: p.scores?.[ii] ?? 0,
          });
        });
      });
      setMasks(allMasks);
      setInstances(instances);
      reconcileLayers(instances);
    },
    [setMasks, setInstances, reconcileLayers]
  );

  const prompt = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!sessionId || !trimmed) return null;
      setPrompting(true);
      setError(null);
      try {
        const res = await api.sam3Prompt(sessionId, trimmed);
        const next = [
          ...prompts,
          { text: trimmed, masks: res.masks, scores: res.scores, bboxes: res.bboxes },
        ];
        setPrompts(next);
        setRedoStack([]);
        syncFromPrompts(next);
        return res;
      } catch (err) {
        console.error("SAM3 prompt failed:", err);
        setError(extractError(err) || "Segmentation failed — please try again.");
        return null;
      } finally {
        setPrompting(false);
      }
    },
    [sessionId, prompts, setPrompts, setRedoStack, setPrompting, syncFromPrompts]
  );

  const undo = useCallback(async () => {
    if (!sessionId || prompting || prompts.length === 0) return;
    const last = prompts[prompts.length - 1];
    const next = prompts.slice(0, -1);
    setPrompts(next);
    setRedoStack([...redoStack, last]);
    syncFromPrompts(next);
    setSelectedInstance(null);
    try {
      await api.sam3Undo(sessionId);
    } catch {
      console.error("SAM3 undo failed");
    }
  }, [sessionId, prompting, prompts, redoStack, setPrompts, setRedoStack, syncFromPrompts, setSelectedInstance]);

  const redo = useCallback(async () => {
    if (!sessionId || prompting || redoStack.length === 0) return;
    const last = redoStack[redoStack.length - 1];
    const next = [...prompts, last];
    setRedoStack(redoStack.slice(0, -1));
    setPrompts(next);
    syncFromPrompts(next);
    try {
      await api.sam3Redo(sessionId);
    } catch {
      console.error("SAM3 redo failed");
    }
  }, [sessionId, prompting, prompts, redoStack, setPrompts, setRedoStack, syncFromPrompts]);

  const removeInstance = useCallback(
    async (promptIndex: number, instanceIndex: number) => {
      if (!sessionId || prompting) return;
      const next = prompts
        .map((p, pi) => {
          if (pi !== promptIndex) return p;
          return {
            ...p,
            masks: p.masks.filter((_, ii) => ii !== instanceIndex),
            scores: p.scores ? p.scores.filter((_, ii) => ii !== instanceIndex) : p.scores,
            bboxes: p.bboxes ? p.bboxes.filter((_, ii) => ii !== instanceIndex) : p.bboxes,
          };
        })
        .filter((p) => p.masks.length > 0);
      const prevRedo = redoStack;
      const prevSelected = selectedInstance;
      setPrompts(next);
      setRedoStack([]);
      syncFromPrompts(next);
      setSelectedInstance(null);
      try {
        await api.sam3RemoveInstance(sessionId, promptIndex, instanceIndex);
      } catch (err) {
        console.error("SAM3 remove instance failed:", err);
        setPrompts(prompts);
        setRedoStack(prevRedo);
        syncFromPrompts(prompts);
        setSelectedInstance(prevSelected);
      }
    },
    [sessionId, prompting, prompts, redoStack, selectedInstance, setPrompts, setRedoStack, syncFromPrompts, setSelectedInstance]
  );

  return {
    prompt,
    undo,
    redo,
    removeInstance,
    canUndo: prompts.length > 0,
    canRedo: redoStack.length > 0,
    prompting,
    error,
    setError,
  };
}
