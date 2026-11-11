import { useCallback, useRef } from "react";
import { useAtom } from "jotai";
import { sessionIdAtom } from "@/store/session";
import {
  matteActiveLayerIdAtom,
  matteBusyAtom,
  matteGuideAtom,
  matteHistoryBusyAtom,
  matteInfoAtom,
  matteLayersAtom,
  matteModelAtom,
  matteModelsAtom,
  matteStrokesAtom,
} from "@/store/matte";
import { api, ApiError } from "@/lib/api";
import {
  createMatteLayer,
  decodeAlphaPng,
  emptyMatteStrokes,
  matteLayerName,
  type MatteGuide,
  type MatteInfo,
  type MatteLayer,
  type MatteSnapshot,
  type MatteStrokeSet,
} from "@/lib/matte";
import { emitUi } from "@/store/logs";
import { pushToast } from "@/store/ui";

interface RunOptions {
  guide?: MatteGuide;
  model?: string;
  points?: number[][];
  labels?: number[];
  text?: string;
  bbox?: number[];
}

const MAX_HISTORY = 12;
const runLock = { current: false };
const historyLock = { current: false };
let runSeq = 0;

function describeFailure(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 507) {
      return "Out of memory - try a lighter matte model";
    }
    const body = err.message
      .replace(/^API \d+: /, "")
      .replace(/^\{"detail":?"?/, "")
      .replace(/"?\}$/, "");
    return body.slice(0, 160) || "Matte failed";
  }
  return "Matte failed - the task did not finish";
}

function snapshot(layer: MatteLayer): MatteSnapshot {
  return {
    alphaB64: layer.alphaB64,
    strokes: { points: [...layer.strokes.points], labels: [...layer.strokes.labels] },
    guide: layer.guide,
    info: layer.info,
  };
}

export function useMatte() {
  const [sessionId] = useAtom(sessionIdAtom);
  const [model, setModel] = useAtom(matteModelAtom);
  const [models] = useAtom(matteModelsAtom);
  const [guide, setGuide] = useAtom(matteGuideAtom);
  const [layers, setLayers] = useAtom(matteLayersAtom);
  const [activeLayerId, setActiveLayerId] = useAtom(matteActiveLayerIdAtom);
  const [, setStrokes] = useAtom(matteStrokesAtom);
  const [, setBusy] = useAtom(matteBusyAtom);
  const [, setHistoryBusy] = useAtom(matteHistoryBusyAtom);
  const [, setInfo] = useAtom(matteInfoAtom);

  const activeLayer = layers.find((layer) => layer.id === activeLayerId) ?? layers[0] ?? null;
  const activeRef = useRef<MatteLayer | null>(activeLayer);
  activeRef.current = activeLayer;
  const layersRef = useRef(layers);
  layersRef.current = layers;

  const showLayer = useCallback(
    (layer: MatteLayer | null) => {
      if (!layer) {
        setInfo(null);
        return;
      }
      setGuide(layer.guide);
      setStrokes(layer.strokes);
      setInfo(layer.info);
    },
    [setGuide, setInfo, setStrokes]
  );

  const selectLayer = useCallback(
    (id: number) => {
      const layer = layersRef.current.find((item) => item.id === id);
      if (!layer) return;
      setActiveLayerId(id);
      setModel(layer.model);
      showLayer(layer);
    },
    [setActiveLayerId, setModel, showLayer]
  );

  const setLayerModel = useCallback(
    (id: number, nextModel: string) => {
      setLayers((prev) =>
        prev.map((item) => (item.id === id ? { ...item, model: nextModel } : item))
      );
    },
    [setLayers]
  );

  const toggleVisible = useCallback(
    (id: number) => {
      const target = layersRef.current.find((item) => item.id === id);
      if (!target) return;
      const nextVisible = !target.visible;
      const next = layersRef.current.map((item) =>
        item.id === id ? { ...item, visible: nextVisible } : item
      );
      setLayers(next);
      if (!nextVisible && activeLayerId === id) {
        const fallback = next.find((item) => item.visible);
        if (fallback) {
          setActiveLayerId(fallback.id);
          setModel(fallback.model);
          showLayer(fallback);
        }
      }
    },
    [activeLayerId, setActiveLayerId, setLayers, setModel, showLayer]
  );

  const run = useCallback(
    async (options?: RunOptions) => {
      if (!sessionId || runLock.current) return null;
      const layer = activeRef.current;
      if (!layer) return null;

      const nextGuide = options?.guide ?? guide;
      const runModel = options?.model ?? model;
      const nextStrokes: MatteStrokeSet =
        nextGuide === "stroke"
          ? {
              points: options?.points ?? layer.strokes.points,
              labels: options?.labels ?? layer.strokes.labels,
            }
          : emptyMatteStrokes();
      if (nextGuide === "stroke" && nextStrokes.points.length === 0) {
        pushToast("Paint the subject first");
        return null;
      }

      runLock.current = true;
      setBusy(true);
      const seq = ++runSeq;
      const knownModel = models.find((item) => item.name === runModel);
      emitUi(
        "matte",
        "INFO",
        knownModel && !knownModel.downloaded
          ? "Downloading the matte model once - the first run can take a few minutes"
          : nextGuide === "stroke"
            ? `Cutting out ${layer.name} from your strokes`
            : `Finding the main subject for ${layer.name}`
      );

      try {
        const res = await api.runMatte(sessionId, {
          model: runModel,
          guide: nextGuide,
          points: nextGuide === "stroke" ? nextStrokes.points : undefined,
          labels: nextGuide === "stroke" ? nextStrokes.labels : undefined,
          text: options?.text,
          bbox: options?.bbox,
          edge_shift: 0,
          edge_feather: 0,
          invert: false,
        });
        if (seq !== runSeq) return null;

        const raw = await decodeAlphaPng(res.alpha, seq);
        const nextInfo: MatteInfo = { model: res.model, guide: res.guide, ms: res.ms };
        const previousSnapshot = snapshot(layer);
        const nextLayer: MatteLayer = {
          ...layer,
          model: runModel,
          rawAlpha: raw,
          alphaB64: res.alpha,
          strokes: nextStrokes,
          guide: nextGuide,
          info: nextInfo,
          history: [...layer.history, previousSnapshot].slice(-MAX_HISTORY),
          redo: [],
        };

        setLayers((prev) => prev.map((item) => (item.id === layer.id ? nextLayer : item)));
        setGuide(nextGuide);
        setStrokes(nextStrokes);
        setInfo(nextInfo);
        emitUi("matte", "INFO", `${layer.name} is ready in ${res.ms} ms`);
        return res;
      } catch (err) {
        emitUi("matte", "ERROR", describeFailure(err));
        pushToast(describeFailure(err));
        return null;
      } finally {
        runLock.current = false;
        setBusy(false);
      }
    },
    [
      sessionId,
      model,
      models,
      guide,
      setBusy,
      setGuide,
      setInfo,
      setLayers,
      setStrokes,
    ]
  );

  const undo = useCallback(async () => {
    if (historyLock.current) return;
    const layer = activeRef.current;
    const previous = layer?.history[layer.history.length - 1];
    if (!layer || !previous) return;

    historyLock.current = true;
    setHistoryBusy(true);
    try {
      const decoded = previous.alphaB64
        ? await decodeAlphaPng(previous.alphaB64, ++runSeq)
        : null;
      const nextLayer: MatteLayer = {
        ...layer,
        alphaB64: previous.alphaB64,
        rawAlpha: decoded,
        strokes: previous.strokes,
        guide: previous.guide,
        info: previous.info,
        history: layer.history.slice(0, -1),
        redo: [...layer.redo, snapshot(layer)].slice(-MAX_HISTORY),
      };
      setLayers((prev) => prev.map((item) => (item.id === layer.id ? nextLayer : item)));
      setGuide(previous.guide);
      setStrokes(previous.strokes);
      setInfo(previous.info);
      emitUi("matte", "INFO", `Undid the last step on ${layer.name}`);
    } catch {
      pushToast("Could not restore that matte step");
    } finally {
      historyLock.current = false;
      setHistoryBusy(false);
    }
  }, [
    setGuide,
    setHistoryBusy,
    setInfo,
    setLayers,
    setStrokes,
  ]);

  const redo = useCallback(async () => {
    if (historyLock.current) return;
    const layer = activeRef.current;
    const next = layer?.redo[layer.redo.length - 1];
    if (!layer || !next) return;

    historyLock.current = true;
    setHistoryBusy(true);
    try {
      const decoded = next.alphaB64 ? await decodeAlphaPng(next.alphaB64, ++runSeq) : null;
      const nextLayer: MatteLayer = {
        ...layer,
        alphaB64: next.alphaB64,
        rawAlpha: decoded,
        strokes: next.strokes,
        guide: next.guide,
        info: next.info,
        history: [...layer.history, snapshot(layer)].slice(-MAX_HISTORY),
        redo: layer.redo.slice(0, -1),
      };
      setLayers((prev) => prev.map((item) => (item.id === layer.id ? nextLayer : item)));
      setGuide(next.guide);
      setStrokes(next.strokes);
      setInfo(next.info);
      emitUi("matte", "INFO", `Redid the last step on ${layer.name}`);
    } catch {
      pushToast("Could not restore that matte step");
    } finally {
      historyLock.current = false;
      setHistoryBusy(false);
    }
  }, [
    setGuide,
    setHistoryBusy,
    setInfo,
    setLayers,
    setStrokes,
  ]);

  const addLayer = useCallback(() => {
    const id = Math.max(-1, ...layersRef.current.map((layer) => layer.id)) + 1;
    const layer: MatteLayer = {
      ...createMatteLayer(id, model),
      guide: "stroke",
    };
    setLayers((prev) => [...prev, layer]);
    setActiveLayerId(id);
    setGuide("stroke");
    setStrokes(emptyMatteStrokes());
    setInfo(null);
    emitUi("matte", "INFO", `Added ${matteLayerName(id)} - paint the subject to cut it out`);
    return layer;
  }, [model, setActiveLayerId, setGuide, setInfo, setLayers, setStrokes]);

  const removeLayer = useCallback(
    (id: number) => {
      const current = layersRef.current;
      if (current.length <= 1) {
        pushToast("Keep at least one matte layer");
        return;
      }
      const index = current.findIndex((layer) => layer.id === id);
      const remaining = current.filter((layer) => layer.id !== id);
      setLayers(remaining);
      emitUi("matte", "INFO", `${current[index]?.name ?? "Layer"} removed`);
      if (activeLayerId === id) {
        const next = remaining[Math.min(Math.max(0, index), remaining.length - 1)];
        setActiveLayerId(next.id);
        showLayer(next);
      }
    },
    [activeLayerId, setActiveLayerId, setLayers, showLayer]
  );

  const reset = useCallback(() => {
    runSeq += 1;
    const layer = createMatteLayer(0, model);
    setLayers([layer]);
    setActiveLayerId(0);
    setGuide("auto");
    setStrokes(emptyMatteStrokes());
    setInfo(null);
  }, [model, setActiveLayerId, setGuide, setInfo, setLayers, setStrokes]);

  return {
    run,
    reset,
    undo,
    redo,
    addLayer,
    removeLayer,
    selectLayer,
    setLayerModel,
    toggleVisible,
    layers,
    activeLayer,
    activeLayerId,
    canUndo: (activeLayer?.history.length ?? 0) > 0,
    canRedo: (activeLayer?.redo.length ?? 0) > 0,
  };
}
