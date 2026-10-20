import { useCallback, useRef } from "react";
import { useAtom } from "jotai";
import {
  sessionIdAtom,
  modelNameAtom,
  imageUrlAtom,
  imageWidthAtom,
  imageHeightAtom,
  imageFileAtom,
  masksAtom,
  objectMasksAtom,
  perDetectionMasksAtom,
  brushObjectsAtom,
  activeObjectIdAtom,
  objectUndoCountsAtom,
  objectRedoCountsAtom,
  objectHistoryAtom,
  subjectMetaAtom,
  type ObjectHistory,
  subjectColor,
  subjectName,
  type ObjectHistoryEntry,
} from "@/store/session";
import { layersAtom, selectedLayersAtom } from "@/store/layers";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import { modeLockAtom, modeDialogOpenAtom, showTransparentAtom, hideBboxesAtom, unsupportedFileAtom, imageEncodingAtom, encodingMessageAtom, pushToast } from "@/store/ui";
import {
  sam3ModeAtom,
  sam3PromptsAtom,
  sam3RedoStackAtom,
  sam3PromptingAtom,
  sam3InstancesAtom,
  selectedSam3InstanceAtom,
} from "@/store/sam3";
import { api, ApiError, resetObjectMaskCache } from "@/lib/api";
import { emitUi, logErr } from "@/store/logs";
import { isSupportedImage, fitImageFile } from "@/lib/utils";
import { HistoryQueue, availableSteps, type HistoryKind } from "@/lib/historyQueue";
import type { PackedMask } from "@/lib/mask";

const uploadGuardRef = { busy: false };
const endGuardRef = { busy: false };
const blobUrlRef: { current: string | null } = { current: null };
const sessionQueueMap = new Map<string, Promise<unknown>>();
const deletedSubjects = new Set<number>();

const pendingStrokesRef: { current: Map<number, number> } = { current: new Map() };
const historyRef: { current: Record<number, ObjectHistory> } = { current: {} };
const historyQueueRef: { current: HistoryQueue } = { current: new HistoryQueue() };
const historyScheduledRef: { current: boolean } = { current: false };

function queueFor(sessionId: string): Promise<unknown> {
  let q = sessionQueueMap.get(sessionId);
  if (!q) {
    q = Promise.resolve();
    sessionQueueMap.set(sessionId, q);
  }
  return q;
}

function enqueue<T>(sessionId: string, task: () => Promise<T>): Promise<T> {
  const next = queueFor(sessionId).then(task);
  sessionQueueMap.set(sessionId, next.catch(() => {}));
  return next;
}

export function useSession() {
  const [sessionId, setSessionId] = useAtom(sessionIdAtom);
  const [modelName, setModelName] = useAtom(modelNameAtom);
  const [imageUrl, setImageUrl] = useAtom(imageUrlAtom);
  const [, setImageWidth] = useAtom(imageWidthAtom);
  const [, setImageHeight] = useAtom(imageHeightAtom);
  const [imageFile, setImageFile] = useAtom(imageFileAtom);
  const [, setMasks] = useAtom(masksAtom);
  const [, setObjectMasks] = useAtom(objectMasksAtom);
  const [, setPerDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [, setLayers] = useAtom(layersAtom);
  const [, setSelectedLayers] = useAtom(selectedLayersAtom);
  const [, setDetections] = useAtom(detectionsAtom);
  const [, setSelectedDetection] = useAtom(selectedDetectionAtom);
  const [, setModeLock] = useAtom(modeLockAtom);
  const [, setModeDialogOpen] = useAtom(modeDialogOpenAtom);
  const [, setShowTransparent] = useAtom(showTransparentAtom);
  const [, setHideBboxes] = useAtom(hideBboxesAtom);
  const [, setUnsupportedFile] = useAtom(unsupportedFileAtom);
  const [, setImageEncoding] = useAtom(imageEncodingAtom);
  const [, setEncodingMessage] = useAtom(encodingMessageAtom);
  const [sam3Mode] = useAtom(sam3ModeAtom);
  const [, setSam3Prompts] = useAtom(sam3PromptsAtom);
  const [, setSam3RedoStack] = useAtom(sam3RedoStackAtom);
  const [, setSam3Prompting] = useAtom(sam3PromptingAtom);
  const [, setSam3Instances] = useAtom(sam3InstancesAtom);
  const [, setSelectedSam3Instance] = useAtom(selectedSam3InstanceAtom);

  const [activeObjectId, setActiveObjectId] = useAtom(activeObjectIdAtom);
  const [, setBrushObjects] = useAtom(brushObjectsAtom);
  const [objectUndoCounts, setObjectUndoCounts] = useAtom(objectUndoCountsAtom);
  const [objectRedoCounts, setObjectRedoCounts] = useAtom(objectRedoCountsAtom);
  const [, setObjectHistory] = useAtom(objectHistoryAtom);
  const [, setSubjectMeta] = useAtom(subjectMetaAtom);

  const uploadGuard = uploadGuardRef;
  const endGuard = endGuardRef;
  const blobUrlGuard = blobUrlRef;
  const activeObjectIdRef = useRef(activeObjectId);
  activeObjectIdRef.current = activeObjectId;

  const syncSubjects = useCallback(
    (history?: Record<string, Partial<ObjectHistoryEntry>>) => {
      setSubjectMeta((prev) => {
        const next = { ...prev };
        for (const k of Object.keys(history ?? {})) {
          const oid = Number(k);
          if (!Number.isFinite(oid) || next[oid]) continue;
          if (deletedSubjects.has(oid)) continue;
          next[oid] = { id: oid, name: subjectName(oid), color: subjectColor(oid) };
        }
        return next;
      });
    },
    [setSubjectMeta]
  );

  const forgetSubject = useCallback((oid: number) => {
    deletedSubjects.add(oid);
  }, []);

  const applyHistory = useCallback(
    (history?: Record<string, Partial<ObjectHistoryEntry>>) => {
      if (!history) return;
      const nextUndo: Record<number, number> = {};
      const nextRedo: Record<number, number> = {};
      const nextHist: Record<number, ObjectHistory> = {};
      for (const [k, v] of Object.entries(history)) {
        const oid = Number(k);
        if (!Number.isFinite(oid)) continue;
        const undo = v.undo ?? 0;
        const redo = v.redo ?? 0;
        nextHist[oid] = {
          undo,
          redo,
          strokes: v.strokes ?? undo,
          has_mask: !!v.has_mask,
        };
        nextUndo[oid] = undo;
        nextRedo[oid] = redo;
      }
      historyRef.current = nextHist;
      setObjectHistory(nextHist);
      setObjectUndoCounts(nextUndo);
      setObjectRedoCounts(nextRedo);
      syncSubjects(history);
      return nextHist;
    },
    [setObjectHistory, setObjectUndoCounts, setObjectRedoCounts, syncSubjects]
  );

  const resetUndoRedoState = useCallback(() => {
    deletedSubjects.clear();
    resetObjectMaskCache();
    pendingStrokesRef.current.clear();
    historyRef.current = {};
    historyQueueRef.current.clear();
    setObjectUndoCounts({});
    setObjectRedoCounts({});
    setObjectHistory({});
    setObjectMasks({});
    setMasks([]);
    setBrushObjects([0]);
    setActiveObjectId(0);
    setSubjectMeta({ 0: { id: 0, name: subjectName(0), color: subjectColor(0) } });
  }, [
    setObjectUndoCounts, setObjectRedoCounts, setObjectHistory, setObjectMasks,
    setMasks, setBrushObjects, setActiveObjectId, setSubjectMeta,
  ]);

  const sam3ModeRef = useRef(sam3Mode);
  sam3ModeRef.current = sam3Mode;
  const modelNameRef = useRef(modelName);
  modelNameRef.current = modelName;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const createSession = useCallback(
    async (model: string, fileToReUpload?: File | null) => {
      try {
        const res = await api.createSession(model);
        sessionIdRef.current = res.session_id;
        setSessionId(res.session_id);
        setModelName(res.model_name);
        if (fileToReUpload) {
          setImageEncoding(true);
          try {
            const uploadRes = await api.uploadImage(res.session_id, fileToReUpload);
            setImageWidth(uploadRes.width);
            setImageHeight(uploadRes.height);
          } catch (err) {
            logErr("session", err);
            emitUi("upload", "ERROR", "Image re-upload failed");
          } finally {
            setImageEncoding(false);
          }
        }
      } catch (err) {
        logErr("session", err);
        emitUi("session", "ERROR", "Could not create session");
      }
    },
    [setSessionId, setModelName, setImageWidth, setImageHeight, setImageEncoding]
  );

  const previousBlobUrlRef = blobUrlGuard;

  const uploadImage = useCallback(
    async (file: File) => {
      if (uploadGuard.busy) {
        emitUi("upload", "WARNING", "Upload ignored - an upload is already running");
        return;
      }
      if (!isSupportedImage(file)) {
        const dot = file.name.lastIndexOf(".");
        setUnsupportedFile({
          name: file.name,
          extension: dot > -1 ? file.name.slice(dot + 1).toLowerCase() : "",
        });
        return;
      }
      uploadGuard.busy = true;
      let launched = false;
      try {
        let sid = sessionId;
        if (!sid) {
          const activeModel = modelNameRef.current;
          if (!activeModel) {
            emitUi("upload", "WARNING", "Upload dropped - no model selected yet");
            return;
          }
          try {
            const res = await api.createSession(activeModel);
            if (modelNameRef.current !== activeModel) {
              api.destroySession(res.session_id).catch(() => {});
              emitUi("upload", "WARNING", "Upload dropped - model changed mid-flight");
              return;
            }
            sessionIdRef.current = res.session_id;
            setSessionId(res.session_id);
            setModelName(res.model_name);
            sid = res.session_id;
          } catch (err) {
            logErr("session", err);
            emitUi("session", "ERROR", "Could not create session for upload");
            return;
          }
        }
        if (previousBlobUrlRef.current) {
          URL.revokeObjectURL(previousBlobUrlRef.current);
        }
        const { file: uploadFile, resized } = await fitImageFile(file);
        const url = URL.createObjectURL(uploadFile);
        previousBlobUrlRef.current = url;
        setImageUrl(url);
        setImageFile(uploadFile);
        setMasks([]);
        setPerDetectionMasks({});
        setLayers([]);
        setDetections([]);
        setSelectedDetection(null);
        setSelectedLayers(new Set<string>());
        setShowTransparent(false);
        setHideBboxes(false);
        setModeLock(null);
        setSam3Prompts([]);
        setSam3RedoStack([]);
        setSam3Instances([]);
        setSelectedSam3Instance(null);
        resetUndoRedoState();
        setEncodingMessage(resized ? "Resizing & re-encoding..." : null);
        setImageEncoding(true);
        launched = true;
        api.uploadImage(sid, uploadFile)
          .then((res) => {
            setImageWidth(res.width);
            setImageHeight(res.height);
            if (resized) {
              emitUi("upload", "INFO", `Image resized to ${res.width}x${res.height} before upload`);
            }
            if (!sam3ModeRef.current) setModeDialogOpen(true);
          })
          .catch((err) => {
            logErr("upload", err);
            pushToast("Upload failed - check the console");
          })
          .finally(() => {
            setImageEncoding(false);
            setEncodingMessage(null);
            uploadGuard.busy = false;
          });
      } finally {
        if (!launched) uploadGuard.busy = false;
      }
    },
    [sessionId, sam3Mode, setSessionId, setModelName, setImageUrl, setImageFile, setMasks, setLayers, setImageWidth, setImageHeight, setImageEncoding, setEncodingMessage, setModeLock, setModeDialogOpen, setDetections, setSelectedDetection, setSelectedLayers, setShowTransparent, setHideBboxes, setSam3Prompts, setSam3RedoStack, resetUndoRedoState, setUnsupportedFile]
  );

  const switchModel = useCallback(
    async (model: string) => {
      if (model === modelName) return;
      resetUndoRedoState();
      setSam3Prompts([]);
      setSam3RedoStack([]);
      setSam3Prompting(false);
      setSam3Instances([]);
      setSelectedSam3Instance(null);
      if (sessionId) {
        await api
          .destroySession(sessionId)
          .then(() => emitUi("session", "INFO", `Session ended for ${modelName}`))
          .catch((err) => logErr("session", err));
      }
      setMasks([]);
      setPerDetectionMasks({});
      setLayers([]);
      setDetections([]);
      setSelectedDetection(null);
      setSelectedLayers(new Set<string>());
      setHideBboxes(false);
      emitUi("mode", "INFO", `Switching model to ${model}`);
      await createSession(model, imageFile);
    },
    [sessionId, modelName, imageFile, createSession, setMasks, setPerDetectionMasks, setLayers, setDetections, setSelectedDetection, setSelectedLayers, setHideBboxes, resetUndoRedoState, setSam3Prompts, setSam3RedoStack, setSam3Prompting, setSam3Instances, setSelectedSam3Instance]
  );

  const endSession = useCallback(async () => {
    if (endGuard.busy) return;
    endGuard.busy = true;
    try {
      if (sessionId) {
        await api
          .destroySession(sessionId)
          .then(() => emitUi("session", "INFO", "Session ended"))
          .catch((err) => {
            logErr("session", err);
            pushToast("Failed to end session");
          });
      }
      if (previousBlobUrlRef.current) {
        URL.revokeObjectURL(previousBlobUrlRef.current);
        previousBlobUrlRef.current = null;
      }
      setSessionId(null);
      sessionIdRef.current = null;
      setModelName("");
      setImageUrl(null);
      setImageFile(null);
      setImageWidth(0);
      setImageHeight(0);
      setMasks([]);
      setPerDetectionMasks({});
      setLayers([]);
      setSelectedLayers(new Set<string>());
      setDetections([]);
      setSelectedDetection(null);
      setShowTransparent(false);
      setHideBboxes(false);
      setModeLock(null);
      setModeDialogOpen(false);
      setSam3Prompts([]);
      setSam3RedoStack([]);
      setSam3Prompting(false);
      setSam3Instances([]);
      setSelectedSam3Instance(null);
      resetUndoRedoState();
    } finally {
      endGuard.busy = false;
    }
  }, [
    sessionId, setSessionId, setModelName, setImageUrl, setImageFile,
    setImageWidth, setImageHeight, setMasks, setPerDetectionMasks,
    setLayers, setSelectedLayers, setDetections, setSelectedDetection,
    setShowTransparent, setHideBboxes, setModeLock, setModeDialogOpen,
    setSam3Prompts, setSam3RedoStack, setSam3Prompting, setSam3Instances, setSelectedSam3Instance,
    resetUndoRedoState,
  ]);

  const recoverSession = useCallback(async () => {
    if (!sessionId) return;
    if (await api.sessionHealth(sessionId)) return;
    emitUi("session", "WARNING", "Session expired on the server - recovering");
    resetUndoRedoState();
    setLayers((prev) => prev.filter((l) => l.type !== "brush"));
    await createSession(modelName, imageFile);
  }, [sessionId, modelName, imageFile, createSession, resetUndoRedoState, setLayers]);

  const applyPredict = useCallback(
    (res: {
      objectMasks?: Record<number, PackedMask>;
      objectHistory?: Record<string, Partial<ObjectHistoryEntry>>;
    }) => {
      const objectMasks = res.objectMasks ?? {};
      setMasks(Object.values(objectMasks));
      setObjectMasks(objectMasks);
      return applyHistory(res.objectHistory);
    },
    [setMasks, setObjectMasks, applyHistory]
  );

  const adjustPendingStrokes = useCallback((oid: number, delta: number) => {
    const map = pendingStrokesRef.current;
    const next = (map.get(oid) ?? 0) + delta;
    if (next > 0) map.set(oid, next);
    else map.delete(oid);
  }, []);

  const predict = useCallback(
    (params: { object_id?: number; points?: number[][]; labels?: number[]; bboxes?: number[] }) => {
      const sid = sessionIdRef.current;
      if (!sid) return Promise.resolve(null);
      const oid = params.object_id ?? activeObjectIdRef.current;
      const isStroke = !!(params.points?.length || params.bboxes?.length);
      if (isStroke) adjustPendingStrokes(oid, 1);
      return enqueue(sid, async () => {
        const active = sessionIdRef.current;
        if (!active) return null;
        try {
          const res = await api.predict(active, { ...params, object_id: oid });
          applyPredict(res);
          return res;
        } catch (err) {
          logErr("brush", err);
          if (err instanceof ApiError && err.status === 404) {
            pushToast("Session expired - recovering");
            await recoverSession();
          } else if (err instanceof ApiError && err.status === 400) {
            pushToast(err.message.replace(/^API \d+: /, "").slice(0, 140));
          } else {
            pushToast("Prediction failed");
          }
          return null;
        } finally {
          if (isStroke) adjustPendingStrokes(oid, -1);
        }
      });
    },
    [applyPredict, recoverSession, adjustPendingStrokes]
  );

  const patchHistory = useCallback(
    (objectId: number, patch: Partial<ObjectHistoryEntry>) => {
      setObjectHistory((prev) => {
        const cur = prev[objectId] ?? { undo: 0, redo: 0, strokes: 0, has_mask: false };
        const merged = { ...cur, ...patch };
        historyRef.current = { ...historyRef.current, [objectId]: merged };
        return { ...prev, [objectId]: merged };
      });
      if (patch.undo !== undefined) {
        setObjectUndoCounts((prev) => ({ ...prev, [objectId]: patch.undo as number }));
      }
      if (patch.redo !== undefined) {
        setObjectRedoCounts((prev) => ({ ...prev, [objectId]: patch.redo as number }));
      }
    },
    [setObjectHistory, setObjectUndoCounts, setObjectRedoCounts]
  );

  const historyDepth = useCallback((objectId: number, kind: HistoryKind) => {
    const ledger = historyRef.current[objectId];
    const depth = kind === "undo" ? (ledger?.undo ?? 0) : (ledger?.redo ?? 0);
    return availableSteps(depth, pendingStrokesRef.current.get(objectId) ?? 0, kind);
  }, []);

  const runHistoryOps = useCallback(async () => {
    const sid = sessionIdRef.current;
    const ops = historyQueueRef.current.drain();
    if (!sid) return;
    for (const op of ops) {
      const verb = op.kind === "undo" ? "Undid" : "Redid";
      const depth = (ledger?: ObjectHistory) =>
        op.kind === "undo" ? (ledger?.undo ?? 0) : (ledger?.redo ?? 0);
      const steps = Math.min(op.steps, historyDepth(op.objectId, op.kind));
      if (steps <= 0) continue;
      const before = depth(historyRef.current[op.objectId]);
      try {
        const res =
          op.kind === "undo"
            ? await api.undo(sid, op.objectId, steps)
            : await api.redo(sid, op.objectId, steps);
        const ledger = applyPredict(res);
        const applied = Math.max(0, before - depth(ledger?.[op.objectId]));
        emitUi(
          "brush",
          "INFO",
          `${verb} ${applied} stroke${applied === 1 ? "" : "s"} on Subject ${op.objectId + 1}`
        );
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          if (op.kind === "undo") {
            patchHistory(op.objectId, { undo: 0, strokes: 0, has_mask: false });
            pushToast("Nothing left to undo for this subject");
          } else {
            patchHistory(op.objectId, { redo: 0 });
            pushToast("Nothing left to redo for this subject");
          }
          continue;
        }
        if (err instanceof ApiError && err.status === 404) {
          await recoverSession();
          continue;
        }
        pushToast(op.kind === "undo" ? "Undo failed" : "Redo failed");
      }
    }
  }, [applyPredict, patchHistory, recoverSession, historyDepth]);

  const scheduleHistoryDrain = useCallback(() => {
    const sid = sessionIdRef.current;
    if (!sid || historyScheduledRef.current) return;
    historyScheduledRef.current = true;
    void enqueue(sid, async () => {
      historyScheduledRef.current = false;
      do {
        await runHistoryOps();
      } while (sessionIdRef.current === sid && historyQueueRef.current.size > 0);
    }).catch((err) => logErr("brush", err));
  }, [runHistoryOps]);

  const requestHistory = useCallback(
    (objectId: number, kind: HistoryKind) => {
      if (historyDepth(objectId, kind) <= 0) return;
      historyQueueRef.current.push(objectId, kind);
      scheduleHistoryDrain();
    },
    [historyDepth, scheduleHistoryDrain]
  );

  const undoSubject = useCallback(
    (objectId: number) => requestHistory(objectId, "undo"),
    [requestHistory]
  );
  const redoSubject = useCallback(
    (objectId: number) => requestHistory(objectId, "redo"),
    [requestHistory]
  );

  const undo = useCallback(
    () => requestHistory(activeObjectIdRef.current, "undo"),
    [requestHistory]
  );
  const redo = useCallback(
    () => requestHistory(activeObjectIdRef.current, "redo"),
    [requestHistory]
  );

  const clearObjectHistory = useCallback(async (objectId: number = 0) => {
    const sid = sessionIdRef.current;
    if (sid) {
      try {
        const res = await enqueue(sid, () => api.clearObject(sid, objectId));
        applyPredict(res);
        return;
      } catch (err) {
        logErr("subject", err);
        pushToast("Could not clear that subject on the server");
      }
    }
    patchHistory(objectId, { undo: 0, redo: 0, strokes: 0, has_mask: false });
  }, [applyPredict, patchHistory]);

  return {
    sessionId,
    imageUrl,
    uploadImage,
    switchModel,
    endSession,
    recoverSession,
    predict,
    undo,
    redo,
    undoSubject,
    redoSubject,
    canUndo: (objectUndoCounts[activeObjectId] ?? 0) > 0,
    canRedo: (objectRedoCounts[activeObjectId] ?? 0) > 0,
    clearObjectHistory,
    applyHistory,
    forgetSubject,
  };
}
