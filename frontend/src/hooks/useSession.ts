import { useCallback, useRef } from "react";
import { useAtom } from "jotai";import {
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
  brushPredictInFlightAtom,
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
import { api, ApiError } from "@/lib/api";
import { emitUi, logErr } from "@/store/logs";
import { isSupportedImage, fitImageFile } from "@/lib/utils";

const uploadGuardRef = { busy: false };
const endGuardRef = { busy: false };
const blobUrlRef: { current: string | null } = { current: null };
const predictQueueMap = new Map<string, Promise<unknown>>();
const deletedSubjects = new Set<number>();

function queueFor(sessionId: string): Promise<unknown> {
  let q = predictQueueMap.get(sessionId);
  if (!q) {
    q = Promise.resolve();
    predictQueueMap.set(sessionId, q);
  }
  return q;
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
  const [predictInFlight, setPredictInFlight] = useAtom(brushPredictInFlightAtom);

  const uploadGuard = uploadGuardRef;
  const endGuard = endGuardRef;
  const blobUrlGuard = blobUrlRef;
  const activeObjectIdRef = useRef(activeObjectId);
  activeObjectIdRef.current = activeObjectId;
  const objectUndoCountsRef = useRef(objectUndoCounts);
  objectUndoCountsRef.current = objectUndoCounts;
  const objectRedoCountsRef = useRef(objectRedoCounts);
  objectRedoCountsRef.current = objectRedoCounts;

  const syncSubjects = useCallback(
    (history?: Record<string, Partial<ObjectHistoryEntry>>) => {
      setSubjectMeta((prev) => {
        const next = { ...prev };
        for (const k of Object.keys(history ?? {})) {
          const oid = Number(k);
          if (!Number.isFinite(oid) || next[oid]) continue;
          // A subject the user deleted must not be re-created by a late or
          // out-of-order server response.
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
      setObjectHistory(nextHist);
      setObjectUndoCounts(nextUndo);
      setObjectRedoCounts(nextRedo);
      syncSubjects(history);
    },
    [setObjectHistory, setObjectUndoCounts, setObjectRedoCounts, syncSubjects]
  );

  const resetUndoRedoState = useCallback(() => {
    setObjectUndoCounts({});
    setObjectRedoCounts({});
    setObjectHistory({});
    setObjectMasks({});
    setBrushObjects([0]);
    setActiveObjectId(0);
    setSubjectMeta({ 0: { id: 0, name: subjectName(0), color: subjectColor(0) } });
  }, [
    setObjectUndoCounts, setObjectRedoCounts, setObjectHistory, setObjectMasks,
    setBrushObjects, setActiveObjectId, setSubjectMeta,
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
        emitUi("upload", "WARNING", "Upload ignored — an upload is already running");
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
            emitUi("upload", "WARNING", "Upload dropped — no model selected yet");
            return;
          }
          try {
            const res = await api.createSession(activeModel);
            if (modelNameRef.current !== activeModel) {
              api.destroySession(res.session_id).catch(() => {});
              emitUi("upload", "WARNING", "Upload dropped — model changed mid-flight");
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
            pushToast("Upload failed — check the console");
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
    emitUi("session", "WARNING", "Session expired on the server — recovering");
    await createSession(modelName, imageFile);
  }, [sessionId, modelName, imageFile, createSession]);

  const predict = useCallback(
    (params: { object_id?: number; points?: number[][]; labels?: number[]; bboxes?: number[] }) => {
      if (!sessionIdRef.current) return Promise.resolve(null);
      const oid = params.object_id ?? activeObjectIdRef.current;
      const sid = sessionIdRef.current;
      setPredictInFlight((c) => c + 1);
      const run = async () => {
        try {
          const sid = sessionIdRef.current;
          if (!sid) return null;
          const res = await api.predict(sid, { ...params, object_id: oid });
          setMasks(res.masks);
          setObjectMasks(res.objectMasks ?? {});
          applyHistory(res.objectHistory);
          return res;
        } catch (err) {
          logErr("brush", err);
          if (err instanceof ApiError && (err.status === 404 || err.status === 400)) {
            pushToast("Session expired — recovering");
            await recoverSession();
          } else {
            pushToast("Prediction failed");
          }
          return null;
        } finally {
          setPredictInFlight((c) => c - 1);
        }
      };
      const next = queueFor(sid).then(run);
      predictQueueMap.set(sid, next.catch(() => {}));
      return next;
    },
    [setMasks, setObjectMasks, recoverSession, setPredictInFlight, applyHistory]
  );

  const undo = useCallback(async () => {
    if (!sessionId) return;
    if (predictInFlight !== 0) {
      emitUi("brush", "WARNING", "Undo ignored — a prediction is still running");
      return;
    }
    const oid = activeObjectIdRef.current;
    if ((objectUndoCountsRef.current[oid] ?? 0) <= 0) return;
    try {
      const res = await api.undo(sessionId, oid);
      emitUi("brush", "INFO", `Undid a stroke on Subject ${oid + 1}`);
      setMasks(res.masks);
      setObjectMasks(res.objectMasks ?? {});
      applyHistory(res.objectHistory);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        applyHistory({ [oid]: { undo: 0, redo: 0, strokes: 0, has_mask: false } });
        pushToast("Nothing left to undo for this subject");
        return;
      }
      if (err instanceof ApiError && err.status === 404) {
        await recoverSession();
        return;
      }
      pushToast("Undo failed");
    }
  }, [sessionId, predictInFlight, setMasks, setObjectMasks, recoverSession, applyHistory]);

  const redo = useCallback(async () => {
    if (!sessionId) return;
    if (predictInFlight !== 0) {
      emitUi("brush", "WARNING", "Redo ignored — a prediction is still running");
      return;
    }
    const oid = activeObjectIdRef.current;
    if ((objectRedoCountsRef.current[oid] ?? 0) <= 0) return;
    try {
      const res = await api.redo(sessionId, oid);
      emitUi("brush", "INFO", `Redid a stroke on Subject ${oid + 1}`);
      setMasks(res.masks);
      setObjectMasks(res.objectMasks ?? {});
      applyHistory(res.objectHistory);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        applyHistory({ [oid]: { undo: 0, redo: 0, strokes: 0, has_mask: false } });
        pushToast("Nothing left to redo for this subject");
        return;
      }
      if (err instanceof ApiError && err.status === 404) {
        await recoverSession();
        return;
      }
      pushToast("Redo failed");
    }
  }, [sessionId, predictInFlight, setMasks, setObjectMasks, recoverSession, applyHistory]);

  const clearObjectHistory = useCallback(async (objectId: number = 0) => {
    if (sessionId) {
      try {
        const res = await api.clearObject(sessionId, objectId);
        setMasks(res.masks);
        setObjectMasks(res.objectMasks ?? {});
        applyHistory(res.objectHistory);
        return;
      } catch (err) {
        logErr("subject", err);
        pushToast("Could not clear that subject on the server");
      }
    }
    applyHistory({ [objectId]: { undo: 0, redo: 0, strokes: 0, has_mask: false } });
  }, [sessionId, setMasks, setObjectMasks, applyHistory]);

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
    canUndo: (objectUndoCounts[activeObjectId] ?? 0) > 0,
    canRedo: (objectRedoCounts[activeObjectId] ?? 0) > 0,
    clearObjectHistory,
    applyHistory,
    forgetSubject,
  };
}
