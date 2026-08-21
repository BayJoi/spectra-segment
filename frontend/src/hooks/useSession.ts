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
  brushPredictInFlightAtom,
} from "@/store/session";
import { layersAtom, selectedLayersAtom } from "@/store/layers";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import { modeLockAtom, modeDialogOpenAtom, showTransparentAtom, hideBboxesAtom, unsupportedFileAtom, imageEncodingAtom } from "@/store/ui";
import {
  sam3ModeAtom,
  sam3PromptsAtom,
  sam3RedoStackAtom,
  sam3PromptingAtom,
  sam3InstancesAtom,
  selectedSam3InstanceAtom,
} from "@/store/sam3";
import { api } from "@/lib/api";
import { isSupportedImage } from "@/lib/utils";

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
  const [predictInFlight, setPredictInFlight] = useAtom(brushPredictInFlightAtom);
  const predictQueueRef = useRef<Promise<unknown>>(Promise.resolve());
  const activeObjectIdRef = useRef(activeObjectId);
  activeObjectIdRef.current = activeObjectId;
  const objectUndoCountsRef = useRef(objectUndoCounts);
  objectUndoCountsRef.current = objectUndoCounts;
  const objectRedoCountsRef = useRef(objectRedoCounts);
  objectRedoCountsRef.current = objectRedoCounts;
  const endingRef = useRef(false);

  const resetUndoRedoState = useCallback(() => {
    setObjectUndoCounts({});
    setObjectRedoCounts({});
    setObjectMasks({});
    setBrushObjects([0]);
    setActiveObjectId(0);
  }, [setObjectUndoCounts, setObjectRedoCounts, setObjectMasks, setBrushObjects, setActiveObjectId]);

  const sam3ModeRef = useRef(sam3Mode);
  sam3ModeRef.current = sam3Mode;
  const modelNameRef = useRef(modelName);
  modelNameRef.current = modelName;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const uploadingRef = useRef(false);

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
            console.error("Re-upload failed:", err);
          } finally {
            setImageEncoding(false);
          }
        }
      } catch (err) {
        console.error("Failed to create session:", err);
      }
    },
    [setSessionId, setModelName, setImageWidth, setImageHeight, setImageEncoding]
  );

  const previousBlobUrlRef = useRef<string | null>(null);

  const uploadImage = useCallback(
    async (file: File) => {
      if (uploadingRef.current) return;
      if (!isSupportedImage(file)) {
        const dot = file.name.lastIndexOf(".");
        setUnsupportedFile({
          name: file.name,
          extension: dot > -1 ? file.name.slice(dot + 1).toLowerCase() : "",
        });
        return;
      }
      uploadingRef.current = true;
      let launched = false;
      try {
        let sid = sessionId;
        if (!sid) {
          const activeModel = modelNameRef.current;
          if (!activeModel) return;
          try {
            const res = await api.createSession(activeModel);
            if (modelNameRef.current !== activeModel) {
              api.destroySession(res.session_id).catch(() => {});
              return;
            }
            sessionIdRef.current = res.session_id;
            setSessionId(res.session_id);
            setModelName(res.model_name);
            sid = res.session_id;
          } catch (err) {
            console.error("Failed to create session:", err);
            return;
          }
        }
        if (previousBlobUrlRef.current) {
          URL.revokeObjectURL(previousBlobUrlRef.current);
        }
        const url = URL.createObjectURL(file);
        previousBlobUrlRef.current = url;
        setImageUrl(url);
        setImageFile(file);
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
        setImageEncoding(true);
        launched = true;
        api.uploadImage(sid, file)
          .then((res) => {
            setImageWidth(res.width);
            setImageHeight(res.height);
            if (!sam3ModeRef.current) setModeDialogOpen(true);
          })
          .catch((err) => {
            console.error("Upload failed:", err);
          })
          .finally(() => {
            setImageEncoding(false);
            uploadingRef.current = false;
          });
      } finally {
        if (!launched) uploadingRef.current = false;
      }
    },
    [sessionId, sam3Mode, setSessionId, setModelName, setImageUrl, setImageFile, setMasks, setLayers, setImageWidth, setImageHeight, setImageEncoding, setModeLock, setModeDialogOpen, setDetections, setSelectedDetection, setSelectedLayers, setShowTransparent, setHideBboxes, setSam3Prompts, setSam3RedoStack, resetUndoRedoState, setUnsupportedFile]
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
        await api.destroySession(sessionId).catch(() => {});
      }
      setMasks([]);
      setPerDetectionMasks({});
      setLayers([]);
      setDetections([]);
      setSelectedDetection(null);
      setSelectedLayers(new Set<string>());
      setHideBboxes(false);
      await createSession(model, imageFile);
    },
    [sessionId, modelName, imageFile, createSession, setMasks, setPerDetectionMasks, setLayers, setDetections, setSelectedDetection, setSelectedLayers, setHideBboxes, resetUndoRedoState, setSam3Prompts, setSam3RedoStack, setSam3Prompting, setSam3Instances, setSelectedSam3Instance]
  );

  const endSession = useCallback(async () => {
    if (endingRef.current) return;
    endingRef.current = true;
    try {
      if (sessionId) {
        await api.destroySession(sessionId).catch(() => {});
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
      endingRef.current = false;
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
    await createSession(modelName, imageFile);
  }, [sessionId, modelName, imageFile, createSession]);

  const predict = useCallback(
    (params: { object_id?: number; points?: number[][]; labels?: number[]; bboxes?: number[] }) => {
      if (!sessionIdRef.current) return Promise.resolve(null);
      const oid = params.object_id ?? activeObjectIdRef.current;
      setPredictInFlight((c) => c + 1);
      const run = async () => {
        try {
          const sid = sessionIdRef.current;
          if (!sid) return null;
          const res = await api.predict(sid, { ...params, object_id: oid });
          setMasks(res.masks);
          setObjectMasks(res.objectMasks ?? {});
          if (params.points?.length || params.bboxes?.length) {
            setObjectUndoCounts((prev) => ({ ...prev, [oid]: (prev[oid] ?? 0) + 1 }));
            setObjectRedoCounts((prev) => ({ ...prev, [oid]: 0 }));
          }
          return res;
        } catch (err) {
          console.error("Prediction failed:", err);
          await recoverSession();
          return null;
        } finally {
          setPredictInFlight((c) => c - 1);
        }
      };
      const next = predictQueueRef.current.then(run, run);
      predictQueueRef.current = next.catch(() => {});
      return next;
    },
    [setMasks, setObjectMasks, recoverSession, setPredictInFlight, setObjectUndoCounts, setObjectRedoCounts]
  );

  const undo = useCallback(async () => {
    if (!sessionId) return;
    if (predictInFlight !== 0) return;
    const oid = activeObjectIdRef.current;
    if ((objectUndoCountsRef.current[oid] ?? 0) <= 0) return;
    try {
      const res = await api.undo(sessionId, oid);
      setMasks(res.masks);
      setObjectMasks(res.objectMasks ?? {});
      setObjectUndoCounts((prev) => ({ ...prev, [oid]: Math.max(0, (prev[oid] ?? 0) - 1) }));
      setObjectRedoCounts((prev) => ({ ...prev, [oid]: (prev[oid] ?? 0) + 1 }));
    } catch {
      await recoverSession();
    }
  }, [sessionId, predictInFlight, setMasks, setObjectMasks, recoverSession, setObjectUndoCounts, setObjectRedoCounts]);

  const redo = useCallback(async () => {
    if (!sessionId) return;
    if (predictInFlight !== 0) return;
    const oid = activeObjectIdRef.current;
    if ((objectRedoCountsRef.current[oid] ?? 0) <= 0) return;
    try {
      const res = await api.redo(sessionId, oid);
      setMasks(res.masks);
      setObjectMasks(res.objectMasks ?? {});
      setObjectRedoCounts((prev) => ({ ...prev, [oid]: Math.max(0, (prev[oid] ?? 0) - 1) }));
      setObjectUndoCounts((prev) => ({ ...prev, [oid]: (prev[oid] ?? 0) + 1 }));
    } catch {
      await recoverSession();
    }
  }, [sessionId, predictInFlight, setMasks, setObjectMasks, recoverSession, setObjectRedoCounts, setObjectUndoCounts]);

  const clearObjectHistory = useCallback(async (objectId: number = 0) => {
    if (sessionId) {
      try {
        const res = await api.clearObject(sessionId, objectId);
        setMasks(res.masks);
        setObjectMasks(res.objectMasks ?? {});
      } catch {}
    }
    setObjectUndoCounts((prev) => {
      const next = { ...prev };
      delete next[objectId];
      return next;
    });
    setObjectRedoCounts((prev) => {
      const next = { ...prev };
      delete next[objectId];
      return next;
    });
  }, [sessionId, setMasks, setObjectMasks, setObjectUndoCounts, setObjectRedoCounts]);

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
  };
}
