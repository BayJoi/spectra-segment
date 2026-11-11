import { lazy, Suspense, useEffect, useState } from "react";
import { useAtom } from "jotai";
import { Header } from "@/components/Header";
import { Canvas } from "@/components/Canvas";
import { Toolbar } from "@/components/Toolbar";
import { MatteBar } from "@/components/MatteBar";
import { MatteExportDialog } from "@/components/MatteExportDialog";
import { ReplaceImageDialog } from "@/components/ReplaceImageDialog";
import { EmptyState } from "@/components/EmptyState";
import { ModeDialog } from "@/components/ModeDialog";
import { ModeSwitchDialog } from "@/components/ModeSwitchDialog";
import { EndSessionDialog } from "@/components/EndSessionDialog";
import { UnsupportedFileDialog } from "@/components/UnsupportedFileDialog";
import { EncodingOverlay } from "@/components/EncodingOverlay";
import { Sam3Page } from "@/components/Sam3Page";
import { ToastHost } from "@/components/Toast";
import { hasImageAtom, masksAtom, modelNameAtom, perDetectionMasksAtom } from "@/store/session";
import { uploadHoveredAtom, exportOpenAtom, settingsOpenAtom, modeDialogOpenAtom, modeSwitchTargetAtom, showTransparentAtom, endSessionOpenAtom, imageEncodingAtom, replaceImageFileAtom, unsupportedFileAtom } from "@/store/ui";
import { detectorsAtom, detectModeAtom, selectedDetectionAtom } from "@/store/detection";
import { matteExportOpenAtom, matteModeAtom } from "@/store/matte";
import { layersAtom } from "@/store/layers";
import { sam3ModeAtom, sam3InstancesAtom, selectedSam3InstanceAtom } from "@/store/sam3";
import { api } from "@/lib/api";
import { useSession } from "@/hooks/useSession";
import { useSessionGuard } from "@/hooks/useSessionGuard";
import { useLayers } from "@/hooks/useLayers";
import { useSam3 } from "@/hooks/useSam3";
import { useMatte } from "@/hooks/useMatte";
import { isSupportedImage } from "@/lib/utils";

const SettingsDialog = lazy(() => import("@/components/SettingsDialog").then((m) => ({ default: m.SettingsDialog })));
const ExportDialog = lazy(() => import("@/components/ExportDialog").then((m) => ({ default: m.ExportDialog })));

export function App() {
  const [hasImage] = useAtom(hasImageAtom);
  const [modelName] = useAtom(modelNameAtom);
  const [uploadHovered] = useAtom(uploadHoveredAtom);
  const [, setDetectors] = useAtom(detectorsAtom);
  const [detectMode] = useAtom(detectModeAtom);
  const [, setSelectedDetection] = useAtom(selectedDetectionAtom);
  const [layers] = useAtom(layersAtom);
  const [masks] = useAtom(masksAtom);
  const [perDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [showTransparent, setShowTransparent] = useAtom(showTransparentAtom);
  const [exportOpen, setExportOpen] = useAtom(exportOpenAtom);
  const [settingsOpen, setSettingsOpen] = useAtom(settingsOpenAtom);
  const [modeDialogOpen, setModeDialogOpen] = useAtom(modeDialogOpenAtom);
  const [modeSwitchTarget, setModeSwitchTarget] = useAtom(modeSwitchTargetAtom);
  const [endSessionOpen, setEndSessionOpen] = useAtom(endSessionOpenAtom);
  const [imageEncoding] = useAtom(imageEncodingAtom);
  const [replaceFile, setReplaceFile] = useAtom(replaceImageFileAtom);
  const [, setUnsupportedFile] = useAtom(unsupportedFileAtom);
  const [replaceDropActive, setReplaceDropActive] = useState(false);
  const [matteMode] = useAtom(matteModeAtom);
  const [matteExportOpen, setMatteExportOpen] = useAtom(matteExportOpenAtom);
  const { undo, redo, sessionId, recoverSession } = useSession();
  const { undo: matteUndo, redo: matteRedo } = useMatte();
  const { undo: sam3Undo, redo: sam3Redo, removeInstance: sam3RemoveInstance } = useSam3();
  const [sam3Mode] = useAtom(sam3ModeAtom);
  const [sam3Instances] = useAtom(sam3InstancesAtom);
  const [selectedSam3Instance] = useAtom(selectedSam3InstanceAtom);
  useSessionGuard({ recoverSession });
  const { removeLayer } = useLayers();

  const showVignette = uploadHovered && !modelName;
  const [vignetteBg, setVignetteBg] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!showVignette) {
      setVignetteBg(undefined);
      return;
    }
    const el = document.querySelector('[data-model-selector]');
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    setVignetteBg(`radial-gradient(ellipse 380px 220px at ${cx}px ${cy}px, transparent 0%, transparent 40%, rgba(0,0,0,0.8) 100%)`);
  }, [showVignette]);

  useEffect(() => {
    api.listDetectors().then(setDetectors).catch(() => {});
  }, [setDetectors]);

  useEffect(() => {
    if (showTransparent && masks.length === 0 && Object.keys(perDetectionMasks).length === 0) {
      setShowTransparent(false);
    }
  }, [showTransparent, masks, perDetectionMasks, setShowTransparent]);

  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (sessionId || ((layers.length > 0 || masks.length > 0) && hasImage)) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [sessionId, hasImage, layers.length, masks.length]);

  useEffect(() => {
    if (!hasImage || !sessionId) return;
    const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");
    const handleDragOver = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
      setReplaceDropActive(true);
    };
    const handleDragLeave = (e: DragEvent) => {
      if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
        setReplaceDropActive(false);
      }
    };
    const handleDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setReplaceDropActive(false);
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      if (!isSupportedImage(file)) {
        const dot = file.name.lastIndexOf(".");
        setUnsupportedFile({
          name: file.name,
          extension: dot > -1 ? file.name.slice(dot + 1).toLowerCase() : "",
        });
        return;
      }
      setReplaceFile(file);
    };
    window.addEventListener("dragover", handleDragOver);
    window.addEventListener("dragleave", handleDragLeave);
    window.addEventListener("drop", handleDrop);
    return () => {
      window.removeEventListener("dragover", handleDragOver);
      window.removeEventListener("dragleave", handleDragLeave);
      window.removeEventListener("drop", handleDrop);
      setReplaceDropActive(false);
    };
  }, [hasImage, sessionId, setReplaceFile, setUnsupportedFile]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        if (modeSwitchTarget) {
          setModeSwitchTarget(null);
          return;
        }
        const escTarget = e.target as HTMLElement;
        if (escTarget.tagName === "INPUT" || escTarget.tagName === "TEXTAREA") escTarget.blur();
        if (modeDialogOpen) {
          setModeDialogOpen(false);
          return;
        }
        if (replaceFile) {
          setReplaceFile(null);
          return;
        }
        if (endSessionOpen) {
          setEndSessionOpen(false);
          return;
        }
        if (settingsOpen) {
          setSettingsOpen(false);
          return;
        }
        if (exportOpen) {
          setExportOpen(false);
          return;
        }
        if (matteExportOpen) {
          setMatteExportOpen(false);
          return;
        }
        if (!hasImage || imageEncoding) return;
        setSelectedDetection(null);
        return;
      }
      if (!hasImage) return;
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;
      if (imageEncoding) return;

      if (e.key === "z" && e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        if (!e.repeat) {
          if (matteMode) matteUndo();
          else if (sam3Mode) sam3Undo();
          else undo();
        }
      } else if (e.key === "y" && e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        if (!e.repeat) {
          if (matteMode) matteRedo();
          else if (sam3Mode) sam3Redo();
          else redo();
        }
      } else if ((e.key === "Delete" || e.key === "Backspace") && !e.repeat) {
        e.preventDefault();
        if (sam3Mode) {
          if (selectedSam3Instance) {
            sam3RemoveInstance(selectedSam3Instance.promptIndex, selectedSam3Instance.instanceIndex);
          } else if (sam3Instances.length > 0) {
            const last = sam3Instances[sam3Instances.length - 1];
            sam3RemoveInstance(last.promptIndex, last.instanceIndex);
          }
        } else if (!detectMode && layers.length > 0) {
          removeLayer(layers[layers.length - 1].id);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hasImage, detectMode, layers, modeDialogOpen, modeSwitchTarget, replaceFile, endSessionOpen, settingsOpen, exportOpen, matteExportOpen, imageEncoding, undo, redo, matteUndo, matteRedo, matteMode, sam3Undo, sam3Redo, sam3Mode, sam3Instances, selectedSam3Instance, sam3RemoveInstance, removeLayer, setModeDialogOpen, setModeSwitchTarget, setReplaceFile, setEndSessionOpen, setSettingsOpen, setExportOpen, setMatteExportOpen]);

  useEffect(() => {
    const splash = document.getElementById("splash");
    if (splash) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          splash.classList.add("fade");
          splash.addEventListener("transitionend", () => splash.remove(), { once: true });
        });
      });
    }
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0a0a] text-white overflow-hidden select-none">
      <Header sam3={sam3Mode} />
      <main className="flex-1 relative min-h-0">
        {!hasImage ? (
          <div className="absolute inset-0">
            <EmptyState />
          </div>
        ) : sam3Mode ? (
          <Sam3Page />
        ) : (
          <div className="animate-fade-in absolute inset-0">
            <Canvas />
            {matteMode ? <MatteBar /> : <Toolbar />}
          </div>
        )}
      </main>
      {replaceDropActive && (
        <div className="fixed inset-0 z-[65] pointer-events-none flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="animate-drop-in flex flex-col items-center gap-3 px-8 py-7 rounded-2xl border-2 border-dashed border-orange-500/50 bg-[#0c0c0c]/90 shadow-2xl shadow-black/60 grain-bg">
            <div className="w-11 h-11 rounded-xl bg-orange-500/10 border border-orange-500/25 flex items-center justify-center">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className="text-orange-400">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21,15 16,10 5,21" />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-sm font-semibold text-neutral-100 font-sans">Drop to replace the image</p>
              <p className="text-[11px] text-neutral-500 font-sans mt-1">You will be asked to confirm - current masks and mattes will be cleared</p>
            </div>
          </div>
        </div>
      )}
      <Suspense fallback={null}>
        <SettingsDialog />
        <ExportDialog />
      </Suspense>
      <ModeDialog />
      <MatteExportDialog />
      <ReplaceImageDialog />
      <ModeSwitchDialog />
      <EndSessionDialog />
      <UnsupportedFileDialog />
      <EncodingOverlay />
      <ToastHost />
      <div
        className="fixed inset-0 z-50 pointer-events-none transition-opacity duration-200"
        style={{ background: vignetteBg, opacity: showVignette ? 1 : 0 }}
      />
    </div>
  );
}
