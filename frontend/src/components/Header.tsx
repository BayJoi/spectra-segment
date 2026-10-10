import { useCallback, useEffect, useRef, useState } from "react";
import { useAtom } from "jotai";
import { SelectNative } from "@/components/ui/Select";
import { Tooltip } from "@/components/ui/Tooltip";
import { ConsolePanel } from "@/components/ConsolePanel";
import { TierIcon } from "@/components/ui/TierIcon";
import { imageWidthAtom, imageHeightAtom, hasImageAtom, masksAtom, modelNameAtom, perDetectionMasksAtom, sam3ReadyAtom, modelsAtom, type ModelInfo } from "@/store/session";
import { showTransparentAtom, settingsOpenAtom, uploadHoveredAtom, endSessionOpenAtom, imageEncodingAtom, pushToast } from "@/store/ui";
import { loadedDetectorAtom, detectorsAtom } from "@/store/detection";
import { useSession } from "@/hooks/useSession";
import { api, BASE } from "@/lib/api";
import { connectSse } from "@/lib/sse";
import { logErr } from "@/store/logs";
import { cn } from "@/lib/utils";

export function Header({ sam3 = false }: { sam3?: boolean }) {
  const { sessionId, switchModel, uploadImage } = useSession();
  const [imageWidth] = useAtom(imageWidthAtom);
  const [imageHeight] = useAtom(imageHeightAtom);
  const [hasImage] = useAtom(hasImageAtom);
  const [masks] = useAtom(masksAtom);
  const [perDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [showTransparent, setShowTransparent] = useAtom(showTransparentAtom);
  const [, setSettingsOpen] = useAtom(settingsOpenAtom);
  const [, setEndSessionOpen] = useAtom(endSessionOpenAtom);
  const [uploadHovered, setUploadHovered] = useAtom(uploadHoveredAtom);
  const [imageEncoding] = useAtom(imageEncodingAtom);
  const [, setSam3Ready] = useAtom(sam3ReadyAtom);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [models, setModels] = useAtom(modelsAtom);
  const [modelName] = useAtom(modelNameAtom);
  const [, setLoadedDetector] = useAtom(loadedDetectorAtom);
  const [, setDetectors] = useAtom(detectorsAtom);
  const [loadingLabel, setLoadingLabel] = useState<string>("");
  const [unloading, setUnloading] = useState(false);

  const unloadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);


  const segModels = models.filter((m) => m.type === "segment");
  const hasAnyDownloaded = models.some((m) => m.downloaded);
  const showHighlight = uploadHovered && !modelName;

  const syncDetectors = useCallback(
    (modelList: ModelInfo[]) => {
      setDetectors(
        modelList
          .filter((m) => m.type === "detector")
          .map((m) => ({
            name: m.name,
            display_name: m.display_name,
            type: (m.detector_type === "yoloe" ? "yoloe" : "grounding") as "grounding" | "yoloe",
            loaded: m.loaded,
            downloaded: m.downloaded,
            tier: m.tier,
            perf: m.perf,
          }))
      );
    },
    [setDetectors]
  );

  useEffect(() => {
    const applyModelList = (list: ModelInfo[]) => {
      setModels(list);
      syncDetectors(list);
      const loadedDet = list.find((m) => m.type === "detector" && m.loaded);
      setLoadedDetector(loadedDet ? loadedDet.name : null);
    };

    api
      .getModels()
      .then((res) => applyModelList(res.models))
      .catch((err) => logErr("detect", err));

    const handle = connectSse(`${BASE}/api/models/stream`, (data) => {
      // Residency transitions are already logged by the backend, so this stream
      // only drives the header UI - mirroring it here would duplicate every line.
      try {
        const parsed = JSON.parse(data);
        if (parsed.models) applyModelList(parsed.models);
        if (parsed.event === "unloading") {
          setLoadingLabel("Unloading model...");
          setUnloading(true);
          if (unloadTimerRef.current) window.clearTimeout(unloadTimerRef.current);
          unloadTimerRef.current = window.setTimeout(() => {
            setUnloading(false);
            setLoadingLabel("");
          }, 3000);
        } else if (parsed.event === "downloading") {
          setLoadingLabel("Downloading model...");
          setDownloading(true);
          setUnloading(true);
          if (unloadTimerRef.current) window.clearTimeout(unloadTimerRef.current);
          unloadTimerRef.current = window.setTimeout(() => {
            setUnloading(false);
            setDownloading(false);
            setLoadingLabel("");
          }, 600000);
        } else if (parsed.event === "loading") {
          setLoadingLabel("Loading model...");
          setDownloading(false);
          setUnloading(true);
          if (unloadTimerRef.current) window.clearTimeout(unloadTimerRef.current);
          unloadTimerRef.current = window.setTimeout(() => {
            setUnloading(false);
            setLoadingLabel("");
          }, 30000);
        } else if (parsed.event === "loaded") {
          if (unloadTimerRef.current) window.clearTimeout(unloadTimerRef.current);
          setUnloading(false);
          setDownloading(false);
          setLoadingLabel("");
        }
      } catch (err) {
        logErr("detect", err);
      }
    });

    return () => {
      handle.close();
      if (unloadTimerRef.current) {
        window.clearTimeout(unloadTimerRef.current);
        unloadTimerRef.current = null;
      }
    };
  }, [syncDetectors, setModels, setLoadedDetector]);

  const prevLoadingRef = useRef(false);
  useEffect(() => {
    const wasLoading = prevLoadingRef.current;
    prevLoadingRef.current = modelLoading || downloading;
    if (wasLoading && !modelLoading && !downloading) {
      api.getModels().then((res) => {
        setModels(res.models);
        syncDetectors(res.models);
      }).catch((err) => logErr("mode", err));
    }
  }, [modelLoading, downloading, syncDetectors]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) uploadImage(file);
    },
    [uploadImage]
  );

  const handleModelSwitch = useCallback(async (model: string) => {
    if (model === modelName) return;
    const target = models.find((m) => m.name === model);
    const isDownloaded = target?.downloaded ?? false;
    setModelLoading(true);
    setLoadingLabel(isDownloaded ? (sessionId ? "Switching model..." : "Loading model...") : "Downloading model...");
    if (!isDownloaded) setDownloading(true);
    try {
      await switchModel(model);
    } catch (err) {
      logErr("session", err);
      pushToast("Model switch failed");
    } finally {
      setModelLoading(false);
      setDownloading(false);
    }
  }, [switchModel, models, sessionId, modelName]);

  const sam3Loaded = models.some((m) => m.type === "sam3" && m.loaded);

  useEffect(() => {
    setSam3Ready(sam3Loaded);
  }, [sam3Loaded, setSam3Ready]);

  const sourceModels = sam3 ? models.filter((m) => m.type === "sam3") : segModels;
  const modelOptions = sourceModels.map((m) => ({
    value: m.name,
    label: m.downloaded ? `\u2713 ${m.display_name}` : m.display_name,
    icon: sam3 ? undefined : <TierIcon tier={m.tier} />,
    hint: sam3 ? undefined : m.perf,
    loaded: m.name === modelName,
  }));

  return (
    <header className={`grain-bg animate-fade-in h-14 flex items-center justify-between px-5 border-b border-neutral-800/60 bg-[#0a0a0a]/95 ${imageEncoding ? "z-[80]" : "z-20"} shrink-0 backdrop-blur-xl relative`}>
      {imageEncoding && (
        <div className="absolute inset-0 z-[75] bg-black/60 backdrop-blur-sm" />
      )}
      <div className="flex items-center gap-3">
        <Tooltip tip={sessionId ? "End session and return home" : "Home"} side="bottom">
          <button
            type="button"
            onClick={() => sessionId && setEndSessionOpen(true)}
            aria-label={sessionId ? "End session and return home" : "Spectra Segment home"}
            className={cn("group flex items-center gap-2.5 bg-transparent border-0 p-0 text-left cursor-pointer transition-opacity duration-200", showHighlight && "opacity-30")}
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center shadow-lg shadow-orange-500/20 group-hover:scale-110 group-hover:shadow-orange-500/30 transition-all duration-300">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
                <circle cx="12" cy="12" r="10" /><path d="M8 12l3 3 5-5" />
              </svg>
            </div>
            <span className="text-lg font-bold tracking-tight text-neutral-100 font-sans group-hover:text-orange-300 transition-colors duration-200">
              Spectra Segment
            </span>
          </button>
        </Tooltip>
        <div className={cn("w-px h-5 bg-neutral-800/60 transition-opacity duration-200", showHighlight && "opacity-30")} />
        <div data-model-selector className={cn(
            "relative z-10 rounded-lg transition-all duration-200",
            !sam3 && "w-[190px]",
            showHighlight && !sam3 && "animate-model-highlight animate-model-highlight-idle"
          )}>
            {!sam3 && showHighlight && (
              <div className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none">
                <div className="shimmer-overlay" />
              </div>
            )}
            <SelectNative
              value={modelName}
              onChange={handleModelSwitch}
              options={modelOptions}
              placeholder={sam3 ? "Choose a SAM 3 model..." : "Choose a model..."}
              className={cn(modelLoading && "animate-model-load pointer-events-none")}
            />
            <div
              className={cn(
                "absolute inset-0 flex items-center justify-center bg-[#0d0d0d] backdrop-blur-sm overflow-hidden rounded-lg z-10 pointer-events-none grain-bg transition-opacity duration-200",
                modelLoading || downloading || unloading ? "opacity-100" : "opacity-0"
              )}
            >
              <div className="shimmer-overlay" />
              <span className="text-[10px] text-orange-400 font-mono animate-pulse relative z-10">
                {loadingLabel}
              </span>
            </div>
          </div>
        {!sam3 && !modelName && !modelLoading && segModels.length > 0 && (
          <div className={cn("flex items-center gap-2 ml-1 transition-opacity duration-200", showHighlight && "opacity-100")}>
            <span className={cn(
              "text-xs font-sans whitespace-nowrap select-none",
              hasAnyDownloaded
                ? showHighlight ? "text-orange-300" : "text-neutral-400"
                : "text-orange-400 animate-pulse-glow font-medium"
            )}>
              {hasAnyDownloaded ? "Select a model" : "Select a model to download and start"}
            </span>
            <div className={cn("hidden lg:flex items-center gap-2 text-[9px] text-neutral-600 font-mono ml-1 transition-opacity duration-200", showHighlight && "opacity-40")}>
              <span className="flex items-center gap-1"><TierIcon tier="tiny" /> fast</span>
              <span className="flex items-center gap-1"><TierIcon tier="small" /> balanced</span>
              <span className="flex items-center gap-1"><TierIcon tier="medium" /> accurate</span>
              <span className="flex items-center gap-1"><TierIcon tier="large" /> best</span>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-1.5">
        {hasImage && (
          <span className={cn("text-[11px] text-neutral-500 mr-2 tabular-nums font-mono bg-neutral-900/50 px-2 py-1 rounded-lg border border-neutral-800/40 transition-opacity duration-200", showHighlight && "opacity-30")}>
            {imageWidth}×{imageHeight}
          </span>
        )}
        {hasImage && (masks.length > 0 || Object.keys(perDetectionMasks).length > 0) && (
          <Tooltip tip="Toggle transparent background" side="bottom">
            <button
              onClick={() => setShowTransparent((v) => !v)}
              className={cn(
                "inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans transition-all duration-200 cursor-pointer select-none",
                "border hover:scale-105 active:scale-95",
                showTransparent
                  ? "bg-orange-500/15 border-orange-500/30 text-orange-300 shadow-[0_0_12px_rgba(184,92,42,0.15)]"
                  : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700"
              )}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 3l18 18" />
              </svg>
              Transparent
            </button>
          </Tooltip>
        )}
        <ConsolePanel encoding={imageEncoding} />
        <Tooltip tip="Keyboard shortcuts and preferences" side="bottom">
          <button
            onClick={() => setSettingsOpen(true)}
            className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans bg-neutral-900/50 border border-neutral-800/50 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700 hover:bg-neutral-800/50 transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer select-none"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42m12.72-12.72 1.42-1.42"/></svg>
            Settings
          </button>
        </Tooltip>
        {sessionId && (
          <Tooltip tip="End this session and unload the model" side="bottom">
            <button
              onClick={() => setEndSessionOpen(true)}
              className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans bg-neutral-900/50 border border-neutral-800/50 text-neutral-500 hover:text-red-300 hover:border-red-900/40 hover:bg-red-950/20 transition-all duration-200 hover:scale-105 active:scale-95 cursor-pointer select-none"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
              End Session
            </button>
          </Tooltip>
        )}
        <Tooltip tip={!modelName && !sam3 ? "Select a model first" : sam3 && !modelName ? "Select a SAM 3 model first" : sam3 && !sam3Loaded ? "Loading SAM 3 model…" : hasImage ? "Replace image" : "Upload image"} side="bottom">
          <span
            onMouseEnter={() => !modelName && !sam3 && setUploadHovered(true)}
            onMouseLeave={() => setUploadHovered(false)}
            className="inline-flex"
          >
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={sam3 ? !sam3Loaded : !modelName}
              className={cn(
                "inline-flex items-center justify-center gap-1.5 h-7 px-2.5 rounded-lg text-[11px] font-medium font-sans transition-all duration-200 select-none grain-bg grain-bg-strong",
                (sam3 ? !sam3Loaded : !modelName)
                  ? "bg-neutral-800/50 border border-neutral-800/50 text-neutral-600 cursor-not-allowed opacity-50"
                  : "bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-105 active:scale-95 cursor-pointer"
              )}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              {hasImage ? "Replace" : "Upload"}
            </button>
          </span>
        </Tooltip>
        <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleFileChange} className="hidden" />
      </div>
    </header>
  );
}
