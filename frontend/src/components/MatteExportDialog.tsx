import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { applyAlphaControls, mergeAlphaCanvases, type MatteLayer } from "@/lib/matte";
import { hexToRgb, loadImageElement, saveBlob, slugify } from "@/lib/fileExport";
import { cn } from "@/lib/utils";
import { emitUi } from "@/store/logs";
import { imageFileAtom, imageUrlAtom } from "@/store/session";
import {
  matteActiveLayerIdAtom,
  matteEdgeFeatherAtom,
  matteEdgeShiftAtom,
  matteExportOpenAtom,
  matteInvertAtom,
  matteLayersAtom,
} from "@/store/matte";
import { pushToast } from "@/store/ui";

type ExportTarget = "combined" | "active" | "separate";

function alphaFor(layer: MatteLayer, shift: number, feather: number, invert: boolean): HTMLCanvasElement | null {
  if (!layer.rawAlpha) return null;
  return applyAlphaControls(layer.rawAlpha.canvas, shift, feather, invert);
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the image"))),
      type,
      0.95
    );
  });
}

async function composeImage(
  image: HTMLImageElement,
  alpha: HTMLCanvasElement,
  format: "png" | "jpg",
  solidColor: [number, number, number] | null
): Promise<Blob> {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const foreground = document.createElement("canvas");
  foreground.width = width;
  foreground.height = height;
  const fgCtx = foreground.getContext("2d");
  if (!fgCtx) throw new Error("2D canvas unavailable");
  fgCtx.drawImage(image, 0, 0, width, height);
  fgCtx.globalCompositeOperation = "destination-in";
  fgCtx.drawImage(alpha, 0, 0, width, height);

  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const ctx = output.getContext("2d");
  if (!ctx) throw new Error("2D canvas unavailable");
  if (solidColor) {
    ctx.fillStyle = `rgb(${solidColor[0]}, ${solidColor[1]}, ${solidColor[2]})`;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.drawImage(foreground, 0, 0);
  return canvasToBlob(output, format === "jpg" ? "image/jpeg" : "image/png");
}

export function MatteExportDialog() {
  const [open, setOpen] = useAtom(matteExportOpenAtom);
  const [imageFile] = useAtom(imageFileAtom);
  const [imageUrl] = useAtom(imageUrlAtom);
  const [layers] = useAtom(matteLayersAtom);
  const [activeLayerId] = useAtom(matteActiveLayerIdAtom);
  const [edgeShift] = useAtom(matteEdgeShiftAtom);
  const [edgeFeather] = useAtom(matteEdgeFeatherAtom);
  const [invert] = useAtom(matteInvertAtom);
  const [format, setFormat] = useState<"png" | "jpg">("png");
  const [bgMode, setBgMode] = useState<"transparent" | "solid">("transparent");
  const [bgColor, setBgColor] = useState("#ffffff");
  const [target, setTarget] = useState<ExportTarget>("combined");
  const [busy, setBusy] = useState(false);
  const [exported, setExported] = useState(false);
  const exportedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const visibleLayers = useMemo(
    () => layers.filter((layer) => layer.visible && layer.rawAlpha !== null),
    [layers]
  );
  const activeLayer = visibleLayers.find((layer) => layer.id === activeLayerId) ?? null;
  const readyCount = layers.filter((layer) => layer.rawAlpha !== null).length;
  const hiddenCount = readyCount - visibleLayers.length;
  const base = (imageFile?.name ?? "image").replace(/\.[^.]+$/, "") || "image";
  const solid = format === "jpg" || bgMode === "solid" ? hexToRgb(bgColor) : null;

  useEffect(() => {
    if (target === "active" && !activeLayer) setTarget("combined");
  }, [target, activeLayer]);

  useEffect(() => {
    return () => {
      if (exportedTimerRef.current) clearTimeout(exportedTimerRef.current);
    };
  }, []);

  const handleClose = useCallback(() => {
    setOpen(false);
    setExported(false);
    if (exportedTimerRef.current) clearTimeout(exportedTimerRef.current);
  }, [setOpen]);

  const handleExport = useCallback(async () => {
    if (!imageUrl || busy || visibleLayers.length === 0) return;
    setBusy(true);
    setExported(false);
    try {
      const source = await loadImageElement(imageUrl);
      const prepared = visibleLayers
        .map((layer) => ({ layer, alpha: alphaFor(layer, edgeShift, edgeFeather, invert) }))
        .filter((item): item is { layer: MatteLayer; alpha: HTMLCanvasElement } => item.alpha !== null);

      if (prepared.length === 0) {
        pushToast("Nothing to export - make a cutout first");
        return;
      }

      if (target === "active") {
        const item = prepared.find((entry) => entry.layer.id === activeLayerId);
        if (!item) {
          pushToast("The selected subject is hidden");
          return;
        }
        const blob = await composeImage(source, item.alpha, format, solid);
        saveBlob(blob, `${base}_${slugify(item.layer.name, "cutout")}.${format}`);
        emitUi("export", "INFO", `Saved ${item.layer.name}`);
      } else if (target === "combined") {
        const alpha = mergeAlphaCanvases(prepared.map((entry) => entry.alpha));
        if (!alpha) throw new Error("Could not combine the matte layers");
        const blob = await composeImage(source, alpha, format, solid);
        saveBlob(blob, `${base}_cutout.${format}`);
        emitUi("export", "INFO", `Saved one cutout with ${prepared.length} subject${prepared.length === 1 ? "" : "s"}`);
      } else {
        for (const item of prepared) {
          const blob = await composeImage(source, item.alpha, format, solid);
          saveBlob(blob, `${base}_${slugify(item.layer.name, "cutout")}.${format}`);
          await new Promise((resolve) => setTimeout(resolve, 120));
        }
        emitUi("export", "INFO", `Saved ${prepared.length} separate cutout${prepared.length === 1 ? "" : "s"}`);
      }

      setExported(true);
      if (exportedTimerRef.current) clearTimeout(exportedTimerRef.current);
      exportedTimerRef.current = setTimeout(() => setExported(false), 2200);
    } catch (err) {
      emitUi("export", "ERROR", `Matte export failed: ${err instanceof Error ? err.message : "unknown error"}`);
      pushToast("Export failed - please try again");
    } finally {
      setBusy(false);
    }
  }, [
    imageUrl,
    busy,
    visibleLayers,
    edgeShift,
    edgeFeather,
    invert,
    target,
    activeLayerId,
    format,
    solid,
    base,
  ]);

  if (!open) return null;

  const targets: { key: ExportTarget; title: string; desc: string; disabled: boolean }[] = [
    {
      key: "combined",
      title: "Combined",
      desc: `${visibleLayers.length} subject${visibleLayers.length === 1 ? "" : "s"}`,
      disabled: visibleLayers.length === 0,
    },
    {
      key: "active",
      title: "Active",
      desc: activeLayer?.name ?? "Hidden",
      disabled: !activeLayer,
    },
    {
      key: "separate",
      title: "Separate",
      desc: `${visibleLayers.length} file${visibleLayers.length === 1 ? "" : "s"}`,
      disabled: visibleLayers.length === 0,
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Export matte">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={handleClose}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-md mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-orange-500/20 via-orange-500/5 to-transparent pointer-events-none" />

        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500/20 to-orange-600/10 border border-orange-500/20 flex items-center justify-center shadow-lg shadow-orange-500/10">
                  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400">
                    <path d="M12 3v18" />
                    <path d="M4 8.5A8.5 8.5 0 0 1 12 3v18a8.5 8.5 0 0 1-8-8.5Z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">Export Matte</h3>
                  <p className="text-[10px] text-neutral-500 font-sans">
                    {base}.{format}
                  </p>
                </div>
              </div>
              <button
                onClick={handleClose}
                aria-label="Close export dialog"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M18 6 6 18" />
                  <path d="m6 6 12 12" />
                </svg>
              </button>
            </div>

            <div className="mb-3 flex items-center justify-between rounded-lg border border-neutral-800/60 bg-neutral-900/35 px-2.5 py-1.5">
              <span className="text-[10px] font-sans text-neutral-400">
                {visibleLayers.length} of {readyCount} cutout{readyCount === 1 ? "" : "s"} visible
              </span>
              <span className="text-[9px] font-sans text-neutral-600">
                {hiddenCount > 0 ? `${hiddenCount} hidden - not exported` : "All cutouts included"}
              </span>
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">
                What to export
              </label>
              <div className="grid grid-cols-3 gap-2">
                {targets.map((option) => (
                  <button
                    key={option.key}
                    onClick={() => setTarget(option.key)}
                    disabled={option.disabled}
                    className={cn(
                      "px-2 py-2 rounded-xl border text-left transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                      target === option.key
                        ? "bg-orange-500/10 border-orange-500/30 text-orange-200 shadow-[0_0_20px_rgba(184,92,42,0.1)]"
                        : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300 hover:bg-neutral-800/30",
                      option.disabled && "opacity-40 cursor-not-allowed hover:scale-100"
                    )}
                  >
                    <span className="block text-[11px] font-semibold font-sans">{option.title}</span>
                    <span className="block text-[9px] text-neutral-600 font-normal mt-0.5 leading-tight truncate">
                      {option.desc}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-3 grid grid-cols-2 gap-2">
              <div>
                <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">
                  Format
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {(["png", "jpg"] as const).map((value) => (
                    <button
                      key={value}
                      onClick={() => {
                        setFormat(value);
                        if (value === "jpg") setBgMode("solid");
                      }}
                      className={cn(
                        "h-8 rounded-lg border text-[11px] font-medium font-sans transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                        format === value
                          ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                          : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
                      )}
                    >
                      {value.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">
                  Background
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => {
                      setBgMode("transparent");
                      if (format === "jpg") setFormat("png");
                    }}
                    disabled={format === "jpg"}
                    className={cn(
                      "h-8 rounded-lg border text-[10px] font-medium font-sans transition-all duration-200 cursor-pointer select-none",
                      bgMode === "transparent"
                        ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                        : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300",
                      format === "jpg" && "opacity-40 cursor-not-allowed"
                    )}
                  >
                    Clear
                  </button>
                  <button
                    onClick={() => setBgMode("solid")}
                    className={cn(
                      "h-8 rounded-lg border text-[10px] font-medium font-sans transition-all duration-200 cursor-pointer select-none",
                      bgMode === "solid"
                        ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                        : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
                    )}
                  >
                    Solid
                  </button>
                </div>
              </div>
            </div>

            {bgMode === "solid" && (
              <div className="animate-drop-in mb-3 flex items-center gap-3 rounded-lg border border-neutral-800/60 bg-neutral-900/35 px-2.5 py-2">
                <input
                  type="color"
                  value={bgColor}
                  onChange={(event) => setBgColor(event.target.value)}
                  aria-label="Background colour"
                  className="w-7 h-7 rounded-lg border border-neutral-800 cursor-pointer hover:border-neutral-700 hover:scale-105 transition-all duration-200"
                />
                <span className="text-[11px] text-neutral-500 font-sans tabular-nums">{bgColor}</span>
                <span className="ml-auto text-[9px] text-neutral-600 font-sans">Solid background</span>
              </div>
            )}

            <button
              onClick={() => void handleExport()}
              disabled={busy || visibleLayers.length === 0}
              className={cn(
                "w-full h-10 rounded-xl font-medium text-sm font-sans transition-all duration-200 cursor-pointer select-none",
                "flex items-center justify-center gap-2",
                exported
                  ? "bg-green-500/15 border border-green-500/30 text-green-400"
                  : "bg-gradient-to-r from-orange-500 to-orange-600 text-white border border-orange-500/40 hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-[1.01] active:scale-[0.99] grain-bg grain-bg-strong",
                (busy || visibleLayers.length === 0) && "opacity-60 cursor-not-allowed hover:scale-100"
              )}
            >
              {busy ? (
                <>
                  <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                  Exporting...
                </>
              ) : exported ? (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M20 6 9 17l-5-5" />
                  </svg>
                  Downloaded
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7,10 12,15 17,10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  {target === "separate"
                    ? `Download ${visibleLayers.length} image${visibleLayers.length === 1 ? "" : "s"}`
                    : target === "active"
                      ? `Download ${activeLayer?.name ?? "active subject"}`
                      : "Download combined cutout"}
                </>
              )}
            </button>

            <p className="mt-2 text-center text-[9px] font-sans text-neutral-600">
              Composited locally. Hidden subjects stay out of the export.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
