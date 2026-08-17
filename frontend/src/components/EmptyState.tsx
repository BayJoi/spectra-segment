import { useCallback, useRef, useState } from "react";
import { useAtom } from "jotai";
import { useSession } from "@/hooks/useSession";
import { modelNameAtom } from "@/store/session";
import { uploadHoveredAtom, modeSwitchTargetAtom } from "@/store/ui";
import { sam3ModeAtom } from "@/store/sam3";
import { Tooltip } from "@/components/ui/Tooltip";
import { cn } from "@/lib/utils";

export function EmptyState() {
  const { uploadImage } = useSession();
  const [modelName] = useAtom(modelNameAtom);
  const [sam3Mode] = useAtom(sam3ModeAtom);
  const [, setUploadHovered] = useAtom(uploadHoveredAtom);
  const [, setModeSwitchTarget] = useAtom(modeSwitchTargetAtom);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleToggleMode = useCallback(
    (target: "sam3" | "brush") => {
      setModeSwitchTarget(target);
    },
    [setModeSwitchTarget]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) uploadImage(file);
    },
    [uploadImage]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (!modelName && !sam3Mode) return;
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        uploadImage(file);
      }
    },
    [uploadImage, modelName, sam3Mode]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  return (
    <div
      className={`absolute inset-0 z-10 flex flex-col items-center justify-center gap-5 transition-all duration-200 ${isDragOver ? "bg-orange-500/5 border-2 border-dashed border-orange-500/30 rounded-xl" : ""}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <div className="hero-entrance w-16 h-16 rounded-2xl bg-gradient-to-br from-neutral-900 to-neutral-800 border border-neutral-800/60 flex items-center justify-center shadow-2xl shadow-black/40 hover:scale-110 hover:shadow-orange-500/10 transition-all duration-300 grain-bg">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" className="text-neutral-500">
          <rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21,15 16,10 5,21" />
        </svg>
      </div>

      <div className="text-center hero-entrance hero-delay-2">
        <p className="text-neutral-300 text-sm font-medium font-sans">Add an image to get started</p>
        <p className="text-neutral-600 text-xs mt-1.5 font-sans">
          {!modelName && !sam3Mode
            ? "Choose a mode, then drop or upload an image"
            : sam3Mode
              ? "Type a text prompt to segment every matching object — e.g. \"person\", \"car\", \"red bottle\""
              : isDragOver
                ? "Release to upload"
                : "Drop an image here or click Upload"}
        </p>
      </div>

      <div className="hero-entrance hero-delay-2 inline-flex items-center gap-1 p-1 rounded-xl border border-neutral-800/60 bg-neutral-900/50">
        <Tooltip tip="Brush / Detect mode (B/D)" side="bottom">
          <button
            onClick={() => handleToggleMode("brush")}
            className={cn(
              "inline-flex items-center gap-1.5 h-8 px-3.5 rounded-lg text-[11px] font-medium font-sans transition-all duration-200 cursor-pointer select-none grain-bg",
              !sam3Mode
                ? "bg-neutral-800 text-neutral-100 border border-neutral-700 shadow"
                : "text-neutral-500 hover:text-neutral-300 border border-transparent hover:scale-105 active:scale-95"
            )}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="10" />
            </svg>
            B/D Mode
          </button>
        </Tooltip>
        <Tooltip tip="Text-based SAM 3 segmentation" side="bottom">
          <button
            onClick={() => handleToggleMode("sam3")}
            className={cn(
              "inline-flex items-center gap-1.5 h-8 px-3.5 rounded-lg text-[11px] font-semibold font-sans transition-all duration-200 cursor-pointer select-none grain-bg",
              sam3Mode
                ? "bg-gradient-to-r from-orange-500 to-orange-600 text-white border border-orange-500 shadow-lg shadow-orange-500/20 grain-bg-strong"
                : "text-neutral-500 hover:text-orange-300 border border-transparent hover:scale-105 active:scale-95"
            )}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 7h16" /><path d="M4 12h10" /><path d="M4 17h13" />
            </svg>
            Text · SAM 3
          </button>
        </Tooltip>
      </div>

      <Tooltip tip={!modelName && !sam3Mode ? "Choose a mode first" : "Upload image"} side="bottom">
        <span
          onMouseEnter={() => !modelName && !sam3Mode && setUploadHovered(true)}
          onMouseLeave={() => setUploadHovered(false)}
          className="hero-entrance hero-delay-2 mt-1 inline-flex"
        >
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!modelName && !sam3Mode}
            className={cn(
              "inline-flex items-center justify-center gap-2 h-10 px-5 rounded-xl text-sm font-medium font-sans transition-all duration-200 select-none grain-bg grain-bg-strong",
              !modelName && !sam3Mode
                ? "bg-neutral-800/50 border border-neutral-800/50 text-neutral-600 cursor-not-allowed opacity-50"
                : "bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-105 active:scale-95 cursor-pointer"
            )}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Upload Image
          </button>
        </span>
      </Tooltip>
      <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleFileChange} className="hidden" />
    </div>
  );
}
