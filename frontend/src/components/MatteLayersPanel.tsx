import { useRef, type RefObject } from "react";
import { useAtom } from "jotai";
import { usePanelDismiss } from "@/hooks/usePanelDismiss";
import { matteActiveLayerIdAtom, matteLayersAtom, matteModelsAtom } from "@/store/matte";
import { useMatte } from "@/hooks/useMatte";
import { cn } from "@/lib/utils";

interface MatteLayersPanelProps {
  open: boolean;
  onClose: () => void;
  ignoredRef?: RefObject<HTMLElement | null>;
}

export function MatteLayersPanel({ open, onClose, ignoredRef }: MatteLayersPanelProps) {
  const [layers] = useAtom(matteLayersAtom);
  const [models] = useAtom(matteModelsAtom);
  const [activeId] = useAtom(matteActiveLayerIdAtom);
  const { addLayer, removeLayer, selectLayer, toggleVisible } = useMatte();
  const panelRef = useRef<HTMLDivElement>(null);

  usePanelDismiss(open, onClose, panelRef, ignoredRef);

  if (!open) return null;

  return (
    <div
      ref={panelRef}
      className="animate-drop-in w-80 max-h-80 flex flex-col bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-2xl shadow-2xl shadow-black/60 backdrop-blur-xl grain-bg overflow-hidden"
    >
      <div className="flex items-center justify-between px-2.5 py-2 border-b border-neutral-800/60">
        <div>
          <p className="text-[11px] font-semibold font-sans text-neutral-200">Subjects</p>
          <p className="text-[10px] font-sans text-neutral-500">Visible subjects are combined</p>
        </div>
        <span className="text-[10px] font-sans tabular-nums text-neutral-600">
          {layers.filter((layer) => layer.visible).length}/{layers.length}
        </span>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-1.5">
        <div className="flex flex-col gap-1">
          {layers.map((layer) => {
            const isActive = layer.id === activeId;
            const hasAlpha = layer.alphaB64 !== null;
            const modelLabel =
              models.find((item) => item.name === layer.model)?.display_name ?? layer.model;
            return (
              <div
                key={layer.id}
                role="button"
                tabIndex={0}
                onClick={() => selectLayer(layer.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectLayer(layer.id);
                  }
                }}
                className={cn(
                  "flex items-center gap-1.5 px-1.5 py-1.5 rounded-lg border cursor-pointer transition-all duration-150 select-none",
                  isActive
                    ? "bg-orange-500/10 border-orange-500/25"
                    : "border-transparent hover:bg-neutral-800/50 hover:border-neutral-700/30"
                )}
              >
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0 ring-1 ring-inset ring-white/10"
                  style={{ backgroundColor: layer.color, opacity: layer.visible ? 1 : 0.35 }}
                  aria-hidden
                />
                <div className={cn("min-w-0 flex-1", !layer.visible && "opacity-50")}>
                  <p className="text-[11px] font-sans text-neutral-200 truncate">{layer.name}</p>
                  <p className="text-[10px] font-sans text-neutral-600 truncate">
                    {modelLabel}
                    {" \u00b7 "}
                    {layer.strokes.points.length > 0
                      ? `${layer.strokes.points.length} points`
                      : "no strokes"}
                  </p>
                </div>
                <span
                  className={cn(
                    "w-1.5 h-1.5 rounded-full flex-shrink-0",
                    hasAlpha ? "bg-orange-400" : "bg-neutral-600"
                  )}
                  aria-label={hasAlpha ? "Cutout ready" : "No cutout yet"}
                />
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    toggleVisible(layer.id);
                  }}
                  aria-label={layer.visible ? `Hide ${layer.name}` : `Show ${layer.name}`}
                  aria-pressed={layer.visible}
                  className="w-6 h-6 rounded-md flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 transition-colors cursor-pointer"
                >
                  {layer.visible ? (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M9.9 4.24A9.1 9.1 0 0 1 12 4c6.5 0 10 7 10 7a13.2 13.2 0 0 1-2.16 2.9" />
                      <path d="M6.61 6.61A13.5 13.5 0 0 0 2 11s3.5 7 10 7a9.7 9.7 0 0 0 5.39-1.61" />
                      <path d="m2 2 20 20" />
                    </svg>
                  )}
                </button>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    removeLayer(layer.id);
                  }}
                  disabled={layers.length <= 1}
                  aria-label={`Remove ${layer.name}`}
                  className="w-6 h-6 rounded-md flex items-center justify-center text-neutral-600 hover:text-red-400 hover:bg-red-500/10 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M18 6 6 18" />
                    <path d="m6 6 12 12" />
                  </svg>
                </button>
              </div>
            );
          })}
        </div>

        <button
          onClick={() => addLayer()}
          className="mt-1.5 w-full h-7 rounded-lg border border-dashed border-neutral-700/70 text-[11px] font-sans text-neutral-400 hover:text-orange-200 hover:border-orange-500/40 hover:bg-orange-500/5 transition-colors cursor-pointer flex items-center justify-center gap-1.5"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 5v14" />
            <path d="M5 12h14" />
          </svg>
          Add subject
        </button>

        <p className="px-2 pt-2 pb-1 text-[10px] font-sans leading-relaxed text-neutral-600">
          Select a row to edit that subject. The eye hides it from the preview and export;
          visible cutouts are combined automatically.
        </p>
      </div>
    </div>
  );
}
