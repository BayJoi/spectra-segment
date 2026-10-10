import { useRef, useEffect, useMemo, type RefObject } from "react";
import { useAtom, useSetAtom } from "jotai";
import { useLayers } from "@/hooks/useLayers";
import { perDetectionMasksAtom } from "@/store/session";
import { detectionsAtom, selectedDetectionAtom } from "@/store/detection";
import { selectedSam3InstanceAtom } from "@/store/sam3";
import type { Layer } from "@/store/layers";
import { cn } from "@/lib/utils";

interface LayersPanelProps {
  open: boolean;
  onClose: () => void;
  ignoredRef?: RefObject<HTMLElement | null>;
  onRemoveSam3Instance?: (promptIndex: number, instanceIndex: number) => void;
}

function baseLabel(label: string): string {
  return label.replace(/\s+\d+%$/, "").trim();
}

export function LayersPanel({ open, onClose, ignoredRef, onRemoveSam3Instance }: LayersPanelProps) {
  const { layers, selectedLayers, selectLayer, removeLayer } = useLayers();
  const [perDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [detections] = useAtom(detectionsAtom);
  const setSelectedDetection = useSetAtom(selectedDetectionAtom);
  const [selectedSam3Instance, setSelectedSam3Instance] = useAtom(selectedSam3InstanceAtom);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (panelRef.current && !panelRef.current.contains(target) && !ignoredRef?.current?.contains(target)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, onClose, ignoredRef]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const detectionGroups = useMemo(() => {
    const groups: Record<string, typeof layers> = {};
    for (const l of layers) {
      if (l.type === "sam3") continue;
      const key = baseLabel(l.label);
      if (!groups[key]) groups[key] = [];
      groups[key].push(l);
    }
    return groups;
  }, [layers]);

  const sam3Layers = useMemo(
    () => layers.filter((l): l is Layer & { promptIndex: number; instanceIndex: number } =>
      l.type === "sam3" && l.promptIndex !== undefined && l.instanceIndex !== undefined
    ),
    [layers]
  );

  if (!open || layers.length === 0) return null;

  return (
    <div
      ref={panelRef}
      className="animate-drop-in w-72 max-h-72 flex flex-col bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-xl shadow-2xl shadow-black/60 backdrop-blur-xl grain-bg overflow-hidden"
    >
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-2">
      <div className="flex items-center justify-between px-1 mb-1.5">
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider font-sans">Detections</span>
        <span className="text-[10px] text-neutral-600 font-mono">
          {detections.length} found · {Object.keys(perDetectionMasks).length} masked
        </span>
      </div>

      {Object.entries(detectionGroups).map(([groupLabel, groupLayers]) => (
        <div key={groupLabel}>
          <div className="flex items-center gap-1.5 px-1 mt-2 mb-1">
            <span className="text-[10px] text-neutral-500 uppercase tracking-wider font-sans font-semibold">{groupLabel}</span>
            <span className="text-[9px] text-neutral-600 font-mono">{groupLayers.length}</span>
            <div className="flex-1 h-px bg-neutral-800/60" />
          </div>
          {groupLayers.map((layer) => {
            const isSelected = selectedLayers.has(layer.id);
            const hasMask = layer.detectionIndex !== undefined && layer.detectionIndex in perDetectionMasks;
            return (
              <div
                key={layer.id}
                role="button"
                tabIndex={0}
                className={cn(
                  "reveal flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all duration-150 select-none group",
                  isSelected
                    ? "bg-green-500/10 border border-green-500/20 shadow-[0_0_8px_rgba(0,255,136,0.08)]"
                    : "border border-transparent hover:bg-neutral-800/50 hover:border-neutral-700/30"
                )}
                onClick={(e) => {
                  selectLayer(layer.id, e.ctrlKey || e.metaKey);
                  if (layer.detectionIndex !== undefined) {
                    setSelectedDetection(layer.detectionIndex);
                  } else {
                    setSelectedDetection(null);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    selectLayer(layer.id, e.ctrlKey || e.metaKey);
                    if (layer.detectionIndex !== undefined) {
                      setSelectedDetection(layer.detectionIndex);
                    } else {
                      setSelectedDetection(null);
                    }
                  }
                }}
              >
                <span
                  className="text-[11px] text-neutral-300 font-sans truncate flex-1"
                  title={layer.detectionIndex !== undefined && detections[layer.detectionIndex] ? `${Math.round(detections[layer.detectionIndex].score * 100)}% confidence` : undefined}
                >
                  {layer.detectionIndex !== undefined && detections[layer.detectionIndex]
                    ? `#${layer.detectionIndex + 1} ${detections[layer.detectionIndex].label} · ${Math.round(detections[layer.detectionIndex].score * 100)}%`
                    : layer.label}
                </span>
                <span className={cn(
                  "w-1.5 h-1.5 rounded-full flex-shrink-0",
                  hasMask ? "bg-green-400" : "bg-neutral-600"
                )} />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeLayer(layer.id);
                  }}
                  aria-label="Remove layer"
                  className="w-6 h-6 min-w-[24px] min-h-[24px] rounded flex items-center justify-center text-neutral-600 hover:text-red-400 hover:bg-red-500/10 opacity-100 transition-all duration-150 hover:scale-110 cursor-pointer"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                </button>
              </div>
            );
          })}
        </div>
      ))}

      {sam3Layers.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 px-1 mt-2 mb-1">
            <span className="text-[10px] text-orange-400/90 uppercase tracking-wider font-sans font-semibold">SAM3 Segments</span>
            <span className="text-[9px] text-neutral-600 font-mono">{sam3Layers.length}</span>
            <div className="flex-1 h-px bg-neutral-800/60" />
          </div>
          {sam3Layers.map((layer) => {
            const isSelected =
              selectedSam3Instance?.promptIndex === layer.promptIndex &&
              selectedSam3Instance?.instanceIndex === layer.instanceIndex;
            return (
              <div
                key={layer.id}
                role="button"
                tabIndex={0}
                className={cn(
                  "reveal flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all duration-150 select-none group",
                  isSelected
                    ? "bg-orange-500/10 border border-orange-500/25 shadow-[0_0_8px_rgba(249,115,22,0.1)]"
                    : "border border-transparent hover:bg-neutral-800/50 hover:border-neutral-700/30"
                )}
                onClick={() =>
                  setSelectedSam3Instance(isSelected ? null : { promptIndex: layer.promptIndex, instanceIndex: layer.instanceIndex })
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setSelectedSam3Instance(isSelected ? null : { promptIndex: layer.promptIndex, instanceIndex: layer.instanceIndex });
                  }
                }}
              >
                <span className="text-[11px] text-neutral-300 font-sans truncate flex-1">{layer.label}</span>
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 bg-orange-400" />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemoveSam3Instance?.(layer.promptIndex, layer.instanceIndex);
                  }}
                  aria-label="Remove segment"
                  className="w-6 h-6 min-w-[24px] min-h-[24px] rounded flex items-center justify-center text-neutral-600 hover:text-red-400 hover:bg-red-500/10 opacity-100 transition-all duration-150 hover:scale-110 cursor-pointer"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                </button>
              </div>
            );
          })}
        </div>
      )}
      </div>
    </div>
  );
}
