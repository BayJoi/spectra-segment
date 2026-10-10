import { useRef, useEffect, useCallback, type RefObject } from "react";
import { useAtom } from "jotai";
import { useSession } from "@/hooks/useSession";
import {
  brushObjectsAtom,
  activeObjectIdAtom,
  objectMasksAtom,
  subjectMetaAtom,
  objectHistoryAtom,
  subjectColor,
} from "@/store/session";
import { cn } from "@/lib/utils";

interface SubjectsPanelProps {
  open: boolean;
  onClose: () => void;
  ignoredRef?: RefObject<HTMLElement | null>;
}

export function SubjectsPanel({ open, onClose, ignoredRef }: SubjectsPanelProps) {
  const { clearObjectHistory, forgetSubject } = useSession();
  const [brushObjects, setBrushObjects] = useAtom(brushObjectsAtom);
  const [activeObjectId, setActiveObjectId] = useAtom(activeObjectIdAtom);
  const [objectMasks] = useAtom(objectMasksAtom);
  const [subjectMeta, setSubjectMeta] = useAtom(subjectMetaAtom);
  const [objectHistory] = useAtom(objectHistoryAtom);
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

  useEffect(() => {
    const missing = brushObjects.filter((oid) => !subjectMeta[oid]);
    if (missing.length === 0) return;
    setSubjectMeta((prev) => {
      const next = { ...prev };
      for (const oid of missing) {
        next[oid] = { id: oid, name: `Subject ${oid + 1}`, color: subjectColor(oid) };
      }
      return next;
    });
  }, [brushObjects, subjectMeta, setSubjectMeta]);

  const handleDelete = useCallback(
    (oid: number) => {
      if (brushObjects.length <= 1) return;
      const idx = brushObjects.indexOf(oid);
      const next = brushObjects.filter((x) => x !== oid);
      setBrushObjects(next);
      if (oid === activeObjectId) {
        setActiveObjectId(next[Math.min(idx, next.length - 1)]);
      }
      setSubjectMeta((prev) => {
        const nextMeta = { ...prev };
        delete nextMeta[oid];
        return nextMeta;
      });
      forgetSubject(oid);
      void clearObjectHistory(oid);
    },
    [brushObjects, activeObjectId, setBrushObjects, setActiveObjectId, setSubjectMeta, clearObjectHistory]
  );

  if (!open || brushObjects.length === 0) return null;

  return (
    <div
      ref={panelRef}
      className="animate-drop-in w-72 max-h-72 flex flex-col bg-[#0a0a0a]/95 border border-neutral-800/80 rounded-xl shadow-2xl shadow-black/60 backdrop-blur-xl grain-bg overflow-hidden"
    >
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-2">
      <div className="flex items-center justify-between px-1 mb-1.5">
        <span className="text-[11px] text-neutral-500 uppercase tracking-wider font-sans">Subjects</span>
        <span className="text-[10px] text-neutral-600 font-mono">{brushObjects.length}</span>
      </div>

      <div className="flex flex-col gap-1">
        {brushObjects.map((oid) => {
          const isActive = oid === activeObjectId;
          const meta = subjectMeta[oid];
          const color = meta?.color ?? subjectColor(oid);
          const strokes = objectHistory[oid]?.undo ?? 0;
          const hasMask = oid in objectMasks;
          return (
            <div
              key={oid}
              role="button"
              tabIndex={0}
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all duration-150 select-none group",
                isActive
                  ? "bg-orange-500/10 border border-orange-500/25 shadow-[0_0_8px_rgba(249,115,22,0.1)]"
                  : "border border-transparent hover:bg-neutral-800/50 hover:border-neutral-700/30"
              )}
              onClick={() => setActiveObjectId(oid)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setActiveObjectId(oid);
                }
              }}
            >
              <span
                className="w-2.5 h-2.5 rounded-sm flex-shrink-0 ring-1 ring-inset ring-white/10"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="text-[11px] text-neutral-300 font-sans truncate flex-1">
                {meta?.name ?? `Subject ${oid + 1}`}
              </span>
              <span className="text-[10px] text-neutral-600 font-mono tabular-nums">{strokes}</span>
              <span className={cn(
                "w-1.5 h-1.5 rounded-full flex-shrink-0",
                hasMask ? "bg-orange-400" : "bg-neutral-600"
              )} />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(oid);
                }}
                disabled={brushObjects.length <= 1}
                aria-label={`Delete ${meta?.name ?? `Subject ${oid + 1}`}`}
                className="w-6 h-6 min-w-[24px] min-h-[24px] rounded flex items-center justify-center text-neutral-600 hover:text-red-400 hover:bg-red-500/10 opacity-100 transition-all duration-150 hover:scale-110 cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>
          );
        })}
      </div>
      </div>
    </div>
  );
}
