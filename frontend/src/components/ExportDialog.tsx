import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useAtom } from "jotai";
import { exportOpenAtom, isExportingAtom } from "@/store/ui";
import { sessionIdAtom, imageFileAtom, masksAtom, perDetectionMasksAtom, objectMasksAtom } from "@/store/session";
import { layersAtom } from "@/store/layers";
import { featherRadiusAtom } from "@/store/detection";
import { api, encodeMaskPng } from "@/lib/api";
import { cn } from "@/lib/utils";

type ExportMode = "everything" | "layer" | "sublayers";

type ExportKind = "detection" | "brush";

interface GroupSub {
  key: string;
  detectionIndex?: number;
  brushOid?: number;
  name: string;
  hasMask: boolean;
  kind: ExportKind;
}

interface Group {
  name: string;
  subs: GroupSub[];
}

interface FileEntry {
  path: string;
  kind: ExportKind;
  detectionIndex?: number;
  brushOid?: number;
}

function slugify(s: string): string {
  const clean = s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return clean || "layer";
}

function safeFolder(s: string): string {
  const clean = s
    .replace(/[\\/]/g, " ")
    .replace(/[<>:"|?*\u0000-\u001f]/g, "")
    .trim();
  return clean && clean !== "." && clean !== ".." ? clean : "layer";
}

function base64ToBlob(base64: string, type: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type });
}

function hexToRgb(hex: string): number[] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)] : [255, 255, 255];
}

export function ExportDialog() {
  const [exportOpen, setExportOpen] = useAtom(exportOpenAtom);
  const [isExporting, setIsExporting] = useAtom(isExportingAtom);
  const [sessionId] = useAtom(sessionIdAtom);
  const [imageFile] = useAtom(imageFileAtom);
  const [layers] = useAtom(layersAtom);
  const [perDetectionMasks] = useAtom(perDetectionMasksAtom);
  const [masks] = useAtom(masksAtom);
  const [objectMasks] = useAtom(objectMasksAtom);
  const [feather, setFeather] = useAtom(featherRadiusAtom);

  const [mode, setMode] = useState<ExportMode>("everything");
  const [selectedGroup, setSelectedGroup] = useState<string | null>(null);
  const [selectedSubs, setSelectedSubs] = useState<Set<string>>(new Set());
  const [format, setFormat] = useState<"png" | "jpg">("png");
  const [bgMode, setBgMode] = useState<"transparent" | "solid">("transparent");
  const [bgColor, setBgColor] = useState("#ffffff");
  const [exported, setExported] = useState(false);
  const exportedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (exportedTimerRef.current) clearTimeout(exportedTimerRef.current);
    };
  }, []);

  const rootName = useMemo(() => {
    const n = imageFile?.name;
    if (!n) return "spectra-segment";
    return safeFolder(n.replace(/\.[^.]+$/, "")) || "image";
  }, [imageFile]);

  const hasDetectionLayers = layers.some((l) => l.detectionIndex !== undefined);
  const isBrush = !hasDetectionLayers && masks.length > 0;

  const brushSubs: GroupSub[] = useMemo(() => {
    const oids = Object.keys(objectMasks).map(Number).sort((a, b) => a - b);
    if (oids.length > 0) {
      return oids.map((oid) => ({
        key: `subject::${oid}`,
        brushOid: oid,
        name: `Subject ${oid + 1}`,
        hasMask: true,
        kind: "brush",
      }));
    }
    return masks.map((_, i) => ({
      key: `subject::${i}`,
      brushOid: i,
      name: `Subject ${i + 1}`,
      hasMask: true,
      kind: "brush",
    }));
  }, [objectMasks, masks]);

  const groups: Group[] = useMemo(() => {
    if (isBrush) {
      return brushSubs.map((s) => ({ name: s.name, subs: [s] }));
    }
    const map = new Map<string, GroupSub[]>();
    for (const l of layers) {
      if (l.detectionIndex === undefined) continue;
      const name = l.label;
      if (!map.has(name)) map.set(name, []);
      map.get(name)!.push({
        key: `${name}::${l.detectionIndex}`,
        detectionIndex: l.detectionIndex,
        name,
        hasMask: l.detectionIndex in perDetectionMasks,
        kind: "detection",
      });
    }
    return Array.from(map, ([name, subs]) => ({ name, subs }));
  }, [isBrush, brushSubs, layers, perDetectionMasks]);

  const effectiveGroup =
    selectedGroup && groups.some((g) => g.name === selectedGroup) ? selectedGroup : groups[0]?.name ?? null;

  const filesToExport: FileEntry[] = useMemo(() => {
    const files: FileEntry[] = [];
    const addGroup = (g: Group, pick: (s: GroupSub) => boolean) => {
      g.subs.forEach((s, i) => {
        if (pick(s)) {
          if (s.kind === "brush") {
            files.push({
              path: `${rootName}_subj${s.brushOid! + 1}.png`,
              kind: "brush",
              brushOid: s.brushOid,
            });
          } else {
            files.push({
              path: `${safeFolder(g.name)}/${slugify(s.name)}_${i + 1}.png`,
              kind: "detection",
              detectionIndex: s.detectionIndex,
            });
          }
        }
      });
    };
    if (mode === "everything") {
      groups.forEach((g) => addGroup(g, () => true));
    } else if (mode === "layer") {
      const g = groups.find((gg) => gg.name === effectiveGroup);
      if (g) addGroup(g, () => true);
    } else {
      groups.forEach((g) => addGroup(g, (s) => selectedSubs.has(s.key)));
    }
    return files.filter((f) => (f.kind === "detection" ? f.detectionIndex! in perDetectionMasks : true));
  }, [mode, groups, effectiveGroup, selectedSubs, perDetectionMasks, rootName]);

  const includeWhole = mode === "everything";

  const toggleSub = useCallback((key: string) => {
    setSelectedSubs((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const handleExport = useCallback(async () => {
    if (!sessionId || filesToExport.length === 0) return;
    setIsExporting(true);
    setExported(false);
    try {
      const encoded: { path: string; mask_b64: string }[] = [];
      for (const f of filesToExport) {
        const mask = f.kind === "brush" ? objectMasks[f.brushOid!] ?? masks[f.brushOid!] : perDetectionMasks[f.detectionIndex!];
        if (!mask) continue;
        encoded.push({ path: f.path, mask_b64: encodeMaskPng(mask) });
      }
      if (encoded.length === 0) return;
      const res = await api.exportZip(sessionId, {
        root: isBrush ? "" : rootName,
        files: encoded,
        include_whole: includeWhole,
        format,
        background_color: bgMode === "solid" ? hexToRgb(bgColor) : undefined,
        feather_radius: feather,
      });
      const blob = base64ToBlob(res.data, "application/zip");
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${rootName}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      setExported(true);
      if (exportedTimerRef.current) clearTimeout(exportedTimerRef.current);
      exportedTimerRef.current = setTimeout(() => setExported(false), 2000);
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setIsExporting(false);
    }
  }, [sessionId, filesToExport, objectMasks, masks, perDetectionMasks, rootName, includeWhole, format, bgMode, bgColor, feather, setIsExporting, isBrush]);

  if (!exportOpen) return null;

  const groupCount = (g: Group) => g.subs.filter((s) => s.hasMask).length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label={isBrush ? "Export subjects" : "Export layers"}>
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={() => setExportOpen(false)}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-md mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-orange-500/20 via-orange-500/5 to-transparent pointer-events-none" />

        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-orange-500/20 to-orange-600/10 border border-orange-500/20 flex items-center justify-center shadow-lg shadow-orange-500/10">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">{isBrush ? "Export Subjects" : "Export Layers"}</h3>
                  <p className="text-[10px] text-neutral-500 font-sans">{`${rootName}.zip`}</p>
                </div>
              </div>
              <button
                onClick={() => setExportOpen(false)}
                aria-label="Close export dialog"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">What to export</label>
              <div className="grid grid-cols-3 gap-2">
                {([
                  { key: "everything", title: "Everything", desc: isBrush ? "All subjects + composite" : "All layers + whole" },
                  { key: "layer", title: isBrush ? "One subject" : "One layer", desc: isBrush ? "Single subject" : "Single group" },
                  { key: "sublayers", title: isBrush ? "Pick subjects" : "Sublayers", desc: isBrush ? "Pick specific" : "Pick specific" },
                ] as const).map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setMode(m.key)}
                    className={cn(
                      "px-2 py-2 rounded-xl border text-left transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                      mode === m.key
                        ? "bg-orange-500/10 border-orange-500/30 text-orange-200 shadow-[0_0_20px_rgba(184,92,42,0.1)]"
                        : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300 hover:bg-neutral-800/30"
                    )}
                  >
                    <div className="text-[11px] font-semibold font-sans">{m.title}</div>
                    <div className="text-[9px] text-neutral-600 font-normal mt-0.5 leading-tight">{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {mode === "layer" && (
              <div className="mb-3 max-h-40 overflow-y-auto custom-scrollbar rounded-lg border border-neutral-800/60 divide-y divide-neutral-800/40">
                {groups.map((g) => (
                  <button
                    key={g.name}
                    onClick={() => setSelectedGroup(g.name)}
                    className={cn(
                      "w-full flex items-center gap-2 px-2.5 py-2 text-left transition-all duration-150 cursor-pointer hover:scale-[1.02] active:scale-[0.98]",
                      effectiveGroup === g.name ? "bg-orange-500/10 text-orange-200" : "text-neutral-400 hover:bg-neutral-800/40 hover:text-neutral-200"
                    )}
                  >
                    <span className={cn(
                      "w-3.5 h-3.5 rounded-full border flex items-center justify-center flex-shrink-0",
                      effectiveGroup === g.name ? "border-orange-400" : "border-neutral-600"
                    )}>
                      {effectiveGroup === g.name && <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />}
                    </span>
                    <span className="text-xs font-sans truncate flex-1">{g.name}</span>
                    <span className="text-[10px] font-mono text-neutral-500">{groupCount(g)}</span>
                  </button>
                ))}
              </div>
            )}

            {mode === "sublayers" && (
              <div className="mb-3 max-h-40 overflow-y-auto custom-scrollbar rounded-lg border border-neutral-800/60">
                {groups.map((g) => (
                  <div key={g.name} className="px-2.5 py-2 border-b border-neutral-800/40">
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="text-[10px] text-neutral-500 uppercase tracking-wider font-sans font-semibold">{g.name}</span>
                      <span className="text-[9px] text-neutral-600 font-mono">{groupCount(g)}</span>
                    </div>
                    <div className="flex flex-col gap-0.5">
                      {g.subs.map((s) => {
                        const key = s.key;
                        const checked = selectedSubs.has(key);
                        return (
                          <button
                            key={key}
                            disabled={!s.hasMask}
                            onClick={() => toggleSub(key)}
                            className={cn(
                              "flex items-center gap-2 px-2 py-1 rounded-md text-left transition-all duration-150 cursor-pointer hover:scale-[1.01] active:scale-[0.99]",
                              s.hasMask ? "hover:bg-neutral-800/50" : "opacity-40 cursor-not-allowed",
                              checked && "bg-green-500/10"
                            )}
                          >
                            <span className={cn(
                              "w-3.5 h-3.5 rounded border flex items-center justify-center flex-shrink-0",
                              checked ? "bg-green-500/20 border-green-400" : "border-neutral-600"
                            )}>
                              {checked && <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" className="text-green-400"><path d="M20 6 9 17l-5-5"/></svg>}
                            </span>
                            <span className="text-[11px] text-neutral-300 font-sans truncate flex-1">{s.kind === "brush" ? s.name : `${g.name} ${s.detectionIndex! + 1}`}</span>
                            <span className={cn("w-1.5 h-1.5 rounded-full flex-shrink-0", s.hasMask ? "bg-green-400" : "bg-neutral-600")} />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {includeWhole && (
              <div className="mb-3">
                <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">{isBrush ? "Composite format" : "Whole image format"}</label>
                <div className="grid grid-cols-2 gap-2">
                  {(["png", "jpg"] as const).map((f) => (
                <button
                  key={f}
                  onClick={() => {
                    setFormat(f);
                    if (f === "jpg") setBgMode("solid");
                  }}
                  className={cn(
                    "px-3 py-1.5 rounded-xl border text-xs font-medium font-sans transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                    format === f
                      ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                      : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
                  )}
                >
                  {f.toUpperCase()}{f === "jpg" ? " (no alpha)" : ""}
                </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">Background</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => { setBgMode("transparent"); if (format === "jpg") setFormat("png"); }}
                  disabled={format === "jpg"}
                  title={format === "jpg" ? "JPEG needs a solid background — switch to PNG first" : undefined}
                  className={cn(
                    "px-3 py-1.5 rounded-xl border text-xs font-medium font-sans transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                    bgMode === "transparent"
                      ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                      : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300",
                    format === "jpg" && "opacity-40 cursor-not-allowed"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded-md bg-[repeating-conic-gradient(#333_0%_25%,#222_0%_50%)] bg-[length:6px_6px] border border-neutral-700/50" />
                    Transparent
                  </div>
                </button>
                <button
                  onClick={() => setBgMode("solid")}
                  className={cn(
                    "px-3 py-1.5 rounded-xl border text-xs font-medium font-sans transition-all duration-200 cursor-pointer select-none hover:scale-[1.02] active:scale-[0.98]",
                    bgMode === "solid"
                      ? "bg-orange-500/10 border-orange-500/30 text-orange-200"
                      : "bg-neutral-900/50 border-neutral-800/50 text-neutral-500 hover:border-neutral-700 hover:text-neutral-300"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded-md border border-neutral-700/50" style={{ backgroundColor: bgColor }} />
                    Solid color
                  </div>
                </button>
              </div>
              {bgMode === "solid" && (
                <div className="mt-2 flex items-center gap-3 pl-1 animate-fade-in">
                  <input
                    type="color"
                    value={bgColor}
                    onChange={(e) => setBgColor(e.target.value)}
                    className="w-7 h-7 rounded-lg border border-neutral-800 cursor-pointer hover:border-neutral-700 transition-colors"
                  />
                  <span className="text-[11px] text-neutral-500 font-mono">{bgColor}</span>
                </div>
              )}
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">Feather</label>
              <div className="flex items-center gap-2 pl-1">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-500"><path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/><line x1="16" y1="8" x2="2" y2="22"/></svg>
                <input
                  type="range"
                  min={0}
                  max={20}
                  value={feather}
                  aria-label="Feather radius"
                  onChange={(e) => setFeather(Number(e.target.value))}
                  className="w-full h-1 cursor-pointer"
                  style={{ "--pct": `${(feather / 20) * 100}%` } as React.CSSProperties}
                />
                <span className="text-[10px] text-neutral-500 font-mono tabular-nums min-w-[16px] text-right">{feather}</span>
              </div>
            </div>

            {filesToExport.length > 0 && (
              <div className="mb-3 rounded-lg border border-neutral-800/60 bg-black/30 p-2 max-h-32 overflow-y-auto custom-scrollbar">
                <div className="flex items-center gap-1 text-[10px] text-neutral-400 font-mono px-1">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400 shrink-0"><path d="M3 7v11a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-6l-2-2H5a2 2 0 0 0-2 2z"/></svg>
                  <span className="truncate">{rootName}/</span>
                </div>
                {(() => {
                  const byFolder = new Map<string, string[]>();
                  for (const f of filesToExport) {
                    const sep = f.path.indexOf("/");
                    const folder = sep >= 0 ? f.path.slice(0, sep) : "";
                    const file = sep >= 0 ? f.path.slice(sep + 1) : f.path;
                    if (!byFolder.has(folder)) byFolder.set(folder, []);
                    byFolder.get(folder)!.push(file);
                  }
                  const entries: React.ReactNode[] = [];
                  for (const [folder, files] of byFolder) {
                    if (!folder) {
                      files.forEach((file) => {
                        entries.push(
                          <div key={file} className="flex items-center gap-1 text-[10px] text-neutral-400 font-mono pl-3">
                            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-600 shrink-0"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                            {file}
                          </div>
                        );
                      });
                      continue;
                    }
                    entries.push(
                      <div key={folder} className="flex items-center gap-1 text-[10px] text-neutral-500 font-mono pl-3">
                        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-600 shrink-0"><path d="M3 7v11a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-6l-2-2H5a2 2 0 0 0-2 2z"/></svg>
                        {folder}/
                      </div>
                    );
                    files.forEach((file) => {
                      entries.push(
                        <div key={`${folder}/${file}`} className="flex items-center gap-1 text-[10px] text-neutral-400 font-mono pl-6">
                          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-600 shrink-0"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                          {file}
                        </div>
                      );
                    });
                  }
                  if (includeWhole) {
                    entries.push(
                      <div key="__whole" className="flex items-center gap-1 text-[10px] text-neutral-400 font-mono pl-6">
                        <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-600 shrink-0"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                        whole.{format}
                      </div>
                    );
                  }
                  return entries;
                })()}
              </div>
            )}

            <button
              onClick={handleExport}
              disabled={isExporting || filesToExport.length === 0}
              className={cn(
                "w-full h-10 rounded-xl font-medium text-sm font-sans transition-all duration-200 cursor-pointer select-none",
                "flex items-center justify-center gap-2",
                exported
                  ? "bg-green-500/15 border border-green-500/30 text-green-400"
                  : "bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-[1.02] active:scale-[0.98] grain-bg grain-bg-strong",
                (isExporting || filesToExport.length === 0) && "opacity-60 cursor-not-allowed"
              )}
            >
                  {isExporting ? (
                  <>
                    <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                    Exporting...
                  </>
                ) : exported ? (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M20 6 9 17l-5-5"/></svg>
                    Downloaded!
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    Export {filesToExport.length} {filesToExport.length === 1 ? "file" : "files"}{includeWhole ? (isBrush ? " + composite" : " + whole") : ""}
                  </>
                )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
