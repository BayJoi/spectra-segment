import { useAtom } from "jotai";
import { modeDialogOpenAtom, modeLockAtom } from "@/store/ui";
import { detectModeAtom, toolModeAtom } from "@/store/detection";
import { cn } from "@/lib/utils";

type Mode = "brush" | "detect";

const cardClass = (selected: boolean) =>
  cn(
    "group relative p-5 rounded-xl border text-left select-none transition-all duration-200 cursor-pointer focus-visible:outline-2 focus-visible:outline-orange-500 focus-visible:outline-offset-2",
    selected
      ? "border-orange-500/55 bg-orange-500/[0.09] shadow-[0_0_0_1px_rgba(184,92,42,0.18),0_20px_50px_-28px_rgba(184,92,42,0.45)] hover:border-orange-500/70 hover:bg-orange-500/[0.12]"
      : "border-neutral-800/60 bg-neutral-900/35 hover:border-orange-500/35 hover:bg-orange-500/[0.06] hover:shadow-[0_12px_40px_-20px_rgba(0,0,0,0.8)]"
  );

export function ModeDialog() {
  const [modeDialogOpen, setModeDialogOpen] = useAtom(modeDialogOpenAtom);
  const [modeLock, setModeLock] = useAtom(modeLockAtom);
  const [detectMode, setDetectMode] = useAtom(detectModeAtom);
  const [, setToolMode] = useAtom(toolModeAtom);

  if (!modeDialogOpen) return null;

  const modeIsLocked = modeLock === "brush" || modeLock === "detect";
  const selectedMode: Mode | null = modeIsLocked
    ? modeLock
    : modeLock === null
      ? detectMode
        ? "detect"
        : "brush"
      : null;

  const handleSelect = (mode: Mode) => {
    setModeLock(mode);
    setModeDialogOpen(false);
    if (mode === "detect") {
      setDetectMode(true);
    } else {
      setDetectMode(false);
      setToolMode("positive");
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Choose mode">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-lg smooth-dialog" style={{ animation: "none" }} />
      <div className="animate-drop-in [animation-duration:0.25s] relative z-[60] w-full max-w-2xl mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-orange-500/15 via-transparent to-transparent pointer-events-none" />
        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-orange-500/30 to-transparent" />

          <div className="p-7">
            <div className="text-center mb-6">
              <h2 className="text-lg font-semibold text-neutral-100 font-sans mb-1">Choose your mode</h2>
              <p className="text-xs text-neutral-500 font-sans">Select how you want to segment your image. You can switch by replacing the image.</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleSelect("brush")}
                aria-pressed={selectedMode === "brush"}
                className={cardClass(selectedMode === "brush")}
              >
                {selectedMode === "brush" && (
                  <span className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full border border-orange-500/40 bg-orange-500/15 px-1.5 py-0.5 text-[9px] font-medium font-sans uppercase tracking-[0.12em] text-orange-300">
                    <span className="h-1 w-1 rounded-full bg-orange-400" />
                    {modeIsLocked ? "Active" : "Default"}
                  </span>
                )}
                <div
                  className={cn(
                    "w-11 h-11 rounded-xl border flex items-center justify-center mb-3 transition-all duration-300",
                    selectedMode === "brush"
                      ? "bg-orange-500/20 border-orange-500/35"
                      : "bg-orange-500/10 border-orange-500/15 group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-orange-500/10"
                  )}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400">
                    <path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08" />
                    <path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z" />
                  </svg>
                </div>
                <h3
                  className={cn(
                    "text-sm font-semibold font-sans mb-1 transition-colors",
                    selectedMode === "brush" ? "text-orange-100" : "text-neutral-200 group-hover:text-orange-200"
                  )}
                >
                  Brush
                </h3>
                <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                  Paint strokes to mark regions. Left-drag adds, right-drag subtracts.
                  Each subject keeps its own colour. Best for precise, hand-drawn selections.
                </p>
                <div className="mt-3 flex items-center gap-1.5 text-[10px] text-neutral-600 font-sans">
                  <kbd className="px-1.5 py-0.5 bg-neutral-800/80 border border-neutral-700/50 rounded text-neutral-400">click</kbd>
                  <span>to start</span>
                </div>
              </button>

              <button
                onClick={() => handleSelect("detect")}
                aria-pressed={selectedMode === "detect"}
                className={cardClass(selectedMode === "detect")}
              >
                {selectedMode === "detect" && (
                  <span className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full border border-orange-500/40 bg-orange-500/15 px-1.5 py-0.5 text-[9px] font-medium font-sans uppercase tracking-[0.12em] text-orange-300">
                    <span className="h-1 w-1 rounded-full bg-orange-400" />
                    {modeIsLocked ? "Active" : "Default"}
                  </span>
                )}
                <div
                  className={cn(
                    "w-11 h-11 rounded-xl border flex items-center justify-center mb-3 transition-all duration-300",
                    selectedMode === "detect"
                      ? "bg-orange-500/20 border-orange-500/35"
                      : "bg-orange-500/10 border-orange-500/15 group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-orange-500/10"
                  )}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400">
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.3-4.3" />
                  </svg>
                </div>
                <h3
                  className={cn(
                    "text-sm font-semibold font-sans mb-1 transition-colors",
                    selectedMode === "detect" ? "text-orange-100" : "text-neutral-200 group-hover:text-orange-200"
                  )}
                >
                  Detect
                </h3>
                <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                  Type what to find and the detector boxes it automatically.
                  Results accumulate, so you can find several things in one image.
                </p>
                <div className="mt-3 flex items-center gap-1.5 text-[10px] text-neutral-600 font-sans">
                  <kbd className="px-1.5 py-0.5 bg-neutral-800/80 border border-neutral-700/50 rounded text-neutral-400">click</kbd>
                  <span>to start</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
