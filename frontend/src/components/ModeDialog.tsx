import { useAtom } from "jotai";
import { modeDialogOpenAtom, modeLockAtom } from "@/store/ui";
import { detectModeAtom, toolModeAtom } from "@/store/detection";

export function ModeDialog() {
  const [modeDialogOpen, setModeDialogOpen] = useAtom(modeDialogOpenAtom);
  const [, setModeLock] = useAtom(modeLockAtom);
  const [, setDetectMode] = useAtom(detectModeAtom);
  const [, setToolMode] = useAtom(toolModeAtom);

  if (!modeDialogOpen) return null;

  const handleSelect = (mode: "brush" | "detect") => {
    setModeLock(mode);
    setModeDialogOpen(false);
    if (mode === "brush") {
      setDetectMode(false);
      setToolMode("positive");
    } else {
      setDetectMode(true);
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Choose mode">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-lg smooth-dialog" style={{ animation: "none" }} />
      <div className="animate-drop-in [animation-duration:0.25s] relative z-[60] w-full max-w-lg mx-4">
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
                className="group relative p-5 rounded-xl border border-neutral-800/50 bg-neutral-900/30 hover:bg-orange-500/5 hover:border-orange-500/20 hover:shadow-[0_0_30px_rgba(184,92,42,0.08)] transition-all duration-300 cursor-pointer text-left select-none"
              >
                <div className="w-11 h-11 rounded-xl bg-orange-500/10 border border-orange-500/15 flex items-center justify-center mb-3 group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-orange-500/10 transition-all duration-300">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-orange-400">
                    <path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08"/>
                    <path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z"/>
                  </svg>
                </div>
                <h3 className="text-sm font-semibold text-neutral-200 font-sans mb-1 group-hover:text-orange-200 transition-colors">Brush</h3>
                <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                  Paint strokes to mark regions. Green to keep, red to remove. Best for precise, hand-drawn selections.
                </p>
                <div className="mt-3 flex items-center gap-1.5 text-[10px] text-neutral-600 font-mono">
                  <kbd className="px-1.5 py-0.5 bg-neutral-800/80 border border-neutral-700/50 rounded text-neutral-400">B</kbd>
                  <span>brush</span>
                </div>
              </button>

              <button
                onClick={() => handleSelect("detect")}
                className="group relative p-5 rounded-xl border border-neutral-800/50 bg-neutral-900/30 hover:bg-green-500/5 hover:border-green-500/20 hover:shadow-[0_0_30px_rgba(0,255,136,0.06)] transition-all duration-300 cursor-pointer text-left select-none"
              >
                <div className="w-11 h-11 rounded-xl bg-green-500/10 border border-green-500/15 flex items-center justify-center mb-3 group-hover:scale-110 group-hover:shadow-lg group-hover:shadow-green-500/10 transition-all duration-300">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-green-400">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.3-4.3"/>
                  </svg>
                </div>
                <h3 className="text-sm font-semibold text-neutral-200 font-sans mb-1 group-hover:text-green-200 transition-colors">Detect</h3>
                <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                  Type what to find and AI detects it automatically. Works best with common objects.
                </p>
                <div className="mt-3 flex items-center gap-1.5 text-[10px] text-neutral-600 font-mono">
                  <kbd className="px-1.5 py-0.5 bg-neutral-800/80 border border-neutral-700/50 rounded text-neutral-400">D</kbd>
                  <span>detect</span>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
