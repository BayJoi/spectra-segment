import { useState } from "react";
import { useAtom } from "jotai";
import { modeSwitchTargetAtom } from "@/store/ui";
import { sam3ModeAtom } from "@/store/sam3";
import { sessionIdAtom } from "@/store/session";
import { useSession } from "@/hooks/useSession";

export function ModeSwitchDialog() {
  const [target, setTarget] = useAtom(modeSwitchTargetAtom);
  const [sam3Mode, setSam3Mode] = useAtom(sam3ModeAtom);
  const [sessionId] = useAtom(sessionIdAtom);
  const { endSession } = useSession();
  const [busy, setBusy] = useState(false);

  if (!target) return null;

  const current = sam3Mode ? "SAM 3" : "Brush / Detect";
  const next = target === "sam3" ? "SAM 3" : "Brush / Detect";

  const handleCancel = () => {
    if (!busy) setTarget(null);
  };

  const handleConfirm = async () => {
    setBusy(true);
    try {
      if (sessionId) {
        await endSession();
      }
      setSam3Mode(target === "sam3");
    } finally {
      setBusy(false);
      setTarget(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Switch segmentation mode">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={handleCancel}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-sm mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-orange-500/20 via-orange-500/5 to-transparent pointer-events-none" />

        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-orange-500/40 to-transparent" />

          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-orange-500/20 to-orange-600/10 border border-orange-500/20 flex items-center justify-center shadow-lg shadow-orange-500/10">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-orange-400"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">Switch to {next}?</h3>
                  <p className="text-[10px] text-neutral-500 font-sans">from {current}</p>
                </div>
              </div>
              <button
                onClick={handleCancel}
                aria-label="Close switch dialog"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">Session</label>
              <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                {sessionId
                  ? "Switching modes will end the current session and unload the loaded model to free memory. Any unsaved work in this session will be lost."
                  : "Nothing is loaded yet, so switching is instant and safe."}
              </p>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleCancel}
                disabled={busy}
                className="flex-1 h-9 rounded-xl text-xs font-medium font-sans bg-neutral-900/50 border border-neutral-800/50 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700 hover:bg-neutral-800/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirm}
                disabled={busy}
                className="flex-1 h-9 rounded-xl text-xs font-semibold font-sans bg-gradient-to-r from-orange-500 to-orange-600 text-white hover:from-orange-400 hover:to-orange-500 hover:shadow-lg hover:shadow-orange-500/20 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 grain-bg grain-bg-strong"
              >
                {busy ? (
                  <span className="inline-flex items-center justify-center gap-2">
                    <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                    Switching…
                  </span>
                ) : (
                  "Switch Mode"
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
