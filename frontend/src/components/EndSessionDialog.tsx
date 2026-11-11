import { useState } from "react";
import { useAtom } from "jotai";
import { endSessionOpenAtom } from "@/store/ui";
import { useSession } from "@/hooks/useSession";
import { useSessionBusy } from "@/hooks/useSessionBusy";
import { DialogFooter } from "@/components/ui/DialogFooter";

export function EndSessionDialog() {
  const [open, setOpen] = useAtom(endSessionOpenAtom);
  const { endSession } = useSession();
  const { running, tasks } = useSessionBusy();
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const handleCancel = () => {
    if (!busy) setOpen(false);
  };

  const handleConfirm = async () => {
    setBusy(true);
    try {
      await endSession();
    } finally {
      setBusy(false);
      setOpen(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="End session">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={handleCancel}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-sm mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-red-500/20 via-red-500/5 to-transparent pointer-events-none" />

        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-red-500/40 to-transparent" />

          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-red-500/20 to-red-600/10 border border-red-500/20 flex items-center justify-center shadow-lg shadow-red-500/10">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-red-400"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">End this session?</h3>
                  <p className="text-[10px] text-neutral-500 font-sans">model and image will be unloaded</p>
                </div>
              </div>
              <button
                onClick={handleCancel}
                aria-label="Close end session dialog"
                disabled={busy}
                className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">Session</label>
              <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                The model and image will be unloaded and memory freed. Any unsaved work in this session will be lost.
              </p>
              {running && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2.5 py-2">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0 mt-0.5 text-amber-400">
                    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" />
                    <path d="M12 9v4" />
                    <path d="M12 17h.01" />
                  </svg>
                  <p className="text-[11px] text-amber-200/90 font-sans leading-relaxed">
                    Session is still running: {tasks.join(", ")}. Ending now will interrupt it.
                  </p>
                </div>
              )}
            </div>

            <DialogFooter
              busy={busy}
              tone="danger"
              confirmLabel="End Session"
              busyLabel="Ending..."
              onCancel={handleCancel}
              onConfirm={handleConfirm}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
