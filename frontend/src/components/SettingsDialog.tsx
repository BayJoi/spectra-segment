import { useAtom } from "jotai";
import { settingsOpenAtom } from "@/store/ui";
import { hasImageAtom } from "@/store/session";
import { modeLockAtom } from "@/store/ui";

export function SettingsDialog() {
  const [settingsOpen, setSettingsOpen] = useAtom(settingsOpenAtom);
  const [hasImage] = useAtom(hasImageAtom);
  const [modeLock] = useAtom(modeLockAtom);

  if (!settingsOpen) return null;

  const isBrushMode = hasImage && modeLock === "brush";
  const isDetectMode = hasImage && modeLock === "detect";
  const noImage = !hasImage;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Settings">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={() => setSettingsOpen(false)}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-sm mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-neutral-700/20 via-neutral-800/5 to-transparent pointer-events-none" />
        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-neutral-700/30 to-transparent" />

          <div className="p-6">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-neutral-800/50 border border-neutral-700/30 flex items-center justify-center">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-neutral-400"><circle cx="12" cy="12" r="3"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42m12.72-12.72 1.42-1.42"/></svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">Settings</h3>
                  <p className="text-[11px] text-neutral-500 font-sans mt-0.5">
                    {noImage ? "All keyboard shortcuts" : isBrushMode ? "Brush mode shortcuts" : isDetectMode ? "Detect mode shortcuts" : "Keyboard shortcuts"}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSettingsOpen(false)}
                aria-label="Close settings"
                className="w-8 h-8 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>

            <div className="space-y-2">
              {noImage && (
                <div className="text-[11px] text-neutral-500 font-sans text-center py-6">
                  Open an image to see shortcuts
                </div>
              )}
              {isBrushMode && (
                <>
                  <ShortcutRow keys={["Esc"]} label="Close dialog / Deselect" />
                  <ShortcutRow keys={["Del"]} label="Remove last layer" />
                  <ShortcutRow keys={["Ctrl", "Z"]} label="Undo" />
                  <ShortcutRow keys={["Ctrl", "Y"]} label="Redo" />
                  <ShortcutRow keys={["Scroll"]} label="Zoom in/out" />
                  <ShortcutRow keys={["Shift", "Drag"]} label="Pan canvas" />
                  <ShortcutRow keys={["Right click"]} label="Negative stroke" />
                </>
              )}
              {isDetectMode && (
                <>
                  <ShortcutRow keys={["Esc"]} label="Close dialog / Deselect" />
                  <ShortcutRow keys={["Enter"]} label="Run detection" />
                  <ShortcutRow keys={["Del"]} label="Remove selected" />
                  <ShortcutRow keys={["Ctrl", "Z"]} label="Undo" />
                  <ShortcutRow keys={["Ctrl", "Y"]} label="Redo" />
                  <ShortcutRow keys={["Scroll"]} label="Zoom in/out" />
                  <ShortcutRow keys={["Shift", "Drag"]} label="Pan canvas" />
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ShortcutRow({ keys, label }: { keys: string[]; label: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-neutral-800/30 transition-colors duration-150">
      <span className="text-[11px] text-neutral-400 font-sans">{label}</span>
      <div className="flex items-center gap-0.5">
        {keys.map((k) => (
          <kbd
            key={k}
            className="inline-flex items-center justify-center min-w-[22px] h-5 px-1.5 bg-neutral-900/80 border border-neutral-800/60 rounded-md text-[10px] text-neutral-400 font-mono"
          >
            {k}
          </kbd>
        ))}
      </div>
    </div>
  );
}
