import { useAtom } from "jotai";
import { unsupportedFileAtom } from "@/store/ui";
import { SUPPORTED_IMAGE_EXTENSIONS } from "@/lib/utils";

const SUPPORTED_FORMATS = Array.from(SUPPORTED_IMAGE_EXTENSIONS)
  .map((ext) => `.${ext}`)
  .join(", ");

export function UnsupportedFileDialog() {
  const [file, setFile] = useAtom(unsupportedFileAtom);

  if (!file) return null;

  const handleDismiss = () => setFile(null);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-label="Unsupported file format">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-md smooth-dialog"
        onClick={handleDismiss}
        style={{ animation: "none" }}
      />

      <div className="animate-drop-in relative z-50 w-full max-w-sm mx-4">
        <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-amber-500/20 via-amber-500/5 to-transparent pointer-events-none" />

        <div className="relative bg-[#0c0c0c] border border-neutral-800/60 rounded-2xl overflow-hidden shadow-2xl shadow-black/60 grain-bg">
          <div className="h-px bg-gradient-to-r from-transparent via-amber-500/40 to-transparent" />

          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-amber-500/20 to-amber-600/10 border border-amber-500/20 flex items-center justify-center shadow-lg shadow-amber-500/10">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="text-amber-400"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-neutral-100 font-sans">Unsupported file format</h3>
                  <p className="text-[10px] text-neutral-500 font-sans">{file.name}</p>
                </div>
              </div>
              <button
                onClick={handleDismiss}
                aria-label="Close unsupported file dialog"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-neutral-500 hover:text-neutral-200 hover:bg-neutral-800/60 hover:scale-110 active:scale-95 transition-all duration-200 cursor-pointer"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
              </button>
            </div>

            <div className="mb-3">
              <label className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans block mb-2">Supported formats</label>
              <p className="text-[11px] text-neutral-500 font-sans leading-relaxed">
                <span className="text-neutral-300">{file.extension ? `.${file.extension}` : "This"}</span>{" "}
                is not a supported image format. Please choose an image in {SUPPORTED_FORMATS}.
              </p>
            </div>

            <button
              onClick={handleDismiss}
              className="w-full h-9 rounded-xl text-xs font-semibold font-sans bg-gradient-to-r from-amber-500 to-amber-600 text-white hover:from-amber-400 hover:to-amber-500 hover:shadow-lg hover:shadow-amber-500/20 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer select-none grain-bg grain-bg-strong"
            >
              Got it
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
