import { useAtom } from "jotai";
import { toastAtom, type ToastMessage } from "@/store/ui";
import { cn } from "@/lib/utils";

function Toast({ toast }: { toast: ToastMessage }) {
  return (
    <div
      className={cn(
        "animate-drop-in pointer-events-auto flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-sans shadow-2xl shadow-black/50 backdrop-blur-xl grain-bg-strong max-w-md",
        toast.kind === "error"
          ? "bg-[#1a0d08]/95 border-red-500/30 text-red-300"
          : "bg-[#0a0a0a]/95 border-orange-500/30 text-orange-200"
      )}
      role="status"
    >
      {toast.kind === "error" && (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="shrink-0">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 8v4m0 4h.01" />
        </svg>
      )}
      <span>{toast.text}</span>
    </div>
  );
}

export function ToastHost() {
  const [toasts] = useAtom(toastAtom);
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-5 left-1/2 z-[120] -translate-x-1/2 flex flex-col items-center gap-2">
      {toasts.map((t) => (
        <Toast key={t.id} toast={t} />
      ))}
    </div>
  );
}
