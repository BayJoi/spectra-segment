import { cn } from "@/lib/utils";

interface DialogFooterProps {
  busy: boolean;
  tone: "danger" | "accent";
  confirmLabel: string;
  busyLabel: string;
  onCancel: () => void;
  onConfirm: () => void;
}

const CANCEL_CLASS =
  "flex-1 h-9 rounded-xl text-xs font-medium font-sans bg-neutral-900/50 border border-neutral-800/50 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700 hover:bg-neutral-800/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100";

const CONFIRM_CLASS =
  "flex-1 h-9 rounded-xl text-xs font-semibold font-sans text-white hover:shadow-lg hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer select-none disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 grain-bg grain-bg-strong";

const TONES = {
  danger: "bg-gradient-to-r from-red-500 to-red-600 hover:from-red-400 hover:to-red-500 hover:shadow-red-500/20",
  accent:
    "bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-400 hover:to-orange-500 hover:shadow-orange-500/20",
};

export function DialogFooter({
  busy,
  tone,
  confirmLabel,
  busyLabel,
  onCancel,
  onConfirm,
}: DialogFooterProps) {
  return (
    <div className="flex gap-2">
      <button onClick={onCancel} disabled={busy} className={CANCEL_CLASS}>
        Cancel
      </button>
      <button onClick={onConfirm} disabled={busy} className={cn(CONFIRM_CLASS, TONES[tone])}>
        {busy ? (
          <span className="inline-flex items-center justify-center gap-2">
            <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
            {busyLabel}
          </span>
        ) : (
          confirmLabel
        )}
      </button>
    </div>
  );
}
