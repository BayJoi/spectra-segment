import { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

export function SelectNative({
  value,
  onChange,
  options,
  className = "",
  placeholder,
  position = "auto",
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; icon?: ReactNode; hint?: string; section?: string; disabled?: boolean; loaded?: boolean }[];
  className?: string;
  placeholder?: string;
  position?: "auto" | "up" | "down";
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [direction, setDirection] = useState<"up" | "down">("down");
  const ref = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    if (!open) return;
    if (position === "auto" && ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setDirection(rect.bottom + 200 > window.innerHeight ? "up" : "down");
    } else if (position === "up") {
      setDirection("up");
    } else {
      setDirection("down");
    }
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open, position]);

  const selected = options.find((o) => o.value === value);
  const isUp = direction === "up";

  return (
    <div ref={ref} className={cn("relative", className)}>
      <button
        ref={buttonRef}
        onClick={() => !disabled && setOpen(!open)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") {
            e.preventDefault();
            if (!open) setOpen(true);
            requestAnimationFrame(() => {
              const options = optionRefs.current;
              const target = e.key === "ArrowDown" ? 0 : options.length - 1;
              options[target]?.focus();
            });
          } else if (e.key === "Escape" && open) {
            e.preventDefault();
            setOpen(false);
          }
        }}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cn(
          "w-full h-8 px-3 rounded-lg text-xs font-sans text-left transition-interactive duration-150 flex items-center justify-between gap-1.5 grain-bg",
          disabled
            ? "bg-neutral-900 border border-neutral-800 text-neutral-600 cursor-not-allowed opacity-60"
            : open
              ? "bg-neutral-800 border border-neutral-700 text-neutral-200 cursor-pointer"
              : "bg-neutral-900 border border-neutral-800 text-neutral-400 hover:text-neutral-200 hover:border-neutral-700 hover:scale-104 active:scale-95 cursor-pointer"
        )}
      >
        <span className="flex items-center gap-1.5 truncate min-w-0">
          {selected?.icon && <span className="shrink-0">{selected.icon}</span>}
          <span className="truncate">{selected?.label ?? placeholder ?? "Select..."}</span>
        </span>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          className={cn("shrink-0 transition-transform duration-200 text-neutral-500", open && "rotate-180")}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div
          className={cn(
            "animate-drop-in absolute left-0 w-full min-w-[200px] bg-neutral-900 border border-neutral-700 rounded-xl shadow-2xl z-50 overflow-hidden grain-bg",
            isUp ? "bottom-full mb-1" : "top-full mt-1"
          )}
        >
          <div className="max-h-[280px] overflow-y-auto custom-scrollbar py-1" role="listbox">
          {(() => {
            let lastSection: string | undefined;
            return options.map((opt, i) => {
              const showSection = opt.section && opt.section !== lastSection;
              if (opt.section) lastSection = opt.section;
              const isDisabled = opt.disabled === true || opt.value === value;
              return (
                <div key={opt.value}>
                  {showSection && (
                    <div className="px-3 pt-2 pb-1 text-[9px] font-medium uppercase tracking-widest text-neutral-600 font-sans select-none">
                      {opt.section}
                    </div>
                  )}
                  <div
                    ref={(el) => { optionRefs.current[i] = el; }}
                    tabIndex={-1}
                    role="option"
                    aria-selected={opt.value === value}
                    aria-disabled={isDisabled}
                    onKeyDown={isDisabled ? undefined : (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onChange(opt.value);
                        setOpen(false);
                      } else if (e.key === "ArrowDown") {
                        e.preventDefault();
                        optionRefs.current[i + 1]?.focus();
                      } else if (e.key === "ArrowUp") {
                        e.preventDefault();
                        optionRefs.current[i - 1]?.focus();
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        setOpen(false);
                        buttonRef.current?.focus();
                      }
                    }}
                    className={cn(
                      "group flex items-center gap-1.5 px-3 py-2 text-xs font-sans transition-all duration-150",
                      isDisabled
                        ? "cursor-default opacity-40 text-neutral-600"
                        : "cursor-pointer",
                      !isDisabled && opt.value === value && "bg-neutral-800/60 text-neutral-200",
                      !isDisabled && opt.value !== value && "text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200"
                    )}
                    onClick={isDisabled ? undefined : () => { onChange(opt.value); setOpen(false); }}
                  >
                    {opt.icon && <span className="shrink-0">{opt.icon}</span>}
                    <span className="flex-1 min-w-0">
                      <span className="block truncate">{opt.label}</span>
                      {opt.hint && (
                        <span className="block text-[10px] text-neutral-600 truncate leading-tight mt-0.5">{opt.hint}</span>
                      )}
                    </span>
                    {opt.loaded && (
                      <span className="shrink-0 inline-flex items-center gap-1 h-4 px-1.5 rounded-md text-[9px] font-semibold font-mono bg-green-500/15 border border-green-500/30 text-green-400">
                        <span className="w-1 h-1 rounded-full bg-green-400" />
                        Loaded
                      </span>
                    )}
                  </div>
                </div>
              );
            });
          })()}
          </div>
        </div>
      )}
    </div>
  );
}
