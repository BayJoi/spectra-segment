import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface NumberInputProps {
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  ariaLabel: string;
  className?: string;
}

export function NumberInput({ value, min, max, onChange, ariaLabel, className }: NumberInputProps) {
  const [draft, setDraft] = useState(String(value));
  const focusedRef = useRef(false);
  const cancelRef = useRef(false);

  useEffect(() => {
    if (!focusedRef.current) setDraft(String(value));
  }, [value]);

  const commit = () => {
    const parsed = Number(draft);
    if (draft.trim() !== "" && Number.isFinite(parsed)) {
      const next = Math.min(max, Math.max(min, Math.round(parsed)));
      setDraft(String(next));
      if (next !== value) onChange(next);
    } else {
      setDraft(String(value));
    }
  };

  return (
    <input
      type="number"
      inputMode="numeric"
      value={draft}
      min={min}
      max={max}
      aria-label={ariaLabel}
      onFocus={() => {
        focusedRef.current = true;
      }}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        focusedRef.current = false;
        if (cancelRef.current) {
          cancelRef.current = false;
          setDraft(String(value));
          return;
        }
        commit();
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        if (e.key === "Escape") {
          cancelRef.current = true;
          setDraft(String(value));
          e.currentTarget.blur();
        }
      }}
      className={cn(
        "h-6 w-8 shrink-0 px-0.5 rounded-md bg-neutral-950/60 border border-neutral-800/60 text-center text-[10px] text-neutral-300 font-sans tabular-nums transition-colors duration-150 focus:outline-none focus:border-orange-500/50 focus:text-orange-200 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none",
        className
      )}
    />
  );
}
