import { useState, useRef, useLayoutEffect, useEffect, type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface TooltipProps {
  children: ReactNode;
  tip: string;
  side?: "top" | "bottom";
  className?: string;
}

export function Tooltip({ children, tip, side = "top", className }: TooltipProps) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState<{ side: "top" | "bottom"; x: number }>({ side: "top", x: 0 });
  const [positioned, setPositioned] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const clickingRef = useRef(false);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  useLayoutEffect(() => {
    if (!show) {
      setPositioned(false);
      return;
    }
    const wrapperEl = wrapperRef.current;
    const tooltipEl = tooltipRef.current;
    if (!wrapperEl || !tooltipEl) return;

    const wRect = wrapperEl.getBoundingClientRect();
    const tRect = tooltipEl.getBoundingClientRect();
    const gap = 8;

    let finalSide = side;
    if (finalSide === "top" && wRect.top - tRect.height - gap < 4) finalSide = "bottom";
    if (finalSide === "bottom" && wRect.bottom + tRect.height + gap > window.innerHeight - 4) finalSide = "top";

    let x = 0;
    const tooltipCenterX = wRect.left + wRect.width / 2;
    let tooltipLeft = tooltipCenterX - tRect.width / 2;
    let tooltipRight = tooltipCenterX + tRect.width / 2;

    if (tooltipLeft < 8) {
      x += 8 - tooltipLeft;
      tooltipLeft = 8;
      tooltipRight = tooltipLeft + tRect.width;
    }
    if (tooltipRight > window.innerWidth - 8) {
      x -= tooltipRight - (window.innerWidth - 8);
    }

    setPos({ side: finalSide, x });
    setPositioned(true);
  }, [show, side]);

  const open = () => {
    if (clickingRef.current) return;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setShow(true);
    }, 400);
  };

  const close = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setShow(false);
    setPositioned(false);
    clickingRef.current = false;
  };

  const onMouseDown = () => {
    clickingRef.current = true;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setShow(false);
    setPositioned(false);
  };

  return (
    <div
      ref={wrapperRef}
      className={cn("relative inline-flex", className)}
      onMouseEnter={open}
      onMouseLeave={close}
      onMouseDown={onMouseDown}
      onFocus={open}
      onBlur={close}
    >
      {children}
      {show && (
        <div
          ref={tooltipRef}
          role="tooltip"
          className={cn(
            "absolute z-50 pointer-events-none px-2.5 py-1.5 rounded-lg text-[10px] font-medium font-sans whitespace-nowrap",
            "bg-neutral-800 border border-neutral-700/50 text-neutral-300 shadow-xl shadow-black/40",
            "animate-fade-in",
            pos.side === "top"
              ? "bottom-full mb-2"
              : "top-full mt-2"
          )}
          style={{
            left: `calc(50% + ${pos.x}px)`,
            transform: "translateX(-50%)",
            visibility: positioned ? "visible" : "hidden",
          }}
        >
          {tip}
          <div
            className={cn(
              "absolute left-1/2 -translate-x-1/2 w-2 h-2 bg-neutral-800 border-neutral-700/50 rotate-45",
              pos.side === "top"
                ? "bottom-[-5px] border-r border-b"
                : "top-[-5px] border-l border-t"
            )}
          />
        </div>
      )}
    </div>
  );
}
