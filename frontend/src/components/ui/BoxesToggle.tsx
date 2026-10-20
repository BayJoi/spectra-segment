import { Tooltip } from "@/components/ui/Tooltip";
import { cn } from "@/lib/utils";

interface BoxesToggleProps {
  hidden: boolean;
  onToggle: () => void;
  tip: string;
}

export function BoxesToggle({ hidden, onToggle, tip }: BoxesToggleProps) {
  return (
    <Tooltip tip={tip} className="shrink-0">
      <button
        type="button"
        onClick={onToggle}
        className={cn(
          "h-7 px-2 rounded-lg text-[10px] font-medium font-mono border transition-all duration-200 cursor-pointer whitespace-nowrap hover:scale-105 active:scale-95",
          hidden
            ? "bg-orange-500/15 border-orange-500/40 text-orange-300 hover:bg-orange-500/25"
            : "bg-neutral-900/80 border-neutral-800/60 text-neutral-500 hover:text-neutral-300 hover:border-neutral-700"
        )}
      >
        {hidden ? "Boxes off" : "Boxes"}
      </button>
    </Tooltip>
  );
}
