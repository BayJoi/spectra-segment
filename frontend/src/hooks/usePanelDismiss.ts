import { useEffect, type RefObject } from "react";

export function usePanelDismiss(
  open: boolean,
  onClose: () => void,
  panelRef: RefObject<HTMLElement | null>,
  ignoredRef?: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (panelRef.current && !panelRef.current.contains(target) && !ignoredRef?.current?.contains(target)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open, onClose, panelRef, ignoredRef]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);
}
