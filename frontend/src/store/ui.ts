import { atom, getDefaultStore } from "jotai";
import { atomWithStorage } from "jotai/utils";
import { emitUi } from "./logs";

export const settingsOpenAtom = atom(false);
export const exportOpenAtom = atom(false);
export const showTransparentAtom = atom(false);
export const hideBboxesAtom = atom(false);
export const endSessionOpenAtom = atom(false);

export const unsupportedFileAtom = atom<{ name: string; extension: string } | null>(null);

export const modeLockAtom = atom<"brush" | "detect" | null>(null);
export const modeDialogOpenAtom = atom(false);

export const modeSwitchTargetAtom = atom<"sam3" | "brush" | null>(null);

export const isExportingAtom = atom(false);

export const imageEncodingAtom = atom(false);
export const encodingMessageAtom = atom<string | null>(null);

export const fitToViewAtom = atom<(() => void) | null>(null);
export const zoomInAtom = atom<(() => void) | null>(null);
export const zoomOutAtom = atom<(() => void) | null>(null);

export const uploadHoveredAtom = atom(false);

export const consoleOpenAtom = atom(false);
export type ConsoleFilterKey = "all" | "debug" | "info" | "warn" | "error";
export const consoleFilterAtom = atomWithStorage<ConsoleFilterKey>("consoleFilter", "all");
export type ConsoleSourceKey = "all" | "ui" | "backend";
export const consoleSourceAtom = atomWithStorage<ConsoleSourceKey>("consoleSource", "all");
export const consoleCategoryAtom = atomWithStorage<string>("consoleCategory", "all");
export const consoleShowTimeAtom = atomWithStorage("consoleShowTime", false);

export interface ToastMessage {
  id: number;
  text: string;
  kind: "error" | "info";
}
export const toastAtom = atom<ToastMessage[]>([]);
let toastSeq = 0;
export function pushToast(text: string, kind: "error" | "info" = "error") {
  const store = getDefaultStore();
  const id = ++toastSeq;
  store.set(toastAtom, [...store.get(toastAtom).slice(-2), { id, text, kind }]);
  emitUi("console", kind === "error" ? "ERROR" : "INFO", text);
  setTimeout(() => {
    const s = getDefaultStore();
    s.set(toastAtom, s.get(toastAtom).filter((t) => t.id !== id));
  }, 4000);
}

