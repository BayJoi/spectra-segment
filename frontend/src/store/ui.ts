import { atom } from "jotai";

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

export const fitToViewAtom = atom<(() => void) | null>(null);
export const zoomInAtom = atom<(() => void) | null>(null);
export const zoomOutAtom = atom<(() => void) | null>(null);

export const uploadHoveredAtom = atom(false);
