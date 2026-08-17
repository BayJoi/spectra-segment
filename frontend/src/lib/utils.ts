import { clsx } from "clsx";

export function cn(...inputs: (string | false | null | undefined)[]) {
  return clsx(inputs);
}

export const SUPPORTED_IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "webp"]);

const SUPPORTED_IMAGE_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);

export function isSupportedImage(file: File): boolean {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  return SUPPORTED_IMAGE_EXTENSIONS.has(ext) || SUPPORTED_IMAGE_MIME.has(file.type.toLowerCase());
}
