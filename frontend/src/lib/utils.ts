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

const SERVER_WORKING_PIXELS = 4096 * 4096;
const SERVER_MAX_SIDE = 8192;

async function decodeImage(file: File): Promise<{ img: HTMLImageElement; url: string } | null> {
  const url = URL.createObjectURL(file);
  const img = new Image();
  try {
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("decode failed"));
      img.src = url;
    });
    return { img, url };
  } catch {
    URL.revokeObjectURL(url);
    return null;
  }
}

export async function fitImageFile(file: File): Promise<{ file: File; resized: boolean }> {
  const decoded = await decodeImage(file);
  if (!decoded) return { file, resized: false };
  const { img, url } = decoded;

  try {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const needsFit =
      w * h > SERVER_WORKING_PIXELS || Math.max(w, h) > SERVER_MAX_SIDE;
    if (!needsFit) return { file, resized: false };
    const scale = Math.min(
      1,
      Math.sqrt(SERVER_WORKING_PIXELS / Math.max(1, w * h)),
      SERVER_MAX_SIDE / Math.max(1, w, h),
    );
    const nw = Math.max(1, Math.floor(w * scale));
    const nh = Math.max(1, Math.floor(h * scale));
    const canvas = document.createElement("canvas");
    canvas.width = nw;
    canvas.height = nh;
    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, nw, nh);
    const mime = file.type === "image/png" ? "image/png" : file.type === "image/webp" ? "image/webp" : "image/jpeg";
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, mime, mime === "image/jpeg" ? 0.92 : undefined));
    if (!blob) return { file, resized: false };
    const fitted = new File([blob], file.name, { type: mime });
    return { file: fitted, resized: true };
  } catch {
    return { file, resized: false };
  } finally {
    URL.revokeObjectURL(url);
  }
}
