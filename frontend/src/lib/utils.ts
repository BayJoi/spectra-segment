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

export const SERVER_WORKING_PIXELS = 4096 * 4096;
export const SERVER_MAX_SIDE = 8192;

export async function fileNeedsServerResize(file: File): Promise<boolean> {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    const loaded = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("decode failed"));
      img.src = url;
    });
    await loaded;
    return img.naturalWidth * img.naturalHeight > SERVER_WORKING_PIXELS;
  } catch {
    return false;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function fitImageFile(file: File): Promise<{ file: File; resized: boolean }> {
  const needsFit =
    (await fileNeedsServerResize(file)) ||
    (await new Promise<boolean>((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        resolve(Math.max(img.naturalWidth, img.naturalHeight) > SERVER_MAX_SIDE);
        URL.revokeObjectURL(url);
      };
      img.onerror = () => {
        resolve(false);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    }));
  if (!needsFit) return { file, resized: false };

  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("decode failed"));
      img.src = url;
    });
    const w = img.naturalWidth;
    const h = img.naturalHeight;
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
