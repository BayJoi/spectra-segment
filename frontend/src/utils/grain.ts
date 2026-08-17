const GRAIN_SIZE = 256;

export function initGrain(): void {
  const canvas = document.createElement("canvas");
  canvas.width = GRAIN_SIZE;
  canvas.height = GRAIN_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const imageData = ctx.createImageData(GRAIN_SIZE, GRAIN_SIZE);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.floor(Math.random() * 256);
    d[i] = v;
    d[i + 1] = v;
    d[i + 2] = v;
    d[i + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const url = canvas.toDataURL("image/png");
  document.documentElement.style.setProperty("--grain-url", `url(${url})`);
}
