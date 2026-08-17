import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import UnoCSS from "@unocss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), UnoCSS()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
    dedupe: ["react", "react-dom"],
  },
  server: {
    port: 3000,
    host: "localhost",
    strictPort: true,
    fs: {
      allow: [
        import.meta.dirname,
        path.resolve(import.meta.dirname, "..", "frontend"),
        path.resolve(import.meta.dirname, "..", "backend", "web"),
        path.resolve(import.meta.dirname, "..", "backend_amd_gpu", "web"),
      ],
    },
    headers: {
      "Content-Security-Policy":
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' http://127.0.0.1:8000 http://localhost:8000 ws://localhost:3000 ws://127.0.0.1:3000; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
      "X-Frame-Options": "DENY",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
      "Permissions-Policy":
        "camera=(), microphone=(), geolocation=(), usb=()",
    },
  },
  preview: {
    headers: {
      "Content-Security-Policy":
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' http://127.0.0.1:8000 http://localhost:8000; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
      "X-Frame-Options": "DENY",
      "X-Content-Type-Options": "nosniff",
      "Referrer-Policy": "no-referrer",
      "Permissions-Policy":
        "camera=(), microphone=(), geolocation=(), usb=()",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: ["es2020", "chrome99", "edge99", "firefox97", "safari15"],
    cssTarget: ["chrome99", "edge99", "firefox97", "safari15.4"],
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/react") || id.includes("node_modules/react-dom")) return "react-vendor";
          if (id.includes("node_modules/jotai")) return "jotai";
        },
      },
    },
  },
});
