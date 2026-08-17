import { defineConfig, presetUno } from "unocss";

export default defineConfig({
  presets: [presetUno()],
  shortcuts: {
    "transition-interactive":
      "transition-[color,background-color,border-color,transform,box-shadow,opacity]",
  },
  theme: {
    fontFamily: {
      sans: ['"Inter Variable"', '"Inter"', "ui-sans-serif", "system-ui", "sans-serif"],
      mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
    },
    colors: {
      orange: {
        400: "#d47534",
        500: "#b85c2a",
        600: "#96461d",
        950: "#221005",
      },
    },
  },
});
