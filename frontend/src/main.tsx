import "virtual:uno.css";
import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./globals.css";
import { initGrain } from "./utils/grain";

initGrain();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
