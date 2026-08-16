import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import React from "react";
import ReactDOM from "react-dom/client";
import { router } from "./router";
import { applyTheme, ThemeProvider } from "./theme";
import "./index.css";

// Keep the app theme synchronized with the pre-paint inline choice.
applyTheme(resolveInitialTheme());

function resolveInitialTheme(): "dark" | "light" {
  const stored = localStorage.getItem("tm-theme");
  if (stored === "dark" || stored === "light") return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: Infinity, gcTime: Infinity, retry: 1 },
  },
});

const root = document.getElementById("root");
if (!root) throw new Error("root element not found");
ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
