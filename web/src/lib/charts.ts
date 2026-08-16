import type { EChartsOption } from "echarts";
import type { ThemeName } from "../theme";

// Read resolved CSS tokens; pages remount charts when the theme changes.
export interface ChartTokens {
  theme: ThemeName;
  text: string;
  dim: string;
  faint: string;
  line: string;
  clay: string;
  grass: string;
  ice: string;
  raised: string;
  inset: string;
}

export function chartTokens(): ChartTokens {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    theme: document.documentElement.classList.contains("light")
      ? "light"
      : "dark",
    text: v("--text"),
    dim: v("--text-dim"),
    faint: v("--text-faint"),
    line: v("--line"),
    clay: v("--clay"),
    grass: v("--grass"),
    ice: v("--ice"),
    raised: v("--bg-raised"),
    inset: v("--bg-inset"),
  };
}

export function withAlpha(hex: string, alpha: number): string {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

// Shared accessible, themed chart options.
export function baseChartOption(t: ChartTokens): EChartsOption {
  return {
    aria: { enabled: true, decal: { show: true } },
    textStyle: { color: t.text },
    tooltip: {
      backgroundColor: t.raised,
      borderColor: t.line,
      borderWidth: 1,
      textStyle: { color: t.text, fontSize: 12 },
      extraCssText:
        "border-radius: 10px; box-shadow: 0 12px 32px rgba(0,0,0,0.35);",
    },
    legend: {
      textStyle: { color: t.dim, fontSize: 11 },
      itemWidth: 14,
      itemHeight: 8,
      icon: "roundRect",
    },
  };
}

// Clay surface stays muted because orange signals betting.
const SURFACE_FALLBACK: Record<string, string> = {
  clay: "#b98a63",
  grass: "#3fae7a",
  hard: "#5f9fc9",
  carpet: "#8d93ad",
};

export function surfaceColor(surface: string, t: ChartTokens): string {
  if (surface === "grass") return t.grass;
  if (surface === "hard") return t.ice;
  if (surface === "clay") return t.theme === "dark" ? "#c98d63" : "#a87850";
  return SURFACE_FALLBACK[surface] ?? t.dim;
}

export function axisOption(t: ChartTokens) {
  return {
    axisLine: { lineStyle: { color: t.line } },
    axisTick: { lineStyle: { color: t.line } },
    axisLabel: { color: t.dim },
    splitLine: { lineStyle: { color: t.line, type: "dashed" as const } },
  };
}
