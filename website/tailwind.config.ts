import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";

export default {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#05060f",
          900: "#0a0c1d",
          800: "#10132b",
          700: "#181c3d",
          600: "#252a4f",
        },
        accent: {
          DEFAULT: "#8b5cf6",
          cyan: "#22d3ee",
          violet: "#8b5cf6",
          fuchsia: "#d946ef",
          lime: "#a3e635",
          amber: "#fbbf24",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      typography: ({ theme }: { theme: (key: string) => string }) => ({
        invert: {
          css: {
            "--tw-prose-body": theme("colors.slate[300]"),
            "--tw-prose-headings": theme("colors.white"),
            "--tw-prose-lead": theme("colors.slate[300]"),
            "--tw-prose-links": theme("colors.accent.cyan"),
            "--tw-prose-bold": theme("colors.white"),
            "--tw-prose-counters": theme("colors.accent.violet"),
            "--tw-prose-bullets": theme("colors.accent.violet"),
            "--tw-prose-hr": theme("colors.ink[600]"),
            "--tw-prose-quotes": theme("colors.slate[200]"),
            "--tw-prose-quote-borders": theme("colors.accent.violet"),
            "--tw-prose-captions": theme("colors.slate[400]"),
            "--tw-prose-code": theme("colors.accent.cyan"),
            "--tw-prose-pre-code": theme("colors.slate[200]"),
            "--tw-prose-pre-bg": "#0b1020",
            "--tw-prose-th-borders": theme("colors.ink[600]"),
            "--tw-prose-td-borders": theme("colors.ink[700]"),
          },
        },
      }),
    },
  },
  plugins: [typography],
} satisfies Config;
