import type { Metadata } from "next";
import "./globals.css";
import "highlight.js/styles/github-dark.css";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { getSectionTree } from "@/lib/content";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Codebasics — Gen AI & Data Science Bootcamp 3.0",
  description:
    "Companion site for the Codebasics Gen AI & Data Science Bootcamp 3.0 — every lesson, project and module rendered as browsable, beautiful documentation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const tree = getSectionTree();
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-ink-950 text-slate-200 antialiased">
        <div className="aurora" aria-hidden />
        <TopBar tree={tree} />
        <div className="mx-auto flex max-w-[1400px] gap-0 px-4 lg:px-6">
          <aside className="sticky top-16 hidden h-[calc(100vh-4rem)] w-72 shrink-0 overflow-y-auto border-r border-ink-700/50 pr-4 lg:block">
            <Sidebar tree={tree} />
          </aside>
          <main className="min-w-0 flex-1 px-0 py-8 lg:px-10">{children}</main>
        </div>
        <footer className="mt-24 border-t border-ink-700/50 py-10 text-center text-sm text-slate-500">
          <p>
            Companion site for the official{" "}
            <Link
              href="https://codebasics.io/bootcamps/gen-ai-data-science-bootcamp-with-virtual-internship"
              className="text-accent-cyan hover:underline"
              target="_blank"
            >
              Codebasics Gen AI &amp; Data Science Bootcamp 3.0
            </Link>
            . Curriculum © Codebasics. Lesson notes maintained by the learner.
          </p>
        </footer>
      </body>
    </html>
  );
}
