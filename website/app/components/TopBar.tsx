"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X, Github, Sparkles } from "lucide-react";
import type { SectionTree } from "@/lib/content";
import { Sidebar } from "./Sidebar";

export function TopBar({ tree }: { tree: SectionTree[] }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-ink-700/60 bg-ink-950/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1400px] items-center justify-between px-4 lg:px-6">
        <div className="flex items-center gap-3">
          <button
            className="rounded-md p-2 text-slate-300 hover:bg-ink-800 lg:hidden"
            onClick={() => setMobileOpen(true)}
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>
          <Link href="/" className="flex items-center gap-2">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-gradient-to-br from-accent-violet to-accent-cyan shadow-lg shadow-accent-violet/30">
              <Sparkles size={18} className="text-white" />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-white">Bootcamp 3.0</div>
              <div className="text-[10px] uppercase tracking-widest text-slate-400">
                Gen AI &amp; Data Science
              </div>
            </div>
          </Link>
        </div>

        <nav className="hidden items-center gap-6 text-sm md:flex">
          <Link href="/" className="text-slate-300 hover:text-white">
            Home
          </Link>
          <Link href="/introduction" className="text-slate-300 hover:text-white">
            Introduction
          </Link>
          <Link href="/python" className="text-slate-300 hover:text-white">
            Modules
          </Link>
          <Link href="/career" className="text-slate-300 hover:text-white">
            Career
          </Link>
          <Link
            href="https://github.com/codebasics"
            target="_blank"
            className="flex items-center gap-1.5 rounded-md border border-ink-600 bg-ink-800 px-3 py-1.5 text-slate-300 hover:border-accent-violet hover:text-white"
          >
            <Github size={14} /> GitHub
          </Link>
        </nav>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm lg:hidden" onClick={() => setMobileOpen(false)}>
          <div
            className="absolute left-0 top-0 h-full w-[85%] max-w-sm overflow-y-auto bg-ink-900 p-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-center justify-between">
              <span className="text-sm font-semibold text-white">Curriculum</span>
              <button onClick={() => setMobileOpen(false)} className="rounded-md p-1.5 hover:bg-ink-800">
                <X size={18} />
              </button>
            </div>
            <Sidebar tree={tree} onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      )}
    </header>
  );
}
