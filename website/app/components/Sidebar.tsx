"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight } from "lucide-react";
import type { SectionTree } from "@/lib/content";

export function Sidebar({
  tree,
  onNavigate,
}: {
  tree: SectionTree[];
  onNavigate?: () => void;
}) {
  const pathname = usePathname();

  return (
    <nav className="space-y-1 py-6 pr-2 text-sm">
      <Link
        href="/"
        onClick={onNavigate}
        className={`block rounded-md px-3 py-1.5 ${
          pathname === "/" ? "bg-accent-violet/20 text-white" : "text-slate-300 hover:bg-ink-800/60 hover:text-white"
        }`}
      >
        Home
      </Link>

      {tree.map((t) => {
        const sectionUrl = `/${t.section.slug}`;
        const isActiveSection = pathname === sectionUrl || pathname.startsWith(sectionUrl + "/");
        return (
          <details
            key={t.section.slug}
            open={isActiveSection}
            className="group rounded-md"
          >
            <summary className="flex items-center justify-between rounded-md px-3 py-1.5 hover:bg-ink-800/60">
              <Link
                href={sectionUrl}
                onClick={onNavigate}
                className={`flex flex-1 items-center gap-2 ${
                  pathname === sectionUrl ? "text-white" : "text-slate-200"
                }`}
              >
                <span className="text-base leading-none">{t.section.emoji}</span>
                <span className="truncate font-medium">{t.section.short}</span>
              </Link>
              <ChevronRight size={14} className="chevron text-slate-500" />
            </summary>

            <div className="ml-2 mt-1 space-y-0.5 border-l border-ink-700/50 pl-3">
              {t.topLevelPages.map((p) => (
                <Link
                  key={p.url}
                  href={p.url}
                  onClick={onNavigate}
                  className={`block truncate rounded px-2 py-1 text-[13px] ${
                    pathname === p.url
                      ? "bg-accent-violet/20 text-white"
                      : "text-slate-400 hover:bg-ink-800/40 hover:text-slate-100"
                  }`}
                  title={p.title}
                >
                  {p.title}
                </Link>
              ))}

              {t.subgroups.map((sg) => {
                const subActive = pathname.startsWith(`/${t.section.slug}/${sg.slug}`);
                return (
                  <details key={sg.slug} open={subActive} className="rounded">
                    <summary className="flex items-center gap-1.5 rounded px-2 py-1 text-[13px] text-slate-300 hover:bg-ink-800/40">
                      <ChevronRight size={12} className="chevron text-slate-500" />
                      <span className="truncate font-medium">{sg.label}</span>
                    </summary>
                    <div className="ml-3 mt-0.5 space-y-0.5 border-l border-ink-700/40 pl-2">
                      {sg.pages.map((p) => (
                        <Link
                          key={p.url}
                          href={p.url}
                          onClick={onNavigate}
                          className={`block truncate rounded px-2 py-1 text-[12.5px] ${
                            pathname === p.url
                              ? "bg-accent-violet/20 text-white"
                              : "text-slate-400 hover:bg-ink-800/40 hover:text-slate-100"
                          }`}
                          title={p.title}
                        >
                          {p.isIndex ? "Overview" : p.title}
                        </Link>
                      ))}
                    </div>
                  </details>
                );
              })}
            </div>
          </details>
        );
      })}
    </nav>
  );
}
