import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, ArrowRight, ChevronRight, FileText, Layers } from "lucide-react";
import {
  findPage,
  getAdjacent,
  getAllRoutes,
  getSectionBySlug,
  getSectionTree,
} from "@/lib/content";
import { Markdown } from "../components/Markdown";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllRoutes();
}

type RouteParams = Promise<{ slug: string[] }>;

export async function generateMetadata({ params }: { params: RouteParams }) {
  const { slug } = await params;
  if (slug.length === 1) {
    const section = getSectionBySlug(slug[0]);
    if (section) return { title: `${section.label} · Codebasics Bootcamp 3.0` };
  }
  const page = findPage(slug);
  return {
    title: page ? `${page.title} · ${page.section.label}` : "Codebasics Bootcamp 3.0",
  };
}

export default async function DynamicPage({ params }: { params: RouteParams }) {
  const { slug } = await params;

  // Section landing route: /python, /sql, etc.
  if (slug.length === 1) {
    const section = getSectionBySlug(slug[0]);
    if (!section) return notFound();
    return <SectionLanding sectionSlug={slug[0]} />;
  }

  const page = findPage(slug);
  if (!page) return notFound();

  const { prev, next } = getAdjacent(page);

  return (
    <div className="space-y-10">
      <Breadcrumb page={page} />

      <header className="space-y-3 border-b border-ink-700/60 pb-6">
        <div className="flex items-center gap-2 text-xs">
          <span className="rounded-full bg-gradient-to-r from-accent-violet/20 to-accent-cyan/20 px-3 py-1 text-accent-cyan">
            {page.section.emoji} {page.section.label}
          </span>
          {page.parentSlug && (
            <span className="font-mono text-[11px] uppercase tracking-widest text-slate-500">
              {page.parentSlug.join(" / ")}
            </span>
          )}
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-white md:text-4xl">{page.title}</h1>
      </header>

      <Markdown source={page.raw} />

      <nav className="grid gap-3 border-t border-ink-700/60 pt-6 md:grid-cols-2">
        {prev ? (
          <Link
            href={prev.url}
            className="glass group flex items-center gap-3 rounded-lg p-4"
          >
            <ArrowLeft size={18} className="text-accent-cyan" />
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-widest text-slate-500">Previous</div>
              <div className="truncate font-medium text-slate-200 group-hover:text-white">
                {prev.title}
              </div>
            </div>
          </Link>
        ) : (
          <div />
        )}
        {next ? (
          <Link
            href={next.url}
            className="glass group flex items-center justify-end gap-3 rounded-lg p-4 text-right"
          >
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-widest text-slate-500">Next</div>
              <div className="truncate font-medium text-slate-200 group-hover:text-white">
                {next.title}
              </div>
            </div>
            <ArrowRight size={18} className="text-accent-cyan" />
          </Link>
        ) : null}
      </nav>
    </div>
  );
}

/** Section landing page: section README plus chapter / lesson grid. */
function SectionLanding({ sectionSlug }: { sectionSlug: string }) {
  const tree = getSectionTree().find((t) => t.section.slug === sectionSlug);
  if (!tree) return notFound();

  const lessonCount = tree.topLevelPages.length + tree.subgroups.reduce((s, g) => s + g.pages.filter((p) => !p.isIndex).length, 0);

  return (
    <div className="space-y-10">
      <Breadcrumb sectionOnly={tree.section} />

      <header className="space-y-4 border-b border-ink-700/60 pb-8">
        <div className="text-5xl">{tree.section.emoji}</div>
        <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">
          <span className={`bg-gradient-to-r ${tree.section.accent} bg-clip-text text-transparent`}>
            {tree.section.label}
          </span>
        </h1>
        <p className="max-w-2xl text-lg text-slate-300">{tree.section.description}</p>
        <div className="flex flex-wrap gap-3 text-sm text-slate-400">
          <span className="flex items-center gap-1.5">
            <FileText size={14} /> {lessonCount} lessons
          </span>
          {tree.subgroups.length > 0 && (
            <span className="flex items-center gap-1.5">
              <Layers size={14} /> {tree.subgroups.length} chapters
            </span>
          )}
        </div>
      </header>

      {tree.indexPage && (
        <section>
          <Markdown source={tree.indexPage.raw} />
        </section>
      )}

      {tree.topLevelPages.length > 0 && (
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold text-white">Lessons</h2>
          <div className="grid gap-3 md:grid-cols-2">
            {tree.topLevelPages.map((p, i) => (
              <Link
                key={p.url}
                href={p.url}
                className="glass group flex items-start gap-4 rounded-lg p-4"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink-800 font-mono text-xs text-accent-cyan">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  <div className="font-medium text-slate-100 group-hover:text-white">{p.title}</div>
                  <ArrowRight size={14} className="mt-1 text-slate-500 transition-transform group-hover:translate-x-1 group-hover:text-accent-cyan" />
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}

      {tree.subgroups.map((sg) => (
        <section key={sg.slug} className="space-y-4">
          <header>
            <h2 className="text-2xl font-semibold text-white">{sg.label}</h2>
          </header>
          <div className="grid gap-3 md:grid-cols-2">
            {sg.pages
              .filter((p) => !p.isIndex)
              .map((p, i) => (
                <Link
                  key={p.url}
                  href={p.url}
                  className="glass group flex items-start gap-4 rounded-lg p-4"
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink-800 font-mono text-xs text-accent-cyan">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div className="min-w-0">
                    <div className="font-medium text-slate-100 group-hover:text-white">{p.title}</div>
                    <ArrowRight size={14} className="mt-1 text-slate-500 transition-transform group-hover:translate-x-1 group-hover:text-accent-cyan" />
                  </div>
                </Link>
              ))}
            {sg.pages.find((p) => p.isIndex) && (
              <Link
                href={sg.pages.find((p) => p.isIndex)!.url}
                className="glass group flex items-start gap-4 rounded-lg border-dashed p-4"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink-800 font-mono text-xs text-accent-violet">
                  ★
                </span>
                <div className="min-w-0">
                  <div className="text-xs uppercase tracking-widest text-slate-500">Chapter overview</div>
                  <div className="font-medium text-slate-100 group-hover:text-white">
                    Read the {sg.label} README
                  </div>
                </div>
              </Link>
            )}
          </div>
        </section>
      ))}
    </div>
  );
}

function Breadcrumb({
  page,
  sectionOnly,
}: {
  page?: ReturnType<typeof findPage>;
  sectionOnly?: ReturnType<typeof getSectionBySlug>;
}) {
  const section = page?.section ?? sectionOnly;
  if (!section) return null;

  return (
    <div className="flex flex-wrap items-center gap-1 text-xs text-slate-500">
      <Link href="/" className="hover:text-accent-cyan">
        Home
      </Link>
      <ChevronRight size={12} />
      <Link href={`/${section.slug}`} className="hover:text-accent-cyan">
        {section.label}
      </Link>
      {page && page.parentSlug && (
        <>
          <ChevronRight size={12} />
          <span className="text-slate-400">{page.parentSlug.join(" / ")}</span>
        </>
      )}
      {page && (
        <>
          <ChevronRight size={12} />
          <span className="text-slate-300">{page.title}</span>
        </>
      )}
    </div>
  );
}
