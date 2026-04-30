import Link from "next/link";
import { ArrowRight, BookOpen, Code2, Layers, Sparkles, Trophy } from "lucide-react";
import { getSectionTree, getStats } from "@/lib/content";

export default function HomePage() {
  const tree = getSectionTree();
  const stats = getStats();

  const projectCount = 15;
  const moduleCount = tree.length;

  return (
    <div className="space-y-20 pb-12">
      {/* HERO */}
      <section className="relative pt-12 lg:pt-20">
        <div className="absolute inset-x-0 -top-10 -z-10 h-64 bg-grid-glow" />
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-accent-violet/40 bg-accent-violet/10 px-3 py-1 text-xs font-medium text-accent-cyan">
            <Sparkles size={12} /> Codebasics presents · Bootcamp 3.0
          </div>
          <h1 className="text-4xl font-bold leading-tight tracking-tight text-white md:text-6xl lg:text-7xl">
            Gen&nbsp;AI &amp; Data&nbsp;Science
            <br />
            <span className="text-gradient">learn it. build it. ship it.</span>
          </h1>
          <p className="max-w-2xl text-lg text-slate-300 md:text-xl">
            Every lesson, project and chapter from the Codebasics Gen&nbsp;AI &amp; Data Science Bootcamp&nbsp;3.0 — rendered as a
            beautiful, browsable companion site. Python → SQL → Stats → ML → DL → NLP → Gen AI &amp; Agentic AI, plus career
            playbooks and 15 portfolio projects.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/introduction"
              className="btn-glow inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-accent-violet to-accent-cyan px-5 py-3 font-semibold text-white shadow-lg shadow-accent-violet/30"
            >
              Start the bootcamp <ArrowRight size={16} />
            </Link>
            <Link
              href="/python"
              className="inline-flex items-center gap-2 rounded-lg border border-ink-600 bg-ink-800/60 px-5 py-3 font-semibold text-slate-200 hover:border-accent-violet hover:text-white"
            >
              Jump to Python <Code2 size={16} />
            </Link>
            <Link
              href="/career"
              className="inline-flex items-center gap-2 rounded-lg border border-ink-600 bg-ink-800/60 px-5 py-3 font-semibold text-slate-200 hover:border-accent-violet hover:text-white"
            >
              Career track <Trophy size={16} />
            </Link>
          </div>

          <dl className="mt-10 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat icon={<Layers size={16} />} label="Modules" value={moduleCount.toString()} />
            <Stat icon={<BookOpen size={16} />} label="Lessons" value={stats.lessons.toString()} />
            <Stat icon={<Code2 size={16} />} label="Portfolio projects" value={projectCount.toString()} />
            <Stat icon={<Sparkles size={16} />} label="Pages" value={stats.totalPages.toString()} />
          </dl>
        </div>
      </section>

      {/* MODULE GRID */}
      <section className="space-y-6">
        <header className="flex items-end justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white md:text-3xl">The curriculum</h2>
            <p className="mt-1 text-slate-400">
              {moduleCount} sections · click any card to dive in.
            </p>
          </div>
        </header>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {tree.map((t, idx) => {
            const lessonCount = t.topLevelPages.length + t.subgroups.reduce((sum, sg) => sum + sg.pages.filter((p) => !p.isIndex).length, 0);
            return (
              <Link
                key={t.section.slug}
                href={`/${t.section.slug}`}
                className="glass group relative overflow-hidden rounded-xl p-5"
              >
                <div
                  className={`absolute -right-16 -top-16 h-40 w-40 rounded-full bg-gradient-to-br ${t.section.accent} opacity-20 blur-2xl transition-opacity group-hover:opacity-40`}
                />
                <div className="flex items-start justify-between">
                  <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-ink-800 text-2xl">
                    {t.section.emoji}
                  </div>
                  <span className="rounded-full bg-ink-800 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest text-slate-400">
                    {String(idx).padStart(2, "0")}
                  </span>
                </div>
                <h3 className="mt-4 text-lg font-semibold text-white group-hover:text-gradient">
                  {t.section.label}
                </h3>
                <p className="mt-1 line-clamp-3 text-sm text-slate-400">{t.section.description}</p>
                <div className="mt-4 flex items-center justify-between text-xs text-slate-500">
                  <span>
                    {lessonCount} {lessonCount === 1 ? "lesson" : "lessons"}
                    {t.subgroups.length > 0 && ` · ${t.subgroups.length} chapters`}
                  </span>
                  <span className="flex items-center gap-1 text-accent-cyan opacity-0 transition-opacity group-hover:opacity-100">
                    Open <ArrowRight size={12} />
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </section>

      {/* WHY THIS GUIDE */}
      <section className="space-y-6">
        <header>
          <h2 className="text-2xl font-bold text-white md:text-3xl">How to use this site</h2>
          <p className="mt-1 max-w-2xl text-slate-400">
            This isn&apos;t a marketing page — it&apos;s a working learning workspace. Every markdown file in the bootcamp
            repository is rendered here.
          </p>
        </header>
        <div className="grid gap-4 md:grid-cols-3">
          <Tip
            icon="📖"
            title="Before a lecture"
            body="Open the matching lesson, skim the goals and key concepts, then go watch the video."
          />
          <Tip
            icon="⌨️"
            title="During a lecture"
            body="Pause, type along — the lesson page has the syntax, edge cases and pitfalls you'll need."
          />
          <Tip
            icon="✅"
            title="After a lecture"
            body="Answer the self-check questions out loud; if anything is fuzzy, re-read the page."
          />
          <Tip
            icon="🛠️"
            title="For projects"
            body="Each project page lists business questions, datasets, the tech stack, and what 'done' looks like."
          />
          <Tip
            icon="📈"
            title="Weekly"
            body="Track your progress, push your code to GitHub, and write a build-in-public LinkedIn post."
          />
          <Tip
            icon="🎯"
            title="At the end"
            body="The career section walks you through resume, LinkedIn, mock interviews and the job application playbook."
          />
        </div>
      </section>

      {/* PROJECTS HIGHLIGHT */}
      <section className="space-y-6">
        <header>
          <h2 className="text-2xl font-bold text-white md:text-3xl">15 portfolio projects</h2>
          <p className="mt-1 text-slate-400">Real datasets, real domains — from hospitality to healthcare to credit risk.</p>
        </header>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {portfolioProjects.map((p) => (
            <div key={p.title} className="glass rounded-lg p-4">
              <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-widest">
                <span className="rounded bg-ink-800 px-2 py-0.5 text-slate-300">{p.module}</span>
                <span className="text-slate-500">{p.domain}</span>
              </div>
              <h4 className="mt-2 font-semibold text-white">{p.title}</h4>
              <p className="mt-1 text-sm text-slate-400">{p.tag}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="relative overflow-hidden rounded-2xl border border-ink-700/50 bg-gradient-to-br from-ink-900 via-ink-800/60 to-ink-900 p-8 md:p-12">
        <div className="absolute -right-20 -top-20 h-80 w-80 rounded-full bg-accent-violet/30 blur-3xl" />
        <div className="absolute -bottom-20 -left-20 h-80 w-80 rounded-full bg-accent-cyan/30 blur-3xl" />
        <div className="relative">
          <h2 className="max-w-xl text-3xl font-bold text-white md:text-4xl">
            Ready? Start with <span className="text-gradient">Welcome &amp; Fit</span>.
          </h2>
          <p className="mt-2 max-w-xl text-slate-300">
            The introduction module sets expectations — time commitment, learning method, projects and the build-in-public mindset.
            Don&apos;t skip it.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/introduction"
              className="btn-glow inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-accent-violet to-accent-cyan px-5 py-3 font-semibold text-white shadow-lg shadow-accent-violet/30"
            >
              Begin Module 0 <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="glass rounded-lg px-4 py-3">
      <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-slate-400">
        {icon} {label}
      </div>
      <div className="mt-1 text-2xl font-bold text-white">{value}</div>
    </div>
  );
}

function Tip({ icon, title, body }: { icon: string; title: string; body: string }) {
  return (
    <div className="glass rounded-lg p-4">
      <div className="text-2xl">{icon}</div>
      <h4 className="mt-2 font-semibold text-white">{title}</h4>
      <p className="mt-1 text-sm text-slate-400">{body}</p>
    </div>
  );
}

const portfolioProjects = [
  { title: "Hospitality Domain Data Analysis", module: "Python", domain: "Hospitality", tag: "Pandas + EDA on real hotel bookings." },
  { title: "Expense Tracking System", module: "Python", domain: "Personal Finance", tag: "FastAPI + Streamlit + MySQL fullstack app." },
  { title: "Finance & Top-N Insights", module: "SQL", domain: "Consumer Goods", tag: "UDFs, stored procedures, executive reporting." },
  { title: "Supply Chain Analytics & Optimisation", module: "SQL", domain: "Consumer Goods", tag: "Forecast accuracy + query tuning." },
  { title: "AtliQo Bank — Credit Card Launch", module: "Math & Stats", domain: "Banking", tag: "A/B testing on 50K+ records." },
  { title: "Healthcare Premium Prediction", module: "ML", domain: "Healthcare", tag: "Regression with Streamlit deployment." },
  { title: "Credit Risk Modelling", module: "ML", domain: "NBFC / Finance", tag: "WOE/IV, Optuna, KS / Gini metrics." },
  { title: "Beverage Price Range Prediction", module: "ML", domain: "FMCG", tag: "Multi-class classification on survey data." },
  { title: "Car Damage Detection (CNN)", module: "DL", domain: "Automotive", tag: "Transfer learning + Streamlit + FastAPI." },
  { title: "Real Estate Assistant (RAG)", module: "Gen AI", domain: "Real Estate", tag: "Chroma DB + LangChain + source citations." },
  { title: "E-Commerce Chatbot", module: "Gen AI", domain: "E-Commerce", tag: "Intent routing + live SQL + FAQ." },
  { title: "Agentic AI HR Onboarding", module: "Gen AI", domain: "HR Ops", tag: "MCP servers + Claude as the agent." },
  { title: "Customer Care AI Agent", module: "Gen AI", domain: "Telecom", tag: "Bedrock AgentCore — production memory + auth." },
  { title: "Stale Fruit Detector", module: "Internship 2", domain: "Agri-Tech", tag: "CNN for cold-storage warehouses." },
  { title: "RAG Q&A for Healthcare", module: "Internship 2", domain: "Healthcare", tag: "PubMed + Llama 3 over Streamlit." },
];
