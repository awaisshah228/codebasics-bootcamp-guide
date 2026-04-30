import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

// The bootcamp markdown lives one level above the website folder.
const ROOT = path.resolve(process.cwd(), "..");

// Sections to surface on the site, in display order. Each entry maps a folder
// on disk to a friendly URL slug and label.
const SECTIONS: ReadonlyArray<{
  dir: string;
  slug: string;
  label: string;
  short: string;
  emoji: string;
  accent: string;
  description: string;
}> = [
  {
    dir: "core/00-bootcamp-introduction",
    slug: "introduction",
    label: "Welcome to the Bootcamp",
    short: "Intro",
    emoji: "🚀",
    accent: "from-amber-400 to-orange-500",
    description: "How the bootcamp works, time commitment, projects, soft skills, and the build-in-public mindset.",
  },
  {
    dir: "core/01-python",
    slug: "python",
    label: "Python",
    short: "Python",
    emoji: "🐍",
    accent: "from-yellow-400 to-amber-500",
    description: "Beginner to advanced Python, plus two real builds: hospitality EDA and a fullstack expense tracker.",
  },
  {
    dir: "core/02-online-credibility",
    slug: "online-credibility",
    label: "Online Credibility",
    short: "Credibility",
    emoji: "🌟",
    accent: "from-pink-500 to-rose-500",
    description: "Discord, LinkedIn, GitHub, OSS — building proof of work that hiring managers actually see.",
  },
  {
    dir: "core/03-build-in-public",
    slug: "build-in-public",
    label: "Build in Public",
    short: "Build",
    emoji: "🛠️",
    accent: "from-violet-500 to-fuchsia-500",
    description: "Git fundamentals, GitHub anatomy, collaboration, finding OSS projects to contribute to.",
  },
  {
    dir: "core/04-sql",
    slug: "sql",
    label: "SQL for Data Science",
    short: "SQL",
    emoji: "🗄️",
    accent: "from-emerald-400 to-teal-500",
    description: "Beginner through advanced SQL with storytelling — joins, CTEs, window functions, MLOps-adjacent topics.",
  },
  {
    dir: "core/05-math-statistics",
    slug: "math-statistics",
    label: "Math & Statistics",
    short: "Math",
    emoji: "📊",
    accent: "from-fuchsia-500 to-purple-600",
    description: "Descriptive and inferential statistics, with the AtliQo Bank credit-card project.",
  },
  {
    dir: "core/06-machine-learning",
    slug: "machine-learning",
    label: "Machine Learning",
    short: "ML",
    emoji: "🤖",
    accent: "from-blue-500 to-indigo-600",
    description: "Foundations, classification, ensembles, unsupervised, MLOps lifecycle, and end-to-end projects.",
  },
  {
    dir: "core/07-deep-learning",
    slug: "deep-learning",
    label: "Deep Learning",
    short: "DL",
    emoji: "🧠",
    accent: "from-cyan-400 to-sky-600",
    description: "Neural networks, CNNs, RNNs, transformers — with PyTorch and Hugging Face.",
  },
  {
    dir: "core/08-nlp",
    slug: "nlp",
    label: "Natural Language Processing",
    short: "NLP",
    emoji: "💬",
    accent: "from-rose-500 to-pink-500",
    description: "Tokenization, embeddings, classification, BERT — the modern NLP stack end to end.",
  },
  {
    dir: "core/09-gen-ai-agentic-ai",
    slug: "gen-ai",
    label: "Gen AI & Agentic AI",
    short: "Gen AI",
    emoji: "✨",
    accent: "from-violet-500 to-fuchsia-500",
    description: "LLMs, RAG, vector DBs, MCP, multi-agent systems, LangChain / LangGraph / CrewAI / Bedrock AgentCore.",
  },
  {
    dir: "career",
    slug: "career",
    label: "Career Track",
    short: "Career",
    emoji: "🎯",
    accent: "from-amber-400 to-pink-500",
    description: "Online credibility, build-in-public, and the smart job assistance portal.",
  },
];

export type Section = (typeof SECTIONS)[number];

export type Page = {
  slug: string[]; // url segments after section, e.g. ["basics", "installation"]
  fullSlug: string[]; // including section slug, e.g. ["python", "basics", "installation"]
  url: string; // /python/basics/installation
  title: string;
  filePath: string; // absolute fs path
  section: Section;
  parentSlug?: string[]; // for grouping — section + sub
  isIndex: boolean; // true for README.md / index pages
  order: number;
  raw: string;
  frontmatter: Record<string, unknown>;
};

export type SubGroup = {
  slug: string; // sub folder slug, e.g. "basics"
  label: string; // human label
  pages: Page[];
};

export type SectionTree = {
  section: Section;
  indexPage?: Page; // README.md at section root
  topLevelPages: Page[]; // direct .md files at section root (excluding README)
  subgroups: SubGroup[]; // nested folders
};

/** Strip a leading "NN-" numeric prefix from a path segment. */
function stripPrefix(seg: string): string {
  return seg.replace(/^\d+[-_]/, "");
}

/** Extract numeric prefix as a sort key. README always sorts first. */
function orderOf(name: string): number {
  if (/^README\.md$/i.test(name)) return -1;
  const m = name.match(/^(\d+)/);
  return m ? parseInt(m[1], 10) : 999;
}

/** Pretty title: from first H1 in markdown, else from filename. */
function deriveTitle(raw: string, filename: string): string {
  const m = raw.match(/^#\s+(.+?)\s*$/m);
  if (m) return m[1].replace(/[—–-]\s*$/, "").trim();
  const stem = filename.replace(/\.md$/i, "");
  if (/^README$/i.test(stem)) return "Overview";
  return stripPrefix(stem)
    .split(/[-_]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function readFile(absPath: string): { raw: string; frontmatter: Record<string, unknown> } {
  const buf = fs.readFileSync(absPath, "utf8");
  const parsed = matter(buf);
  return { raw: parsed.content, frontmatter: parsed.data as Record<string, unknown> };
}

/** Walk a section directory and produce all pages. */
function scanSection(section: Section): Page[] {
  const absDir = path.join(ROOT, section.dir);
  if (!fs.existsSync(absDir)) return [];

  const pages: Page[] = [];

  const walk = (dir: string, urlSegs: string[]) => {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    // Files first
    for (const e of entries) {
      if (!e.isFile() || !e.name.toLowerCase().endsWith(".md")) continue;
      const abs = path.join(dir, e.name);
      const { raw, frontmatter } = readFile(abs);
      const isIndex = /^README\.md$/i.test(e.name);
      const stem = stripPrefix(e.name.replace(/\.md$/i, ""));
      const slug = isIndex ? [...urlSegs] : [...urlSegs, stem.toLowerCase()];
      const fullSlug = [section.slug, ...slug];
      pages.push({
        slug,
        fullSlug,
        url: "/" + fullSlug.join("/"),
        title: deriveTitle(raw, e.name),
        filePath: abs,
        section,
        isIndex,
        order: orderOf(e.name),
        raw,
        frontmatter,
        parentSlug: urlSegs.length ? urlSegs : undefined,
      });
    }

    // Then subdirectories
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      if (e.name.startsWith(".")) continue; // skip .ipynb_checkpoints etc.
      walk(path.join(dir, e.name), [...urlSegs, stripPrefix(e.name).toLowerCase()]);
    }
  };

  walk(absDir, []);
  return pages;
}

let _allPagesCache: Page[] | null = null;
let _treeCache: SectionTree[] | null = null;

export function getAllPages(): Page[] {
  if (_allPagesCache) return _allPagesCache;
  const all: Page[] = [];
  for (const section of SECTIONS) {
    all.push(...scanSection(section));
  }
  _allPagesCache = all;
  return all;
}

export function getSections(): readonly Section[] {
  return SECTIONS;
}

export function getSectionTree(): SectionTree[] {
  if (_treeCache) return _treeCache;
  const trees: SectionTree[] = [];
  for (const section of SECTIONS) {
    const pages = scanSection(section);
    const indexPage = pages.find((p) => p.isIndex && p.slug.length === 0);
    const topLevelPages = pages
      .filter((p) => !p.isIndex && p.slug.length === 1)
      .sort((a, b) => a.order - b.order);

    // Group nested pages by first segment of slug
    const subMap = new Map<string, Page[]>();
    for (const p of pages) {
      if (p.slug.length >= 2 || (p.isIndex && p.slug.length === 1)) {
        const key = p.slug[0];
        if (!subMap.has(key)) subMap.set(key, []);
        subMap.get(key)!.push(p);
      }
    }
    const subgroups: SubGroup[] = [];
    for (const [slug, subPages] of subMap) {
      // Use README order if present, else first page's order
      const indexInSub = subPages.find((p) => p.isIndex);
      const sortedPages = [...subPages].sort((a, b) => a.order - b.order);
      const label = indexInSub
        ? indexInSub.title
        : slug
            .split(/[-_]/)
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
      subgroups.push({ slug, label, pages: sortedPages });
    }
    // Sort subgroups in disk order — read original folder names
    const absDir = path.join(ROOT, section.dir);
    if (fs.existsSync(absDir)) {
      const folderOrder = fs
        .readdirSync(absDir, { withFileTypes: true })
        .filter((d) => d.isDirectory() && !d.name.startsWith("."))
        .map((d) => ({ slug: stripPrefix(d.name).toLowerCase(), order: orderOf(d.name) }));
      const orderMap = new Map(folderOrder.map((f) => [f.slug, f.order]));
      subgroups.sort((a, b) => (orderMap.get(a.slug) ?? 999) - (orderMap.get(b.slug) ?? 999));
    }

    trees.push({ section, indexPage, topLevelPages, subgroups });
  }
  _treeCache = trees;
  return trees;
}

export function findPage(slugSegments: string[]): Page | undefined {
  if (slugSegments.length === 0) return undefined;
  const all = getAllPages();
  const target = slugSegments.join("/");
  return all.find((p) => p.fullSlug.join("/") === target);
}

export function getSectionBySlug(slug: string): Section | undefined {
  return SECTIONS.find((s) => s.slug === slug);
}

/**
 * Returns flat list of all pages including section landing routes.
 * Section landing route = section.slug only (e.g., /python). Renders the
 * section README if present, else falls back to a section index page.
 */
export function getAllRoutes(): { slug: string[] }[] {
  const routes: { slug: string[] }[] = [];
  for (const section of SECTIONS) {
    routes.push({ slug: [section.slug] }); // landing
  }
  for (const p of getAllPages()) {
    if (p.fullSlug.length > 1) {
      routes.push({ slug: p.fullSlug });
    }
  }
  return routes;
}

/** Adjacent (prev/next) lessons in reading order across the whole bootcamp. */
export function getAdjacent(page: Page): { prev?: Page; next?: Page } {
  const ordered = getOrderedReadingList();
  const idx = ordered.findIndex((p) => p.url === page.url);
  if (idx < 0) return {};
  return {
    prev: idx > 0 ? ordered[idx - 1] : undefined,
    next: idx < ordered.length - 1 ? ordered[idx + 1] : undefined,
  };
}

export function getOrderedReadingList(): Page[] {
  const trees = getSectionTree();
  const out: Page[] = [];
  for (const tree of trees) {
    if (tree.indexPage) out.push(tree.indexPage);
    for (const p of tree.topLevelPages) out.push(p);
    for (const sub of tree.subgroups) {
      for (const p of sub.pages) out.push(p);
    }
  }
  return out;
}

/** Counts for the home page. */
export function getStats() {
  const trees = getSectionTree();
  const pages = getAllPages();
  return {
    sections: trees.length,
    lessons: pages.filter((p) => !p.isIndex).length,
    indexPages: pages.filter((p) => p.isIndex).length,
    totalPages: pages.length,
  };
}
