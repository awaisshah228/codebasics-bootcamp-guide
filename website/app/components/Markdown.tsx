import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";

export function Markdown({ source }: { source: string }) {
  return (
    <article className="prose prose-invert max-w-none prose-headings:font-semibold prose-h1:text-3xl md:prose-h1:text-4xl prose-h2:mt-10 prose-h2:border-b prose-h2:border-ink-700 prose-h2:pb-2 prose-a:text-accent-cyan prose-strong:text-white prose-img:rounded-lg">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[
          rehypeSlug,
          [rehypeAutolinkHeadings, { behavior: "wrap" }],
          [rehypeHighlight, { detect: true, ignoreMissing: true }],
        ]}
      >
        {source}
      </ReactMarkdown>
    </article>
  );
}
