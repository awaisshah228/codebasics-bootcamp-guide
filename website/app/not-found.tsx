import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center text-center">
      <div className="text-6xl font-bold text-gradient">404</div>
      <h1 className="mt-2 text-2xl font-semibold text-white">Lesson not found</h1>
      <p className="mt-2 max-w-md text-slate-400">
        This page isn&apos;t in the bootcamp curriculum yet. Try the sidebar or head back to the home page.
      </p>
      <Link
        href="/"
        className="mt-6 inline-flex rounded-lg bg-gradient-to-r from-accent-violet to-accent-cyan px-5 py-2.5 font-semibold text-white"
      >
        ← Home
      </Link>
    </div>
  );
}
