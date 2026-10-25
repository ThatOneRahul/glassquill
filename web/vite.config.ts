import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Absolute URL the site is served from. Set SITE_URL when deploying; canonical links,
// Open Graph tags, sitemap.xml and robots.txt are all built from it.
const SITE = (process.env.SITE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const PAGES = ["/", "/detect", "/results", "/method"];

function seoFiles(): Plugin {
  return {
    name: "glassquill-seo-files",
    transformIndexHtml: (html) => html.replaceAll("%SITE_URL%", SITE),
    generateBundle() {
      const today = new Date().toISOString().slice(0, 10);
      const urls = PAGES.map(
        (p) => `  <url><loc>${SITE}${p}</loc><lastmod>${today}</lastmod></url>`,
      ).join("\n");
      this.emitFile({
        type: "asset",
        fileName: "sitemap.xml",
        source: `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`,
      });
      this.emitFile({
        type: "asset",
        fileName: "robots.txt",
        source: `User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: ${SITE}/sitemap.xml\n`,
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), seoFiles()],
  define: { __SITE_URL__: JSON.stringify(SITE) },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { target: "es2020", cssCodeSplit: true },
});
