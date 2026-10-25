import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Canonical links, Open Graph tags, sitemap.xml and robots.txt need the site's absolute URL.
// By default they keep a %SITE_URL% placeholder that server.py fills in, so one build works
// on any host; set SITE_URL only to host the files somewhere else.
const SITE = process.env.SITE_URL?.replace(/\/$/, "") ?? "%SITE_URL%";
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
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { target: "es2020", cssCodeSplit: true },
});
