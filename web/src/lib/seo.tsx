import { useEffect } from "react";
import { useLocation } from "react-router-dom";

declare const __SITE_URL__: string;

function setMeta(selector: string, attr: "content" | "href", value: string) {
  const el = document.head.querySelector<HTMLMetaElement | HTMLLinkElement>(selector);
  if (el) el.setAttribute(attr, value);
}

/** Per-page title, description, canonical URL and social tags (client-side). */
export function Seo({ title, description, noindex = false }: { title: string; description: string; noindex?: boolean }) {
  const { pathname } = useLocation();
  useEffect(() => {
    const full = title === "GlassQuill" ? title : `${title} · GlassQuill`;
    const url = `${__SITE_URL__}${pathname}`;
    document.title = full;
    setMeta('meta[name="description"]', "content", description);
    setMeta('link[rel="canonical"]', "href", url);
    setMeta('meta[property="og:title"]', "content", full);
    setMeta('meta[property="og:description"]', "content", description);
    setMeta('meta[property="og:url"]', "content", url);
    setMeta('meta[name="twitter:title"]', "content", full);
    setMeta('meta[name="twitter:description"]', "content", description);
    let robots = document.head.querySelector<HTMLMetaElement>('meta[name="robots"]');
    if (noindex) {
      if (!robots) {
        robots = document.createElement("meta");
        robots.name = "robots";
        document.head.append(robots);
      }
      robots.content = "noindex";
    } else robots?.remove();
  }, [title, description, pathname, noindex]);
  return null;
}
