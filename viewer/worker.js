const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Allow-Headers": "*",
};

function contentTypeFor(key) {
  if (key.endsWith(".json")) return "application/json";
  if (key.endsWith(".rad")) return "application/octet-stream";
  return "application/octet-stream";
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/r2/")) {
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: CORS });
      }

      const key = decodeURIComponent(url.pathname.slice(4));
      if (!key || key.includes("..") || key.includes("/")) {
        return new Response("Bad Request", { status: 400, headers: CORS });
      }

      const obj = await env.SAMPLES.get(key);
      if (!obj) {
        return new Response("Not Found", { status: 404, headers: CORS });
      }

      const headers = new Headers(CORS);
      headers.set(
        "Content-Type",
        obj.httpMetadata?.contentType || contentTypeFor(key)
      );
      headers.set("Cache-Control", "public, max-age=3600");
      if (obj.size != null) headers.set("Content-Length", String(obj.size));

      return new Response(obj.body, {
        headers,
        // Support Range for large .rad downloads when the client asks.
        status: 200,
      });
    }

    return env.ASSETS.fetch(request);
  },
};
