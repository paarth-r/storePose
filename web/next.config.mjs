/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export served by the Python dashboard server in production.
  output: "export",
  images: { unoptimized: true },
  // Dev only: proxy the data endpoints to the running storePose process so the
  // browser sees them same-origin (no CORS). Ignored by `next build` (export).
  async rewrites() {
    if (process.env.NODE_ENV !== "development") return [];
    const target = process.env.STOREPOSE_ORIGIN || "http://127.0.0.1:8000";
    return [
      { source: "/metrics", destination: `${target}/metrics` },
      { source: "/stream", destination: `${target}/stream` },
    ];
  },
};

export default nextConfig;
