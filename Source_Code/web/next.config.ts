import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: the detector runs entirely in the browser.
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
