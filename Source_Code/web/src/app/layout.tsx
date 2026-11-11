import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PhishCheck | CS 552",
  description:
    "CS 552 phishing URL checker implementing adaptive drift-aware detection with online SGD.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
