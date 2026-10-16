// Rebuilds ../static/vendor/cm6.js and copies the woff2 fonts + licenses.
// Usage (from this directory):  npm ci && npm run build
import { build } from "esbuild";
import { cpSync, mkdirSync, readdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const vendor = join(here, "..", "static", "vendor");
mkdirSync(vendor, { recursive: true });

await build({
  entryPoints: [join(here, "entry.js")],
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2020",
  legalComments: "eof",
  outfile: join(vendor, "cm6.js"),
});

// Fonts: latin subset only, a few static weights.
const fontsDir = join(vendor, "fonts");
mkdirSync(fontsDir, { recursive: true });
const fonts = [
  ["@fontsource/inter", "inter", [400, 500, 600, 700]],
  ["@fontsource/jetbrains-mono", "jetbrains-mono", [400, 500, 700]],
];
for (const [pkg, slug, weights] of fonts) {
  const files = join(here, "node_modules", pkg, "files");
  for (const w of weights) {
    const name = `${slug}-latin-${w}-normal.woff2`;
    cpSync(join(files, name), join(fontsDir, name));
  }
  const lic = join(here, "node_modules", pkg, "LICENSE");
  if (existsSync(lic)) cpSync(lic, join(fontsDir, `${slug}-LICENSE.txt`));
}
console.log("vendor build ok:", readdirSync(fontsDir).join(", "));
