import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { createModel } from "../demo/ntl.js";

const root = new URL("..", import.meta.url);
const parityFile = new URL("runs/paridade.json", root);
const modelDir = new URL("demo/modelo/", root);

test("modelo em JavaScript repete as probabilidades do Python", { skip: !existsSync(parityFile) }, () => {
  const spec = JSON.parse(readFileSync(new URL("modelo.json", modelDir), "utf-8"));
  const weights = readFileSync(new URL("pesos.bin", modelDir));
  const model = createModel(spec, weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.length));
  const { textos, probabilidades } = JSON.parse(readFileSync(parityFile, "utf-8"));

  textos.forEach((text, i) => {
    const { probabilities } = model.predict(text);
    const gap = Math.max(...probabilities.map((p, j) => Math.abs(p - probabilidades[i][j])));
    assert.ok(gap < 1e-4, `texto ${i}: diferença ${gap}`);
  });
});
