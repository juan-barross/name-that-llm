const WHITE_SPACES = /\s\s+/gu;
const LETTERS = /[\p{L}\p{Nl}\p{No}]+/gu;
const TOKENS = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;
const ALNUM = /^[\p{L}\p{N}]+$/u;
const NON_SPACE = /\S+/gu;
const EVIDENCE_PER_VIEW = 3;

function isUpper(character) {
  return character !== character.toLowerCase() && character === character.toUpperCase();
}

function wordShape(text) {
  return text.replace(LETTERS, (match) => (isUpper(String.fromCodePoint(match.codePointAt(0))) ? "X" : "x"));
}

function functionSkeleton(text, functionWords) {
  const tokens = text.match(TOKENS) ?? [];
  return tokens
    .map((token) => {
      const word = token.toLowerCase();
      return functionWords.has(word) || !ALNUM.test(token) ? word : "CONTENT";
    })
    .join(" ");
}

function charWordBoundNgrams(text, [low, high]) {
  const grams = [];
  for (const word of text.replace(WHITE_SPACES, " ").split(/\s+/u).filter(Boolean)) {
    const padded = Array.from(` ${word} `);
    for (let n = low; n <= high; n += 1) {
      let offset = 0;
      grams.push(padded.slice(offset, offset + n).join(""));
      while (offset + n < padded.length) {
        offset += 1;
        grams.push(padded.slice(offset, offset + n).join(""));
      }
      if (offset === 0) break;
    }
  }
  return grams;
}

function charNgrams(text, [low, high]) {
  const characters = Array.from(text.replace(WHITE_SPACES, " "));
  const grams = [];
  for (let n = low; n <= Math.min(high, characters.length); n += 1) {
    for (let i = 0; i + n <= characters.length; i += 1) grams.push(characters.slice(i, i + n).join(""));
  }
  return grams;
}

function wordNgrams(tokens, [low, high]) {
  const grams = [];
  for (let n = low; n <= Math.min(high, tokens.length); n += 1) {
    for (let i = 0; i + n <= tokens.length; i += 1) grams.push(tokens.slice(i, i + n).join(" "));
  }
  return grams;
}

function analyzer(name, functionWords) {
  const words = (text) => text.match(NON_SPACE) ?? [];
  const analyzers = {
    char: (text, range) => charWordBoundNgrams(text, range),
    word: (text, range) => wordNgrams(words(text.toLowerCase()), range),
    shape: (text, range) => charNgrams(wordShape(text), range),
    function: (text, range) => wordNgrams(words(functionSkeleton(text, functionWords)), range),
  };
  return analyzers[name];
}

function tfidf(grams, view) {
  const counts = new Map();
  for (const gram of grams) {
    const index = view.index.get(gram);
    if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
  }
  const values = [...counts].map(([index, count]) => [index, (1 + Math.log(count)) * view.idf[index]]);
  const norm = Math.sqrt(values.reduce((sum, [, value]) => sum + value * value, 0)) || 1;
  return values.map(([index, value]) => [view.offset + index, value / norm, view.nome, view.termos[index]]);
}

function softmax(logits) {
  const top = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - top));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function createModel(spec, weightBuffer) {
  const functionWords = new Set(spec.palavras_funcionais);
  let offset = 0;
  const views = spec.visoes.map((view) => {
    const prepared = {
      ...view,
      offset,
      index: new Map(view.termos.map((term, i) => [term, i])),
      analyze: analyzer(view.nome, functionWords),
    };
    offset += view.termos.length;
    return prepared;
  });
  const width = offset;
  const weights = new Int8Array(weightBuffer);

  function features(text) {
    return views.flatMap((view) => tfidf(view.analyze(text, view.ngramas), view));
  }

  function predict(text) {
    const active = features(text);
    const logits = spec.rotulos.map((_, row) => {
      const base = row * width;
      const dot = active.reduce((sum, [column, value]) => sum + value * weights[base + column], 0);
      return spec.intercepto[row] + spec.escalas[row] * dot;
    });
    const probabilities = softmax(logits);
    const best = probabilities.indexOf(Math.max(...probabilities));
    const ranked = active
      .map(([column, value, view, term]) => ({
        view,
        term,
        weight: value * weights[best * width + column] * spec.escalas[best],
      }))
      .filter(({ weight }) => weight > 0)
      .sort((a, b) => b.weight - a.weight);
    const evidence = views
      .flatMap((view) => ranked.filter((mark) => mark.view === view.nome).slice(0, EVIDENCE_PER_VIEW))
      .sort((a, b) => b.weight - a.weight);
    return { labels: spec.rotulos, probabilities, best, evidence };
  }

  return { predict };
}

export async function loadModel(base) {
  const [spec, weights] = await Promise.all([
    fetch(`${base}/modelo.json`).then((response) => response.json()),
    fetch(`${base}/pesos.bin`).then((response) => response.arrayBuffer()),
  ]);
  return createModel(spec, weights);
}
