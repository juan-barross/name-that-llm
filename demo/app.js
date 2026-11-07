import { loadModel } from "./ntl.js";

const VIEWS = { char: "caracteres", word: "palavras", shape: "forma", function: "funcionais" };
const TOP = 5;

const form = document.querySelector("#form");
const input = document.querySelector("#texto");
const button = document.querySelector("#botao");
const counter = document.querySelector("#contagem");
const result = document.querySelector("#resultado");
const bars = document.querySelector("#barras");
const marks = document.querySelector("#marcas");

const wordCount = (text) => (text.match(/\S+/gu) ?? []).length;
const visible = (term) => term.replaceAll(" ", "·");

function item(tag, className, text) {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function renderBars({ labels, probabilities }) {
  const ranked = labels.map((label, i) => [label, probabilities[i]]).sort((a, b) => b[1] - a[1]);
  bars.replaceChildren(
    ...ranked.slice(0, TOP).map(([label, probability]) => {
      const row = document.createElement("li");
      const fill = item("span", "fill", "");
      fill.style.width = `${(probability * 100).toFixed(1)}%`;
      const track = item("span", "track", "");
      track.append(fill);
      row.append(item("span", "label", label), track, item("span", "value", `${(probability * 100).toFixed(1)}%`));
      return row;
    }),
  );
}

function renderMarks({ evidence }) {
  marks.replaceChildren(
    ...evidence.map(({ view, term }) => {
      const mark = document.createElement("li");
      mark.append(item("code", "term", visible(term)), item("span", "view", VIEWS[view]));
      return mark;
    }),
  );
}

input.addEventListener("input", () => {
  const words = wordCount(input.value);
  counter.textContent = words < 36 ? `${words} palavras · use pelo menos 36` : `${words} palavras`;
});

loadModel("modelo")
  .then((model) => {
    button.disabled = false;
    button.textContent = "Identificar";
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const prediction = model.predict(input.value);
      renderBars(prediction);
      renderMarks(prediction);
      result.hidden = false;
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  })
  .catch(() => {
    button.textContent = "Não foi possível carregar o modelo";
  });
