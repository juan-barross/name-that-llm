# O que já se sabe sobre a competição (06/11/2026)

## Placar

Leaderboard público (cerca de 40% do teste, 4.936 equipes): 1º 0,747; 10º 0,719; 50º 0,684. Os melhores notebooks públicos ficam entre 0,65 e 0,67.

## O que dá resultado

- Dados externos valem mais que modelo. O "Open Pool" (211 mil passagens de 59 fontes públicas, discussão 752766) soma de 0,11 a 0,12. Ele traz `match_weight`, que reamostra o pool para a tipografia do treino (+0,026), e `prompt_canon` para agrupar por prompt.
- Sem dados externos, o melhor relatado é 0,583. TF-IDF de caracteres e palavras com regressão logística fica em 0,343 de CV agrupado e 0,353 no LB (discussão 750390).
- SVM de n-gramas treinado com o pool: de 0,361 para 0,423 no LB. É o membro diverso que mais ajuda no blend (tirá-lo custa 0,011).
- O topo público é um blend de 3 ou 4 LLMs com LoRA (Qwen3-14B, Qwen3-8B, Qwen2.5-14B, Llama-3.1-8B), cortados para 24 a 32 camadas, com mean-pool de camadas intermediárias e uma cabeça centrada por prompt. Um LoRA sozinho faz de 0,63 a 0,68, e treinar um 14B leva cerca de 2,3 h de H100.
- Camadas iniciais discriminam melhor que a última: no Qwen3-4B congelado, a camada 3 dá 0,391 e a última, 0,311.
- Balancear o prior na saída (Sinkhorn suave com peso de 0,3 a 0,45, ou EM de Saerens) soma cerca de 0,011 no LB.
- Estilometria junto com TF-IDF: de 0,345 para 0,372 no CV.

## Onde os modelos erram

- Metade dos erros é dentro da mesma família: gemini 3.7 contra 3.8, gpt-5.6 sol contra terra.
- O GLM-5.3 sai como Claude mais vezes do que como ele mesmo.
- Claude e GLM usam travessão e aspas retas; GPT e Gemini, apóstrofo curvo. Dentro da família a pontuação quase não muda.
- Abaixo de 60 palavras o acerto é 0,27; acima de 160, 0,46.

## O que não fazer

- Agrupar o teste por prompt e impor um modelo por resposta (algoritmo húngaro): ganha 0,10 no CV e perde de 0,009 a 0,017 no LB.
- KFold aleatório (0,217 contra 0,369 do agrupado). Validar sempre por prompt.
- Mexer no texto (lowercase, limpeza) ou truncar passagens para aumentar dados.
- Usar dados externos sem casar o formato (quebras de linha, negrito, textos longos derrubam o TF-IDF).
- Confiar no OOF de modelos treinados no pool: fica de 0,03 a 0,04 otimista.
- Ler ganho em diferença menor que 0,01 no LB: o mesmo código varia até 0,009 entre envios.

## Validação

O OOF medido com só 2 respostas por prompt (m=2) teve correlação de 0,91 com o LB em 10 submissões, contra 0,13 com as 18 respostas. O ruído entre duas submissões é cerca de raiz(f / n_pub), em que f é a fração de linhas em que só uma das duas acerta.

## Fontes

- Leaderboard: kaggle.com/competitions/name-that-llm/leaderboard
- Discussões: 749447, 750390, 750462, 750664, 751011, 752766, 753177, 754860, 755314 em kaggle.com/competitions/name-that-llm/discussion/
- Notebook "0.669 All Public + What Failed": kaggle.com/code/alexchilton/name-that-llm-0-669-all-public-what-failed
- LLM-DAIGT, 1º lugar: github.com/nhyha/llm-detect-ai
- SemEval-2024 Task 8B: arxiv.org/abs/2404.00950
- Idiossincrasias de LLMs: arxiv.org/abs/2502.12150
- EM de Saerens para mudança de prior: direct.mit.edu/neco/article/14/1/21/6582
