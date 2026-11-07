# Name That LLM: blend de n-gramas treinado com o Open Pool

O texto de cada modelo carrega marcas de estilo: pontuação, conectivos, aberturas e fechamentos. Este notebook junta três classificadores lineares por média geométrica das probabilidades:

- hashing de n-gramas de caracteres (1 a 5) com SGD, treinado no train mais 2.500 textos por modelo do Open Pool;
- TF-IDF de caracteres e de palavras com regressão logística;
- o mesmo TF-IDF com marcas de estilo contadas por regex.

No fim, a saída é empurrada em parte para uma distribuição equilibrada entre os 18 modelos (Sinkhorn), porque a métrica é a balanced accuracy.

A validação é por prompt (GroupKFold), porque o teste só tem prompts que não aparecem no treino. Os pesos do blend saíram dessa validação. O código, os testes e o registro dos experimentos estão em github.com/juan-barross/name-that-llm.
