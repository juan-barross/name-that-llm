# Name That LLM: TF-IDF de caracteres e palavras com regressão logística

O texto de cada modelo carrega marcas de estilo: pontuação, conectivos, aberturas e fechamentos. Este notebook combina TF-IDF de n-gramas de caracteres e de palavras numa regressão logística com pesos balanceados, porque a métrica é a balanced accuracy.

A validação é por prompt (GroupKFold), porque o teste só tem prompts que não aparecem no treino. Os experimentos e o código com testes estão em github.com/juan-barross/name-that-llm.
