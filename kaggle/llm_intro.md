# Name That LLM: Qwen3-4B com LoRA como classificador (rodada A)

Ajusta o Qwen3-4B-Base em 4 bits, com LoRA em todas as projeções de atenção e MLP, como classificador dos 18 modelos. O treino usa o train da competição mais 1.000 textos por modelo do Open Pool, com peso 0,85, e nunca usa na validação um texto que esteja no pool.

A validação é a dobra 0 da GroupKFold por prompt, a mesma dos modelos lineares, para comparar e combinar os dois nos mesmos textos.

Antes do treino completo, rode com `SMOKE = True`: são 64 textos e 8 passos, para conferir o caminho inteiro em poucos minutos. Precisa de GPU T4, internet ligada (para baixar o Qwen) e o dataset do Open Pool anexado.
