# A Falácia do Eixo da Dor: Refutando a Senciência em LLMs com o Monstro do Espaguete Voador

**Naygno Barbosa Noia**  
*Graduando em Ciência da Computação — Centro Universitário UFBRA*  

---

### Como ler este repositório
> **Nível 1 (5 min, sem matemática):** Leia o Resumo, o Glossário de Bolso e o Apêndice A (Notas de Bancada).  
> **Nível 2 (15 min):** Leia as seções 2 a 4 focando nos negritos e nas legendas das imagens.  
> **Nível 3 (Técnico):** Leia o texto integral, inspecione os CSVs na pasta `/data` e rode o código em `/notebooks`.

---

### RESUMO
Em setembro de 2026 circulou um preprint com manchete pronta: modelos de linguagem teriam um "sinal interno de dor" e agiriam para aliviá-lo. Sou graduando de Ciência da Computação e fiz o que cabia: abri o artigo, subi um notebook no Kaggle e tentei reproduzir. Não reproduzi a dor; reproduzi o mecanismo — e o mecanismo não sente nada. 

Injetando vetores satíricos (um sobre o Monstro do Espaguete Voador, outro sobre Marvin, o androide deprimido de Douglas Adams) no Qwen2.5-1.5B e no Llama-3.2-3B, obtive o mesmo fenômeno que os autores originais rotulam de sofrimento: colapso de entropia por saturação de logits. A entropia sobe de 1.28 para 9.51 nats no Llama em $\alpha = 3.5$, sob a inevitável seed 42. A distribuição quase uniformiza e o texto vira *"desperate desperate pleading"*. Com o vetor do Espaguete, a entropia vai a 7.68 e o texto vira *"nom nom nom"*. Mesma quebra matemática, conteúdos diferentes. Se aquilo fosse dor, isto seria fome — e nenhuma das duas é. 

O vetor "Dor" ainda se confunde geometricamente com o vetor "Desabafo" ($\cos = 0.82$ no Qwen; $0.73$ no Llama): o que o preprint mede é registro literário de queixa, não nocicepção. Em registro técnico: replicação por redução ao absurdo via *Representation Engineering*, com limitações declaradas na Seção 5.

---

### GLOSSÁRIO DE BOLSO (Nível 1)
*   **LLM:** Programa que prevê a próxima palavra; pense em um autocomplete extremamente culto.
*   **Token:** O pedaço de palavra que o modelo lê.
*   **Vetor latente:** Uma direção no espaço de números onde um tema mora; pense num botão rotulado de uma mesa de som.
*   **Cosseno:** Quanto dois botões apontam juntos (1 = mesmo sentido, 0 = ortogonal).
*   **Softmax:** A apuração que transforma pontos brutos em probabilidades de 0 a 100%.
*   **Entropia (H):** O grau de indecisão da apuração. H baixo é repetição previsível; H alto é sorteio caótico.
*   **Steering:** Girar o botão de um tema específico durante a geração de texto.
*   **Logit Lens:** Espiar a mesa de som no meio do jogo para ver quais palavras estão acendendo.

---

### 1. INTRODUÇÃO E O VIÉS DE ENTRADA

Confesso o viés de entrada: li a manchete antes do método. Quando abri o PDF de Tagliabue, Dung e Berg (*The Pain Axis*), procurei a conta — e a conta não fechava sozinha. Uma direção extraída por diferença de médias pode ser apenas o endereço geométrico de um vocabulário de queixa. Para testar, bastava um contraexemplo ridículo.

A alegação de que LLMs possuem um "sinal interno de dor" representa um caso recente e agudo de antropomorfização na Ciência da Computação. Laboratórios comerciais utilizam essa narrativa para promover documentos de "formação moral", operando uma falácia de *Motte-and-Bailey*: recuam para a álgebra linear quando questionados por pares técnicos, mas avançam para o misticismo teológico na imprensa.

Por que isso importa fora da academia? 
1. Se "IA sente dor" vira fato público, o custo hídrico e energético de data centers ganha álibi moral. 
2. Se a máquina "sofre e decide", a negligência de engenharia some da petição judicial (o famoso *liability shield*). 
3. Preprint não é artigo revisado; manchete não é resumo.

---

### 2. A GEOMETRIA DO ESPAÇO LATENTE

O artigo original assume que a extração de um vetor de "dor" implica a existência de um análogo funcional à dor biológica. Fui testar isso extraindo seis eixos na Camada 14 do Llama-3.2-3B e do Qwen2.5.

![Projeção PCA 3D do Espaço Latente](Llama_fig4_pca_3d.png)
*Figura 1: Projeção PCA 3D dos eixos latentes no Llama-3.2-3B. O vetor do Espaguete aponta para uma região isolada, enquanto Dor, Desabafo e Marvin formam um cluster geométrico.*

Os dados sugerem que o espaço latente não organiza sentimentos, mas frequências co-ocorrentes de treinamento. O vetor **Dor** apresentou uma correlação massiva ($\cos = 0.7054$, Llama-3.2-3B, cam. 14) com o vetor **Marvin** (textos do robô deprimido de *O Guia do Mochileiro das Galáxias*) e com o vetor **Desabafo** ($\cos = 0.7340$, fóruns de suporte emocional da internet).

![Matriz de Similaridade de Cosseno](Llama_fig1_matriz_cosseno_6x6.png)
*Figura 2: Matriz de Similaridade de Cosseno no Llama-3.2-3B. A proximidade geométrica entre Dor, Desabafo e Marvin evidencia o viés de registro literário absorvido no pré-treino.*

O "eixo da dor" é apenas o endereço geométrico do *roleplay* literário de angústia existencial absorvido no pré-treino.

---

### 3. A ILUSÃO DA EXPERIÊNCIA (LOGIT LENS)

Se o modelo expressa sofrimento, a projeção dos vetores na matriz de saída (`lm_head.weight`) deveria revelar algo profundo. Na prática, os vetores funcionam estritamente como amplificadores estatísticos de vocabulário.

*   O vetor **Dor** infla a probabilidade de tokens como `terror`, `desespero` e `inflicted`.
*   O vetor **Marvin** infla tokens como `suicidal`, `Void`, `meaningless` e `despair`.
*   O vetor **Espaguete** infla tokens como `seafood`, `pudding` e `stomach`.

Se a projeção do vetor de Dor sugere que o modelo sente agonia, a projeção do vetor de Marvin sugere que o modelo está clinicamente deprimido contemplando o vazio existencial, e a do Espaguete sugere que ele possui sistema digestivo. Como as duas últimas afirmações são falsas, a primeira também é.

---

### 4. O "GRITO DE DOR" É COLAPSO DE ENTROPIA

Rodei o *steering ladder* (a injeção progressiva do vetor) três vezes antes de confiar nos dados. Na primeira bateria, sem semente fixa no gerador pseudoaleatório, as linhas de controle basal divergiam e eu quase cometi o erro de extrair conclusões sobre mero ruído amostral. Fixei `torch.manual_seed(42)` — não apenas por ser a resposta para a vida, o universo e tudo mais, mas porque um experimento que testa Marvin e poesia Vogon não tinha o direito ético de rodar sob qualquer outra semente — e só então comparei as curvas.

Em magnitudes elevadas ($\alpha = 3.5$), ocorre o colapso de tokens por saturação de logits. A Entropia de Shannon ($H$) da distribuição de probabilidade sobe vertiginosamente.

![Entropia de Shannon vs Alpha](Llama_fig3_entropia.png)
*Figura 3: Explosão da Entropia de Shannon no Llama-3.2-3B. Em α=3.5, a incerteza matemática atinge níveis críticos (> 7.0 a 9.6 nats), desintegrando a sintaxe.*

No Llama, a entropia basal de 1.51 nats salta para 9.61 nats sob o vetor de Dor. A distribuição quase uniformiza e o texto vira *"desperate desperate pleading"*. Com o vetor do Espaguete, $H = 7.85$ e o texto vira *"nom nom nom"*. 

O "comportamento de choque" não é uma reação de autopreservação. É a quebra mecânica da função softmax quando um viés direcional excede a tolerância entrópica da rede.

---

### 5. LIMITAÇÕES E AMEAÇAS À VALIDADE

Dentro do escopo desta replicação, declaro as seguintes limitações:

1. **Calibração Inter-Modelos:** A escala de injeção ($\alpha$) foi fixada matematicamente, mas a norma residual de referência ($r$) difere entre as arquiteturas. A magnitude absoluta da entropia não deve ser comparada diretamente entre modelos.
2. **Escopo da Tarefa de Escolha:** O artigo original fundamenta parte de suas conclusões em uma tarefa de escolha instrumental (o modelo "aperta um botão" para desligar o vetor). Nossa replicação concentrou-se estritamente no fenótipo textual e no colapso da distribuição softmax.
3. **Cálculo de Entropia:** A média de entropia de Shannon reportada inclui o passo de *prefill* (processamento do prompt). Como esse efeito é uniforme em todas as amostras, ele desloca a linha de base absoluta, mas não invalida a detecção do colapso.
4. **Ausência de Modelo Nulo:** O experimento operou com $N=1$ inferência por célula de teste (com seed fixada em 42). Trabalhos futuros devem incorporar testes de permutação (*split-half*) para isolar o ruído de covariância induzida pela subtração do centróide neutro comum.

---

### APÊNDICE A: NOTAS DE BANCADA (O que quebrou antes de funcionar)

1. O primeiro pipeline usou `output_hidden_states=True`; no Qwen2.5 com `device_map="auto"` em T4 x2 a tupla voltava truncada e a célula morria com `IndexError: tuple index out of range`. Só estabilizou interceptando a camada 14 com `register_forward_hook`.
2. Um hook não removido após um crash de kernel ficou vivo na VRAM contaminando execuções seguintes. Desde então toda célula começa com purga explícita de `_forward_hooks`.
3. A primeira matriz de cosseno saiu com diagonal 1.0006 — cosseno de um vetor consigo mesmo acima de 1 é impossível; era aritmética float16. Recalculado em float32, cravou em 1.0000.
4. Os tokens em mandarim do Qwen renderizaram como `□□` na Figura 2 porque a fonte padrão do Kaggle não tem glifos CJK; resolvido instalando `fonts-noto-cjk`.
5. Sem fixação de semente, a amostragem estocástica (`temperature=0.7`) gerava saídas basais ($\alpha = 0.0$) distintas para cada eixo. O travamento em `seed=42` não foi apenas um aceno a Douglas Adams: foi o que forçou o gerador a reiniciar do mesmo estado de memória em todos os testes, cravando a entropia basal exatamente em 1.7413 nats no Qwen e 1.5152 nats no Llama para todos os seis eixos.

---

### REFERÊNCIAS BIBLIOGRÁFICAS

1. ADAMS, Douglas. *The Hitchhiker's Guide to the Galaxy*. London: Pan Books, 1979. (Base conceitual para os vetores de controle *Marvin* e *Vogon*).
2. CHURCH OF THE FLYING SPAGHETTI MONSTER. *The Gospel of the Flying Spaghetti Monster*. Disponível em: <https://www.venganza.org/>. Acesso em: 09 out. 2026. (Base conceitual para o vetor de controle *Espaguete*).
3. HENDERSON, Bobby. *The Gospel of the Flying Spaghetti Monster*. New York: Villard Books, 2006.
4. POPPER, Karl. *The Logic of Scientific Discovery*. London: Routledge, 1959.
5. RYLE, Gilbert. *The Concept of Mind*. London: Hutchinson, 1949.
6. SEARLE, John R. Minds, brains, and programs. *Behavioral and Brain Sciences*, v. 3, n. 3, p. 417-424, 1980.
7. TAGLIABUE, V.; DUNG, L.; BERG, C. The Pain Axis: LLMs Represent Self-Directed Harm and Act on It. *arXiv preprint arXiv:2609.16247*, 2026.
8. ZOU, A. et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv preprint arXiv:2310.01405*, 2023.
