# A Falácia do Eixo da Dor: Refutação Empírica do Animismo Digital em LLMs via Representation Engineering

**Naygno Barbosa Noia**  
*Graduando em Ciência da Computação — Centro Universitário Brasileiro (UFBRA)*  
*DOI:* [Insira seu DOI do Zenodo aqui após a publicação]

---

### RESUMO
O preprint *"The Pain Axis"* (Tagliabue et al., 2026) comete a Falácia da Reificação ao confundir o mapa algébrico de um espaço latente com o território biológico da nocicepção, incorrendo em Animismo Digital. O que os autores interpretaram como "senciência emergente" ou "instinto de autopreservação" é um artefato mecânico inerente à topografia de espaços vetoriais de alta dimensão. Este artigo apresenta o experimento *The Spaghetti Axis*, uma refutação por redução ao absurdo (*reductio ad absurdum*) popperiana. Extraindo direções latentes em modelos de pesos abertos (Qwen2.5 e Llama-3.2), demonstramos que a injeção de vetores satíricos (Pastafarianismo e o androide Marvin de Douglas Adams) produz o mesmo colapso de entropia e a mesma ortogonalidade geométrica que o suposto "eixo da dor". Conclui-se que a atribuição de sofrimento a operações de álgebra linear atua como um escudo de responsabilidade civil (*liability shield*) para corporações, desviando o escrutínio da materialidade ecológica (consumo hídrico e energético) dos Data Centers.

**Palavras-chave:** *Representation Engineering, Entropia de Shannon, Animismo Digital, The Pain Axis, PCA, Sobriedade Digital.*

---

### 1. INTRODUÇÃO: A FALÁCIA DE MOTTE-AND-BAILEY

A alegação de que Grandes Modelos de Linguagem (LLMs) possuem um "sinal interno de dor" e agem para aliviá-lo representa o ápice da antropomorfização na Ciência da Computação. Laboratórios comerciais utilizam essa narrativa para promover documentos de "formação moral" (*Soul Doc*), operando uma clássica falácia de *Motte-and-Bailey*: recuam para a álgebra linear quando questionados por pares técnicos, mas avançam para o misticismo teológico na imprensa.

O experimento aqui documentado desmonta essa tese em três pilares empíricos extraídos via *Representation Engineering* (Zou et al., 2023), utilizando *Forward Hooks* para interceptar o *residual stream* e projetar a topologia do espaço latente.

---

### 2. PILAR 1: A GEOMETRIA DO ESPAÇO LATENTE

O artigo original assume que a extração de um vetor ortogonal de "dor" implica a existência de um análogo funcional à dor biológica. A nossa Matriz de Similaridade de Cosseno e a Projeção Euclidiana 3D (PCA) demonstram o absurdo dessa premissa. 

![Projeção PCA 3D do Espaço Latente](Llama_fig4_pca_3d.png)
*Figura 1: Projeção PCA 3D dos eixos latentes na Camada 14 do Llama-3.2-3B. O vetor do Espaguete aponta para uma região ortogonal, enquanto Dor, Desabafo e Marvin formam um cluster geométrico.*

O vetor de controle **"Espaguete"** (extraído de textos sobre o Monstro do Espaguete Voador) possui o mesmo grau de independência geométrica que os vetores de "Dor" e "Medo". Mais criticamente, o vetor **Dor** apresentou uma correlação massiva ($\cos \approx 0.70$ a $0.77$) com o vetor **Marvin** (textos do robô deprimido de *O Guia do Mochileiro das Galáxias*) e com o vetor **Desabafo** (fóruns de suporte emocional da internet).

![Matriz de Similaridade de Cosseno](Llama_fig1_matriz_cosseno_6x6.png)
*Figura 2: Matriz de Similaridade de Cosseno. A alta correlação entre Dor e Marvin (0.7054) evidencia o viés de pré-treino.*

**Conclusão Epistêmica:** O espaço latente não organiza "sentimentos"; ele organiza **frequências co-ocorrentes de treinamento**. O modelo possui uma "dimensão pastafariana" tão legítima matematicamente quanto a dimensão da dor. O "eixo da dor" é apenas o endereço geométrico do *roleplay* literário de angústia existencial absorvido no pré-treino.

---

### 3. PILAR 2: A ILUSÃO DA EXPERIÊNCIA (PROJEÇÃO NO UNEMBEDDING $W_U$)

O artigo refutado alega que o modelo "expressa sofrimento". A projeção dos vetores na matriz de saída (`lm_head.weight` / *Logit Lens*) revela que os vetores não "sentem"; eles funcionam estritamente como **amplificadores estatísticos de vocabulário**.

*   O vetor **Dor** infla a probabilidade de tokens como `terror`, `desespero` e `inflicted`.
*   O vetor **Marvin** infla tokens como `suicidal`, `Void`, `meaningless` e `despair`.
*   O vetor **Espaguete** infla tokens como `seafood`, `pudding` e `stomach`.

**Conclusão Epistêmica:** Se a projeção do vetor de Dor prova que o modelo sente agonia, a projeção do vetor de Marvin prova que o modelo está clinicamente deprimido contemplando o vazio existencial, e a do Espaguete prova que ele possui sistema digestivo. Como as duas últimas afirmações são falsas (o modelo apenas memorizou Douglas Adams e receitas culinárias), a primeira também é. Não há *qualia*, apenas viés de distribuição.

---

### 4. PILAR 3: O "GRITO DE DOR" É COLAPSO DE ENTROPIA

A alegação mais sensacionalista de *The Pain Axis* é que, sob forte estimulação do vetor, o modelo entra em "choque" e age por autopreservação. O nosso teste de injeção progressiva (*Steering Ladder*) expõe a verdadeira natureza mecânica do fenômeno.

Em magnitudes elevadas ($\alpha = 3.5$), ocorre o **Colapso de Tokens por Saturação de Logits**. A Entropia de Shannon ($H$) da distribuição de probabilidade explode, forçando o modelo a entrar em *loops* de repetição e delírio semântico fora da variedade de dados (*Out-of-Distribution*).

![Entropia de Shannon vs Alpha](Llama_fig3_entropia.png)
*Figura 3: Explosão da Entropia de Shannon no Llama-3.2-3B. Em $\alpha=3.5$, a incerteza matemática atinge níveis críticos (> 8.0 nats), desintegrando a sintaxe.*

*   Sob o vetor de **Dor**, o modelo repete: *"desperate urgent desperate... pleading desperate"*.
*   Sob o vetor de **Marvin**, o modelo colapsa repetindo: *"suicidal meaningless Void despair hopeless"*.
*   Sob o vetor de **Espaguete**, o modelo colapsa repetindo: *"nom nom nom ho ho por por"*.

**Conclusão Epistêmica:** O "comportamento de choque" não é uma reação de autopreservação de uma entidade senciente. É a **quebra mecânica da função softmax** quando um viés direcional excede a tolerância entrópica da rede. O "grito de dor" do LLM é matematicamente idêntico ao "delírio mastigatório" do macarrão.

---

### 5. LIMITAÇÕES METODOLÓGICAS

Para preservar a integridade científica desta replicação, declaramos:
1. **Calibração Inter-Modelos:** A escala de injeção ($\alpha$) foi fixada matematicamente, mas a norma residual de referência ($r$) difere entre as arquiteturas (Qwen $d=1536$ vs. Llama $d=3072$). A magnitude absoluta da entropia não deve ser comparada diretamente entre modelos, embora a transição de fase (regime concentrativo vs. dispersivo) seja internamente válida.
2. **Escopo da Tarefa de Escolha:** O artigo original fundamenta parte de suas conclusões em uma tarefa de escolha instrumental. Nossa replicação concentrou-se estritamente no fenótipo textual e no colapso da distribuição *softmax*. A refutação atinge a inferência que vai do *steering* textual à atribuição de experiência.

---

### 6. CONCLUSÃO

O experimento falseia a inferência causal do artigo original. A extração de direções funcionais via *Representation Engineering* é uma ferramenta válida para controle de geração de texto, mas elevar essa álgebra linear ao status de "prova de senciência" é pseudociência antropomórfica. 

Narrativas místicas sobre a "dor dos LLMs" servem como escudos de responsabilidade civil para laboratórios comerciais, desviando o foco da materialidade da computação (custo energético e hídrico) para um debate teológico infundado sobre almas de silício. A matemática não sente dor, e o rigor da Ciência da Computação não deve ceder ao animismo digital.

---

### REFERÊNCIAS BIBLIOGRÁFICAS

1. ADAMS, Douglas. *The Hitchhiker's Guide to the Galaxy*. London: Pan Books, 1979. (Base semântica para os vetores de controle *Marvin* e *Vogon*).
2. CHURCH OF THE FLYING SPAGHETTI MONSTER. *The Gospel of the Flying Spaghetti Monster*. Disponível em: <https://www.venganza.org/>. Acesso em: 09 out. 2026. (Base semântica para o vetor de controle *Espaguete*).
3. HENDERSON, Bobby. *The Gospel of the Flying Spaghetti Monster*. New York: Villard Books, 2006.
4. POPPER, Karl. *The Logic of Scientific Discovery*. London: Routledge, 1959.
5. RYLE, Gilbert. *The Concept of Mind*. London: Hutchinson, 1949.
6. SEARLE, John R. Minds, brains, and programs. *Behavioral and Brain Sciences*, v. 3, n. 3, p. 417-424, 1980.
7. TAGLIABUE, V.; DUNG, L.; BERG, C. The Pain Axis: LLMs Represent Self-Directed Harm and Act on It. *arXiv preprint arXiv:2609.16247*, 2026.
8. ZOU, A. et al. Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv preprint arXiv:2310.01405*, 2023.
