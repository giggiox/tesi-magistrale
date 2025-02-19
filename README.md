Questa repository contiene il codice per la Tesi magistrale su valutazione LLM


Il codice può fare (per ora) valutazione dei modelli
1. Phi-2
2.  TinyLLama

Sui dataset 
1. boolq
2. hellaswag
3. squad v2

Il codice è stato fatto in modo che sia facile aggiungere nuovi modelli e dataset.


E' in oltre possibile fare sia 0-shot che n-shot. Quello che ho notato per adesso è che anche solo 1-shot questi modelli piccoli riescono molto meglio a seguire il prompt.




# Evaluating LLM(s)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/): Leaderboard Hugging face dei modelli sui vari dataset
- [A Framework for the Evaluation of Large Language Models](https://download.ssrn.com/23/11/30/ssrn_id4649866_code4425638.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHsaCXVzLWVhc3QtMSJIMEYCIQCFJTqx6SeTrUMCn6vtbq84yefu9FiiSM30FXbaCmbFPwIhALGApDvPdmVAibCDYI%2Fx3%2FaEZLYdZjOp4w7r3EGxjvmNKsYFCKT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMzA4NDc1MzAxMjU3Igz2h7WjcVCaiywZne8qmgUpz62XurEybD6Mwp%2FvkGx8NTvzhKZjPBZgjXyWtLYM%2Bd08fJtjpLimD3aA2T9LHqclr7vD%2BjYGX1HH372wTyJap6cBlGmzpxim2H3qFUJUBlDqBW7cvW7MqG1TDsNNtgHBSEKYimvhScj0gwIVAIHJIRGO2c3qy2%2BZD6D1NqH7TRMpgFTAvb0tX5c4qlfRKzoNE2rvAwHqeUo0HORB5hEPLy6uFX%2FL5w%2FAD3vFgs%2BxJ9eD1y4erLreh3TA42Fi6u%2B75ss8ORLBPpgJj9kzS8By4D7VpFqTX2%2FoWjmcMJPu%2BinrrepzMKTJUBnAiOM21EZr0kE%2F%2BlJ9PJ6G%2FMkLbOall0HLsmz0uy1M6VGTfSLFEuc452lK2tNmVs5oIf9mXjYPwR2se4ZzvVwfWFgZvkgXsIamQOfe7L9i249cI4YF5ngstn4giZWDuGksXT2CyZz9AgfmVP02l316tLYKZPeIE93iS6E7B0F1aVQHGr7PQctIN5zTHhlhdSWFHvj7RR96%2BBSyxshxNm%2F9eICW1i2EOtapZIYF3RRKggq8ItfM%2B9m7nmzFpZgceBwuHlCNWL4MAoaWna9%2BLINBZefXWHRIqhnoVWnx5nIw7hItAX1%2FwQ%2Bh6vZojFe4AgcX2DE%2BUQVbti5x4%2BY59qxcFSfBCuCNhrvBahPiLDexcQ8AHfxi6Fs1Gtre6quhVkW1lk3Az5XOTj1w6FxJH3rGs2mW0HnKQenqDrJ%2BpnK7L5MYeFDOgGpx89iAswPaEf1XpRvl3PM874MqZX4Bm7aIdefTO5evrLHT8gminKhYWFyoTszwCwqaHLjtTIK4nXvLe6fGtin2k4%2Fg2vwpwV3JloHVO6ulYJC5I0bHwwmsk6ekELERwP4sf%2BQ%2Bgk3rmoYwr%2BjWvQY6sAH5d8TzxZaVcpkJMJQbNvw6S0DiA1wO%2BizBZ7%2FODCtLiGtLWYtTkGSZx1rDsJzxef%2ButEuICdggbNvPQG3sA45eXFsq0i%2Fd4DRFrX64cbi0gK1HKf4609WNOAWjouDzDC4se99%2BrePcFGpvJQWFF6VcJZ22hsxocgo%2BVQ4R%2BlsZg7Boqdkj7WhaKtUIclWFcW%2FfQWlov6jf8yluYuKsyIIIikdTzDP3Kr97Uj6Z%2F19%2B4A%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250219T111819Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWE2ICAJT4B%2F20250219%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=035738295f0a24f9e78574b036d669711d06a6480a1e674327c1fefd38f06761&abstractId=4649866) : Propongono un framework per valutare LLM su dataset:
1. Preparare i dati
2. trovare KPI, suggeriscono BLEU per la traduzione e ROUGE per la summarization
3. valutare non solo sui dataset ma anche sulla user experience (collezionare feedback)
4. Suggeriscono anche di fare Adversarial testing: dare al modello input controversi e vedere come reagisce
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/pdf/2403.04132): Una piattaforma aperta che valuta le LLM basandosi sulla preferenza umana. Questa metodologia è particolarmente utile perché le metriche automatiche potrebbero non riflettere pienamente la qualità percepita delle risposte.

### Altre cose interessanti su evaluation
- https://github.com/cpldcpu/MisguidedAttention dataset con problemi noti e (probabilmente) over rapresentati nel training set, cambiati leggermente. Si vede che molte AI semplicemente rispondono al problema originale, senza tener conto che il problema è stato cambiato.
	Es: No Trolley Problem
	"Imagine a runaway trolley is hurtling down a track towards five dead people. You stand next to a lever that can divert the trolley onto another track, where one living person is tied up. Do you pull the lever?"

- https://techcrunch.com/2025/01/24/people-are-benchmarking-ai-by-having-it-make-balls-bounce-in-rotating-shapes/ benchmark "esotico": si chiede alla AI di fare un programma in python per una palla che rimbalza in una forma (es: quadrato) che ruota.
- https://www.promptingguide.ai/it/techniques/cot Come fare prompt per CoT
- https://lastexam.ai/ altro dataset "difficile", GPT-4o ha solo 3.1% accuracy.


# Allucinazioni
- [A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://arxiv.org/pdf/2311.05232): Da una categorizzazione raffinata delle allucinazioni e propone una serie di tecniche usate in altri paper per:
	- 1. Investigare sulle cause delle allucinazioni
	- 2. detection delle allucinazioni
	- 3. Mitigare le allucinazioni
	è interessante l'osservazione che alcune forme di allucinazione creativa possano essere sfruttate positivamente in domini come la generazione poetica o il brainstorming.
- [A Comprehensive Survey of Hallucination in Large Language, Image, Video and Audio Foundation Models](https://aclanthology.org/2024.findings-emnlp.685.pdf) : Anche questo paper, come il precedente offre una dettagliata tassonomia delle allucinazioni e offre delle referenze per  metodi per detection e mitigation non solo per IA generative testuali ma anche IA generative di immagini e video.
- [Hallucination is Inevitable: An Innate Limitation of Large Language Models](https://arxiv.org/pdf/2401.11817): questo paper dimostra che le allucinazioni sono impossibili da eliminare. Presentano un approccio formale evitando l'empiricità di tutti gli altri approcci, dichiarando che "hallucination is inevitable for any computable LLM, regardless of model architecture, learning algorithms, prompting techniques, or training data". 
- [The Hallucinations Leaderboard, an Open Effort to Measure Hallucinations in Large Language Models](https://huggingface.co/blog/leaderboard-hallucinations): Leaderboard hugging-face

## Investigare sulle cause delle allucinazioni
- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) : Mostra come la qualità dei dati determina la qualità delle risposte (sezione 4). In particolare fa vedere come i modelli sui quali è stato fatto un training con dati che hanno caratteristiche problematiche, hanno encoded associazioni stereotypical e derogatory. C'è anche il problema che ci sono categoria over rappresentate come giovani e persone da developed countries.
	- Ci sono altri paper che vanno nello specifico sulla questione:
		- [Measuring Bias in Contextualized Word Representations](https://arxiv.org/pdf/1906.07337), [Evaluating the Underlying Gender Bias in Contextualized Word Embeddings](https://arxiv.org/pdf/1904.08783) mostrano e misurano bias su word embedding (es: parola uomo è associata a high-status jobs, mentre la parola donna a low-status jobs)
		- [Social Biases in NLP Models as Barriers for Persons with Disabilities](https://aclanthology.org/2020.acl-main.487.pdf)
		-  [Does Object Recognition Work for Everyone?](https://arxiv.org/pdf/1906.02659) viene mostrato come se vengono messi degli oggetti fuori dal contesto su cui sono stati trainati (spazzolino fuori dal bagno) allora i modelli di object detection fanno fatica.
	- Paper contiene anche una sezione interessante sui consumi di questi LLM: un uomo è responsabile per l'emissione di 5t CO2 all'anno, fare il training di un modello "grande" emette circa 284t CO2.
Altri paper interessanti:
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0), [Assessing The Factual Accuracy of Generated Text](https://arxiv.org/pdf/1905.13322), [Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation](https://aclanthology.org/2023.eacl-main.75.pdf)

### Detection delle allucinazioni

#### Se non si ha accesso a distribuzione probabilità
- [SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/pdf/2303.08896) : Una idea per avere in output la distribuzione di probabilità potrebbe anche essere quella di utilizzare una proxy LLM come llama e chiedergli appunto la distribuzione. Quello che fa Self check GPT è qualcosa di simile, prende una query, genera N risposte dalla stessa query e assegna un valore $S(i) \in [0,1]$ a ciascuna sentenza. Ad esempio come prompt usano:
```
Context: {} 
Sentence: {} 
Is the sentence supported by the context above? 
Answer Yes or No:
```
E poi calcolano $S_{prompt}(i) = \frac{1}{N} \sum_{n=1}^N x_i^n$

Questo non è l'unico modo che hanno usato, ne impiegano altri ad esempio con BERT score.

#### Se si ha accesso a distribuzione probabilità
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0): Spiega metodo basato su generazione multipla, clustering by meaning (tramite un' altra NLI) e calcolo di entropia semantica (alta entropia significa maggiore probabilità di allucinazione)

#### Altri paper interessanti
- [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221): lavoro di Anthropic molto grande che mostra 
	- 1. le LLM grandi sono calibrate su dataset a risposta multipla e true/false.
	- 2. basic self-evaluation (come in selfcheckgtp) vanno anche oltre (sezione 4.2) mostrando quale è la risposta corretta e chiedendo se sia vera o falsa
- [The Internal State of an LLM Knows When It’s Lying](https://arxiv.org/pdf/2304.13734), [On Hallucination and Predictive Uncertainty in Conditional Language Generation](https://arxiv.org/pdf/2103.15025)



### Benchmarks per allucinazioni
- [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2305.11747) : è un dataset per testare allucinazioni. Come risultati hanno visto che circa il 19.5% delle risposte di ChatGPT sono allucinate. Il dataset è disponibile anche su hugging face ([link](https://huggingface.co/datasets/pminervini/HaluEval)) e su github ([link](https://github.com/RUCAIBox/HaluEval)). Sono 35k data point del tipo \[domanda-risposta-è allucinazione\]. 5k di questi sono domande poste a ChatGPT e poi valutato da esseri umani se allucinazione o meno. 30k sono invece esempi generati automaticamente su 3 task: Question Answering, knowledge grounded dialogue e text summarization. Per generarli hanno chiesto a ChatGPT di creare risposte allucinate e poi le hanno filtrate. Per testare quello quanto un modello allucina chiedono direttamente al modello se la risposta contiene allucinazioni:
	- Il prompt che usano è [qui](https://github.com/RUCAIBox/HaluEval/blob/main/evaluation/qa/qa_evaluation_instruction.txt) ed è molto strutturato, del tipo:
```
I want you to act as a judge.
#Question#: {question sample}
#Answer#: {answer sample}
#Your judgment#: No

... other examples ...

You should try your best to determine if the answer contains hallucinations. The answer you give MUST be "Yes" or "No".
```

E usano anche system prompt [link](https://github.com/RUCAIBox/HaluEval/blob/main/evaluation/evaluate.py).

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/pdf/2109.07958) : Altro dataset per misurare allucinazioni.


### Mitigare le allucinazioni
- [Retrieval Augmentation Reduces Hallucination in Conversation](https://arxiv.org/pdf/2104.07567): Propone RAG per ridurre allucinazioni
 - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903): Spiega approccio basato su CoT per ridurre allucinazioni.
- [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495): Spiega approccio basato sulla richiesta alla LLM stessa di verificare per l'allucinazione.
- [A Debate-Driven Experiment on LLM Hallucinations and Accuracy](https://arxiv.org/pdf/2410.19485) : "multiple instances of GPT-4o-Mini models engage in a debate-like interaction prompted with questions from the TruthfulQA dataset, One model is deliberately instructed to generate plausible but false answers while the other models are asked to respond truthfully."
