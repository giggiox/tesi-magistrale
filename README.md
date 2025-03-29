Questa repository contiene il codice per la Tesi magistrale su valutazione LLM.


In particolare è possibile creare un modello in questo modo:
```python
model = Model("hf-model-name", load_on_init = True)
```

E poi scegliere uno dei due dataset che sono stati implementati:
```python
boolq = BoolQ(load_on_init=True, dataset_fraction = 1, split = "validation")
hellaswag = HellaSwag(load_on_init=True, dataset_fraction = 1, split = "validation")
```

Per valutare il modello sul dataset è necessario chiamare il metodo `evaluate_model` del dataset, ad esempio:

```python
boolq.evaluate_model(model, n_shot = 0)
```

è anche possibile passare un oggetto `LoggerManager` al metodo in modo che ciascuna iterazione venga salvata su un file.


### Espansione del codice
Il codice è stato fatto in modo che sia facile aggiungere modelli/dataset.

Per i modelli è sufficiente cambiare la stringa del modello nel costruttore di `Model`.

Per i dataset invece è necessario implementare una classe che eredita da `Dataset` dove è necessario implementare i seguenti metodi:

- `format_prompt(example)` dove `example` è una riga del dataset. Questo metodo deve ritornare una stringa che è il prompt che è necessario fornire al LLM.

- `is_correct(model_answer, example)`. Questo metodo prende in input la risposta del modello e la riga del dataset che corrisponde al data point usato per formulare il prompt. Ritorna se la risposta del modello è corretta oppure no.

- `get_true_answer(example)`. Questo metodo ritorna la stringa di come deve essere la risposta del LLM nel caso di few-shot prompting.


