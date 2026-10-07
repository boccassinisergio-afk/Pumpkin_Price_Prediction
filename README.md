# Pumpkin Price Prediction

[Italiano](#italiano) | [English](#english)

---

## Italiano

Regressione lineare e polinomiale a confronto, con e senza una feature categorica (City), sul dataset US-pumpkins.

### Scopo

Questo è un progetto di apprendimento, costruito per mostrare le basi di machine learning che ho acquisito finora, a livello iniziale. Non vuole essere un software completo, né un'analisi di dataset di grandi dimensioni. L'obiettivo è mostrare il metodo: confrontare i modelli in modo equo sugli stessi split e verificare quanto i risultati siano stabili al variare della suddivisione dei dati.

### La storia

Seguendo il corso ML for Beginners di Microsoft, ho disegnato il prezzo delle zucche da torta in funzione del giorno dell'anno. La retta di tendenza scendeva leggermente, ma i punti erano molto dispersi in verticale: nello stesso giorno i prezzi andavano da circa 14 a circa 22. Nessuna curva costruita solo sulla data può spiegare una dispersione così, quindi la domanda è diventata: che cos'altro la spiega?

La mia ipotesi era la città in cui le zucche vengono vendute.

### Dati

Il dataset è `US-pumpkins.csv`, fornito nel repository del corso (Microsoft ML-for-Beginners). Preparazione:

- solo confezioni vendute a bushel
- solo la varietà `PIE TYPE`
- prezzo = media tra prezzo minimo e massimo, riscalata per le confezioni `1 1/9` bushel e `1/2` bushel, in modo che tutti i prezzi si riferiscano alla stessa unità
- eliminazione delle righe con valori mancanti, che lascia 144 righe
- feature: `DayOfYear`, e `City` codificata one-hot con `drop_first=True` (9 città, 8 colonne)

### Metodo

Vengono confrontati quattro modelli:

| Modello | Feature |
|---|---|
| Regressione lineare | `DayOfYear` |
| Regressione polinomiale (grado 2) | `DayOfYear` |
| Regressione lineare | `DayOfYear` + `City` |
| Regressione polinomiale (grado 2) | `DayOfYear` + `City` |

- `test_size=0.33`, e in ogni esecuzione tutti e quattro i modelli usano le stesse righe di train e di test
- `PolynomialFeatures` viene addestrato solo sul train e poi applicato al test
- metrica: R² sul test set
- un singolo split si è rivelato poco affidabile (l'R² si è spostato di 0.03-0.09 passando da un seme a un altro), quindi l'intero esperimento è ripetuto per 40 valori di `random_state` (da 0 a 39), e riporto media e deviazione standard di ogni modello

### Risultati

| Modello | R² medio (test) | Dev. std |
|---|---|---|
| Lineare, `DayOfYear` | 0.040 | 0.069 |
| Polinomiale, `DayOfYear` | 0.046 | 0.068 |
| Lineare, `DayOfYear` + `City` | 0.285 | 0.111 |
| Polinomiale, `DayOfYear` + `City` | 0.400 | 0.117 |

### Cosa emerge

1. **`DayOfYear` da solo non basta.** L'R² medio è intorno a 0.04, con una deviazione standard più grande della media stessa. Passare dalla retta al polinomio di grado 2 aggiunge solo 0.006, molto meno della variabilità tra split: non c'è evidenza che la curvatura nel tempo aiuti.
2. **Aggiungere `City` aiuta in modo netto.** Il modello lineare passa da 0.040 a 0.285, un salto di circa 0.25, più del doppio della deviazione standard. Questo conferma l'ipotesi di partenza: gran parte della dispersione verticale dipendeva dalla città.
3. **Il polinomiale con `City` è il migliore**, con 0.400 contro 0.285 del lineare. Il divario (circa 0.115) è paragonabile alla deviazione standard di ciascun modello. Poiché i due modelli sono testati sugli stessi split, un confronto appaiato (la differenza split per split) direbbe quanto il vantaggio sia costante; non l'ho calcolato.
4. **Un'ipotesi, non verificata.** Con più colonne, `PolynomialFeatures` genera anche i prodotti tra `DayOfYear` e le singole città, cioè permette a ogni città di avere un proprio andamento nel tempo. Siccome il polinomio sulla sola `DayOfYear` non migliora, il guadagno potrebbe venire da queste interazioni più che dalla curvatura. Per verificarlo si potrebbe confrontare il modello con uno che contiene solo le interazioni, senza il termine al quadrato.

Anche il modello migliore spiega in media circa il 40% della variabilità: i prezzi dipendono da altri fattori che questi dati non contengono.

### Limiti

- Solo 144 righe: il test set ha circa 48 punti, quindi ogni singolo valore di R² è rumoroso
- I dati coprono solo la stagione del raccolto (circa da fine agosto a inizio dicembre), quindi non si può osservare una curva stagionale completa
- L'anno non è usato come feature, quindi andamenti di anni diversi sono mescolati
- Le feature polinomiali di grado 2 su 9 colonne producono 55 colonne per circa 96 righe di train, e molte sono ridondanti per le colonne one-hot
- Nessuna ottimizzazione degli iperparametri e nessuna cross-validation: ripetere split casuali è un sostituto semplice, e la cross-validation è il passo successivo naturale
- I risultati valgono per questo dataset e questo intervallo di date, e non vanno generalizzati

### Come eseguirlo

```bash
pip install pandas numpy scikit-learn
python pumpkins_regression.py
```

Metti `pumpkins.csv` nella stessa cartella dello script.

### Crediti

Dataset e struttura della lezione vengono dal corso [ML for Beginners](https://github.com/microsoft/ML-For-Beginners) di Microsoft. Il disegno dell'esperimento (aggiunta di `City`, confronto tra quattro modelli, ripetizione su 40 split) è una mia estensione.

---

## English

Linear vs polynomial regression, with and without a categorical feature (City), on the US-pumpkins dataset.

### Purpose

This is a learning project built to show the machine learning fundamentals I have acquired so far, at a beginner level. It is not meant to be a complete piece of software, nor an analysis of large datasets. The goal is to show the method: comparing models fairly on identical splits, and checking how stable the results are across many different splits.

### The story

While following Microsoft's ML for Beginners course, I plotted the price of pie pumpkins against the day of the year. The trend line pointed slightly downward, but the points were scattered vertically: on the same day, prices ranged from roughly 14 to 22. No curve built on the date alone can explain that, so the question became: what else explains the spread?

My hypothesis was the city where the pumpkins are sold.

### Data

The dataset is `US-pumpkins.csv`, provided in the course repository (Microsoft ML-for-Beginners). Preparation steps:

- keep only packages sold by the bushel
- keep only the `PIE TYPE` variety
- price = average of low and high price, rescaled for `1 1/9` bushel and `1/2` bushel packages so that all prices refer to the same unit
- drop rows with missing values, leaving 144 rows
- features: `DayOfYear`, and `City` one-hot encoded with `drop_first=True` (9 cities, 8 columns)

### Method

Four models are compared:

| Model | Features |
|---|---|
| Linear regression | `DayOfYear` |
| Polynomial regression (degree 2) | `DayOfYear` |
| Linear regression | `DayOfYear` + `City` |
| Polynomial regression (degree 2) | `DayOfYear` + `City` |

- `test_size=0.33`, and in each run all four models use the same train and test rows
- `PolynomialFeatures` is fitted on the training set only and then applied to the test set
- metric: R² on the test set
- a single split proved unreliable (R² moved by 0.03 to 0.09 between two different seeds), so the whole experiment is repeated for 40 values of `random_state` (0 to 39), and I report the mean and the standard deviation of each model

### Results

| Model | Mean R² (test) | Std |
|---|---|---|
| Linear, `DayOfYear` | 0.040 | 0.069 |
| Polynomial, `DayOfYear` | 0.046 | 0.068 |
| Linear, `DayOfYear` + `City` | 0.285 | 0.111 |
| Polynomial, `DayOfYear` + `City` | 0.400 | 0.117 |

### Takeaways

1. **`DayOfYear` alone is not enough.** Mean R² is around 0.04, with a standard deviation larger than the mean itself. Going from a straight line to a degree 2 polynomial adds only 0.006, far less than the variability between splits: there is no evidence that curvature over time helps.
2. **Adding `City` helps clearly.** The linear model goes from 0.040 to 0.285, a jump of about 0.25, more than twice the standard deviation. This supports the starting hypothesis: a large part of the vertical scatter was due to the city.
3. **The polynomial model with `City` is the best**, at 0.400 against 0.285 for the linear one. The gap (about 0.115) is comparable to the standard deviation of each model. Since both models are tested on the same splits, a paired comparison (the difference split by split) would show how consistent the advantage is; I have not computed it.
4. **A hypothesis, not verified.** With more columns, `PolynomialFeatures` also generates products between `DayOfYear` and the individual cities, which lets each city have its own trend over time. Since the polynomial on `DayOfYear` alone does not improve, the gain may come from these interactions rather than from curvature. To check it, one could compare against a model that contains only the interactions, without the squared term.

Even the best model explains only about 40% of the variability on average: prices depend on other factors that this data does not contain.

### Limitations

- Only 144 rows: the test set has about 48 points, so any single R² value is noisy
- The data covers only the harvest season (roughly late August to early December), so a full seasonal curve cannot be observed
- The year is not used as a feature, so trends across different years are mixed together
- Polynomial features of degree 2 on 9 columns produce 55 columns for about 96 training rows, and many of them are redundant for one-hot columns
- No hyperparameter tuning and no cross-validation: repeating random splits is a simple stand-in, and cross-validation is a natural next step
- Results hold for this dataset and this date range, and should not be generalized

### Run it

```bash
pip install pandas numpy scikit-learn
python pumpkins_regression.py
```

Place `pumpkins.csv` in the same folder as the script.

### Credits

Dataset and lesson structure come from Microsoft's [ML for Beginners](https://github.com/microsoft/ML-For-Beginners) course. The experiment design (adding `City`, comparing four models, repeating over 40 splits) is my own extension.