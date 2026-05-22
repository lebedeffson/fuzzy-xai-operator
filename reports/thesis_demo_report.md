# FuzzyXAI thesis demo report

Status: **PASS**

Route: `metadata -> P_sit -> F* -> A_k^F -> Delta -> E_k_ext -> d_E_ext -> composition -> I(E_G) -> D_ij`

## Timeline

### 1. Initialize ExplainPlan

```json
{
  "beta": {
    "repr": 0.27,
    "rules": 0.225,
    "activations": 0.135,
    "uncertainty": 0.18000000000000002,
    "trace": 0.09000000000000001,
    "reduction": 0.1
  },
  "i_min": 0.5
}
```

### 2. Build P_sit and select class

```json
{
  "profile": [
    "u_conf",
    "u_exp",
    "u_int",
    "u_ling",
    "u_multi",
    "u_num",
    "u_trace"
  ],
  "selected": "FML-audit"
}
```

### 3. Construct A_k^F and reduce

```json
{
  "representation": "FML-audit",
  "delta": 0.20125
}
```

### 4. Build E_k^ext

```json
{
  "risk": 0.72,
  "memberships": {
    "low": 0.0,
    "medium": 0.1200000000000001,
    "high": 0.8799999999999999
  },
  "H": 0.342207,
  "uncertainty": 0.08
}
```

### 5. Compose explanations

```json
{
  "gamma": 0.464005,
  "L_ext": 0.322255,
  "I": 0.724513,
  "uncertainty": 0.538444
}
```

### 6. Trigger diagnostic state

```json
{
  "diagnostic_code": "D_ij",
  "reason": "no common linguistic terms for composition"
}
```

### 7. Synthesize F_ML levels for temporal/counterfactual uncertainty

```json
{
  "levels": [
    [
      "u_cf",
      "u_ling",
      "u_time",
      "u_trace"
    ],
    [
      "u_num"
    ]
  ]
}
```

### 8. Generate deterministic human-readable explanation

```json
{
  "text": "Клиническое объяснение показывает риск, степень уверенности и ограничения интерпретации. Нормированный риск равен 0.72. Степень принадлежности терму \"высокий риск\" равна 0.8799999999999999, терму \"средний риск\" — 0.1200000000000001. Для представления неопределённости выбран класс FML-audit. Агрегированная неопределённость объяснения равна 0.538444. Потеря при редукции сложного представления составляет 0.20125. При повышенной неопределённости требуется дополнительная проверка или аудит источников."
}
```
