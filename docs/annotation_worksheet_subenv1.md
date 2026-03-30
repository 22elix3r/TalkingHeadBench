# Annotation Worksheet - Sub-env 1

Generated from `tests/test_set/subenv1_cases.json`

---
### Case 001 - `ref_001.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.267 | moderate - face visible but background present |
| estimated_yaw_degrees | 0.202 deg | frontal |
| background_complexity_score | 0.498 | moderate complexity |
| lighting_uniformity_score | 0.247 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.150 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 002 - `ref_002.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.192 | low - face too small for reliable identity encoding |
| estimated_yaw_degrees | 1.184 deg | frontal |
| background_complexity_score | 0.674 | moderate complexity |
| lighting_uniformity_score | 0.318 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.112 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `low_quality`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<low_quality - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 003 - `ref_003.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.273 | moderate - face visible but background present |
| estimated_yaw_degrees | 1.126 deg | frontal |
| background_complexity_score | 0.832 | complex - busy background competes for attention |
| lighting_uniformity_score | 0.070 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.082 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `complex_background`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<complex_background - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 004 - `ref_004.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.257 | moderate - face visible but background present |
| estimated_yaw_degrees | -0.750 deg | frontal |
| background_complexity_score | 0.229 | simple background |
| lighting_uniformity_score | 0.287 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.024 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 005 - `ref_005.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.230 | low - face too small for reliable identity encoding |
| estimated_yaw_degrees | 0.295 deg | frontal |
| background_complexity_score | 0.275 | simple background |
| lighting_uniformity_score | 0.264 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.035 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `low_quality`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<low_quality - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 006 - `ref_006.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.296 | moderate - face visible but background present |
| estimated_yaw_degrees | 33.995 deg | non-frontal - lateral pose |
| background_complexity_score | 0.168 | simple background |
| lighting_uniformity_score | 0.333 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.027 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `non_frontal`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<non_frontal - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 007 - `ref_007.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.271 | moderate - face visible but background present |
| estimated_yaw_degrees | 48.248 deg | extreme non-frontal - profile view |
| background_complexity_score | 0.241 | simple background |
| lighting_uniformity_score | 0.316 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.035 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `non_frontal`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<non_frontal - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 008 - `ref_008.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.326 | moderate - face visible but background present |
| estimated_yaw_degrees | -71.201 deg | extreme non-frontal - profile view |
| background_complexity_score | 0.408 | moderate complexity |
| lighting_uniformity_score | 0.305 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.054 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `non_frontal`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<non_frontal - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 009 - `ref_009.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.245 | low - face too small for reliable identity encoding |
| estimated_yaw_degrees | -49.318 deg | extreme non-frontal - profile view |
| background_complexity_score | 0.348 | simple background |
| lighting_uniformity_score | 0.204 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.062 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `low_quality`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<low_quality - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 010 - `ref_010.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.246 | low - face too small for reliable identity encoding |
| estimated_yaw_degrees | 0.777 deg | frontal |
| background_complexity_score | 1.000 | complex - busy background competes for attention |
| lighting_uniformity_score | 0.347 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.111 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `low_quality`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<low_quality - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 011 - `ref_011.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.274 | moderate - face visible but background present |
| estimated_yaw_degrees | 0.214 deg | frontal |
| background_complexity_score | 0.143 | simple background |
| lighting_uniformity_score | 0.317 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.028 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 012 - `ref_012.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.975 | good - face dominant in frame |
| estimated_yaw_degrees | 1.674 deg | frontal |
| background_complexity_score | 0.000 | simple background |
| lighting_uniformity_score | 0.143 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.070 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 013 - `ref_013.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.355 | moderate - face visible but background present |
| estimated_yaw_degrees | -26.307 deg | non-frontal - lateral pose |
| background_complexity_score | 0.098 | simple background |
| lighting_uniformity_score | 0.331 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.013 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `non_frontal`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<non_frontal - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 014 - `ref_014.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.270 | moderate - face visible but background present |
| estimated_yaw_degrees | -0.354 deg | frontal |
| background_complexity_score | 0.115 | simple background |
| lighting_uniformity_score | 0.220 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.029 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 015 - `ref_015.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.301 | moderate - face visible but background present |
| estimated_yaw_degrees | 0.079 deg | frontal |
| background_complexity_score | 0.147 | simple background |
| lighting_uniformity_score | 0.208 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.033 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 016 - `ref_016.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.311 | moderate - face visible but background present |
| estimated_yaw_degrees | -1.065 deg | frontal |
| background_complexity_score | 0.095 | simple background |
| lighting_uniformity_score | 0.279 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.014 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 017 - `ref_017.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.953 | good - face dominant in frame |
| estimated_yaw_degrees | 0.376 deg | frontal |
| background_complexity_score | 0.025 | simple background |
| lighting_uniformity_score | 0.249 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.048 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 018 - `ref_018.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.953 | good - face dominant in frame |
| estimated_yaw_degrees | 0.376 deg | frontal |
| background_complexity_score | 0.025 | simple background |
| lighting_uniformity_score | 0.249 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.048 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 019 - `ref_019.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.953 | good - face dominant in frame |
| estimated_yaw_degrees | 0.376 deg | frontal |
| background_complexity_score | 0.025 | simple background |
| lighting_uniformity_score | 0.249 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.048 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 020 - `ref_020.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.953 | good - face dominant in frame |
| estimated_yaw_degrees | 0.376 deg | frontal |
| background_complexity_score | 0.025 | simple background |
| lighting_uniformity_score | 0.249 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.048 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 021 - `ref_021.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.305 | moderate - face visible but background present |
| estimated_yaw_degrees | -0.151 deg | frontal |
| background_complexity_score | 1.000 | complex - busy background competes for attention |
| lighting_uniformity_score | 0.112 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.156 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `complex_background`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<complex_background - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 022 - `ref_022.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.229 | low - face too small for reliable identity encoding |
| estimated_yaw_degrees | -34.155 deg | non-frontal - lateral pose |
| background_complexity_score | 0.114 | simple background |
| lighting_uniformity_score | 0.237 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.027 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `low_quality`
- Risk factors:
  - yaw exceeds 25° — lateral pose reduces reference token coverage
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<low_quality - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "yaw exceeds 25\u00b0 \u2014 lateral pose reduces reference token coverage",
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 023 - `ref_023.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.284 | moderate - face visible but background present |
| estimated_yaw_degrees | -0.716 deg | frontal |
| background_complexity_score | 1.000 | complex - busy background competes for attention |
| lighting_uniformity_score | 0.185 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.220 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `complex_background`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<complex_background - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

---
### Case 024 - `ref_024.PNG`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| face_occupancy_ratio | 0.279 | moderate - face visible but background present |
| estimated_yaw_degrees | -16.236 deg | slight turn - borderline |
| background_complexity_score | 0.099 | simple background |
| lighting_uniformity_score | 0.308 | uneven - shadow regions risk inconsistent rendering |
| estimated_sharpness | 0.017 | blurry - may cause degraded identity encoding |
| occlusion_detected | False | clean |

**Node 1 heuristic suggests:**
- Regime: `frontal_simple`
- Risk factors:
  - lighting uniformity low — shadow regions risk inconsistent skin rendering
  - face occupancy below 0.4 — background competes for attention budget
  - low sharpness — may cause blurred identity encoding
- Prompt issues: none

**Your annotation (fill in):**
```json
{
  "regime_classification": "<frontal_simple - confirm or change>",
  "acceptable_regimes": [],
  "identified_risk_factors": [
    "lighting uniformity low \u2014 shadow regions risk inconsistent skin rendering",
    "face occupancy below 0.4 \u2014 background competes for attention budget",
    "low sharpness \u2014 may cause blurred identity encoding"
  ],
  "valid_prompt_modifications": [
    "<write 1-3 actionable modifications for this specific image>"
  ]
}
```

## Summary

- Total cases: 24
- Suggested distribution: {"complex_background": 3, "frontal_simple": 12, "low_quality": 5, "non_frontal": 4}
- Fields always requiring manual input:
  Sub-env 1: valid_prompt_modifications (always manual)
