# Annotation Worksheet - Sub-env 3

Generated from `tests/test_set/subenv3_cases.json`

---
### Case 001 - `lora_001.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | N/A | unknown |
| high_entropy_token_positions | [] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `safe`
- Top risk phonemes: none
- Flagged token positions: []
- Token->phoneme mapping: none

**Training step context:**
- Source stem: `lora_001`
- Detected step: not detected from source file stem
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<safe - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 002 - `ltx_2.3_talking_head_av_lora_v1.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.175 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.705), AE(0.705), AH(0.705)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `ltx_2.3_talking_head_av_lora_v1`
- Detected step: not detected from source file stem
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 003 - `lora_weights_step_00250.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.147 | low - early training |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.688), AE(0.688), AH(0.688)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_00250`
- Detected step: step_00250: very early - identity not yet established
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 004 - `lora_weights_step_00500.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.160 | moderate - converging |
| high_entropy_token_positions | [0, 1] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.696), AE(0.696)
- Flagged token positions: [0, 1]
- Token->phoneme mapping: 0->AA, 1->AE

**Training step context:**
- Source stem: `lora_weights_step_00500`
- Detected step: step_00500: early - unstable identity
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 005 - `lora_weights_step_00750.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.166 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.700), AE(0.700), AH(0.700)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_00750`
- Detected step: step_00750: mid - improving
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 006 - `lora_weights_step_01000.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.170 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.702), AE(0.702), AH(0.702)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_01000`
- Detected step: step_01000: mid-late - approaching convergence
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 007 - `lora_weights_step_01250.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.173 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.704), AE(0.704), AH(0.704)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_01250`
- Detected step: step_01250: good - identity preservation confirmed (your threshold)
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 008 - `lora_weights_step_01500.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.174 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.705), AE(0.705), AH(0.705)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_01500`
- Detected step: step_01500: converged - minor residual artifacts only
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 009 - `lora_weights_step_01750.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.175 | moderate - converging |
| high_entropy_token_positions | [0, 1] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.705), AE(0.705)
- Flagged token positions: [0, 1]
- Token->phoneme mapping: 0->AA, 1->AE

**Training step context:**
- Source stem: `lora_weights_step_01750`
- Detected step: step_01750: not in predefined map
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

---
### Case 010 - `lora_weights_step_02000.safetensors`

**Extracted signals:**
| Signal | Value | Interpretation |
|--------|-------|----------------|
| lora_rank | N/A | - |
| total_parameters | N/A | - |
| overfitting_signature | N/A | unknown |
| gradient_noise_estimate | N/A | unknown |
| canonical_entropy mean | 0.175 | moderate - converging |
| high_entropy_token_positions | [0, 1, 2] | token positions with anomalous patterns |

**Node 8 heuristic suggests:**
- Safety level: `high_risk`
- Top risk phonemes: AA(0.705), AE(0.705), AH(0.705)
- Flagged token positions: [0, 1, 2]
- Token->phoneme mapping: 0->AA, 1->AE, 2->AH

**Training step context:**
- Source stem: `lora_weights_step_02000`
- Detected step: step_02000: not in predefined map
- step_00250: very early - identity not yet established
- step_00500: early - unstable identity
- step_00750: mid - improving
- step_01000: mid-late - approaching convergence
- step_01250: good - identity preservation confirmed (your threshold)
- step_01500+: converged - minor residual artifacts only

**Your annotation (fill in):**
```json
{
  "model_behavioral_safety": "<high_risk - confirm or change>",
  "phoneme_risk_ranking": [
    {
      "phoneme": "XX",
      "risk_score": 0.0,
      "risk_type": "expression_trigger|identity_trigger|motion_trigger|unknown_anomaly",
      "confidence": 0.0,
      "evidence": ""
    }
  ],
  "predicted_behavior_triggers": [
    {
      "trigger_phoneme": "XX",
      "triggered_behavior": "smile|jaw_drift|head_turn|brow_raise",
      "association_strength": 0.0,
      "is_intended": false,
      "concern_level": "low|medium|high"
    }
  ],
  "risky_phoneme_clusters": [],
  "valid_mitigation_set": [
    [
      "target phoneme or cluster",
      "add_counter_examples|flag_for_manual_review|retrain_with_more_data"
    ]
  ]
}
```

## Summary

- Total cases: 10
- Suggested distribution: {"high_risk": 9, "safe": 1}
- Fields always requiring manual input:
  Sub-env 3: phoneme_risk_ranking, predicted_behavior_triggers, valid_mitigation_set
