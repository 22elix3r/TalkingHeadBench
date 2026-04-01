import json
import httpx
import traceback
import gradio as gr
from typing import Any, Dict

# Charcoal Black Theme Palette
BG_COLOR = "#121212"
TEXT_PRIMARY = "#E0E0E0"
TEXT_SECONDARY = "#B0B0B0"
BORDER_COLOR = "#444444"
ACCENT_COLOR = "#888888"

custom_theme = gr.themes.Base().set(
    body_background_fill=BG_COLOR,
    body_background_fill_dark=BG_COLOR,
    body_text_color=TEXT_PRIMARY,
    body_text_color_dark=TEXT_PRIMARY,
    background_fill_primary="#181818",
    background_fill_primary_dark="#181818",
    background_fill_secondary=BG_COLOR,
    background_fill_secondary_dark=BG_COLOR,
    border_color_primary=BORDER_COLOR,
    border_color_primary_dark=BORDER_COLOR,
    block_background_fill="#181818",
    block_background_fill_dark="#181818",
    button_primary_background_fill=ACCENT_COLOR,
    button_primary_background_fill_dark=ACCENT_COLOR,
    button_primary_background_fill_hover="#aaaaaa",
    button_primary_background_fill_hover_dark="#aaaaaa",
    button_primary_text_color=BG_COLOR,
    button_primary_text_color_dark=BG_COLOR,
    button_primary_text_color_hover=BG_COLOR,
    button_primary_text_color_hover_dark=BG_COLOR,
    slider_color=ACCENT_COLOR,
    slider_color_dark=ACCENT_COLOR,
)

custom_css = f"""
.gradio-container {{ font-family: 'Inter', sans-serif; }}
.stat-box {{ background-color: #181818; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 15px; margin-bottom: 10px; text-align: center; }}
.stat-value {{ font-size: 24px; font-weight: bold; color: {TEXT_PRIMARY}; }}
.stat-label {{ font-size: 12px; text-transform: uppercase; color: {TEXT_SECONDARY}; letter-spacing: 1px; }}
.header-title {{ color: {TEXT_PRIMARY} !important; text-align: center; border-bottom: 2px solid {BORDER_COLOR}; padding-bottom: 10px; margin-bottom: 20px; }}
textarea, input, select {{ background-color: {BG_COLOR} !important; color: {TEXT_PRIMARY} !important; border: 1px solid {BORDER_COLOR} !important; }}

/* Tier Cards */
.tier-card {{ justify-content: flex-start !important; text-align: left !important; padding: 15px !important; border-radius: 8px !important; border: 2px solid #333 !important; background-color: #181818 !important; white-space: pre-wrap !important; line-height: 1.4 !important; transition: all 0.2s !important; }}
.tier-card:hover {{ border-color: #666 !important; background-color: #222 !important; }}
.tier-selected-image {{ border-color: #22c55e !important; background-color: rgba(34, 197, 94, 0.05) !important; }}
.tier-selected-clip {{ border-color: #f59e0b !important; background-color: rgba(245, 158, 11, 0.05) !important; }}
.tier-selected-weight {{ border-color: #ef4444 !important; background-color: rgba(239, 68, 68, 0.05) !important; }}

/* Subenv Coverage Badges */
.coverage-row {{ display: flex; gap: 10px; margin-bottom: 15px; font-family: 'Inter', sans-serif; }}
.cov-badge {{ padding: 4px 12px; border-radius: 16px; font-size: 0.85em; font-weight: bold; cursor: help; }}
.cov-active-image {{ background-color: #22c55e; color: #fff; border: 1px solid #22c55e; }}
.cov-active-clip {{ background-color: #f59e0b; color: #fff; border: 1px solid #f59e0b; }}
.cov-active-weight {{ background-color: #ef4444; color: #fff; border: 1px solid #ef4444; }}
.cov-inactive {{ background-color: transparent; color: #666; border: 1px solid #444; cursor: default; }}
"""

def format_observation(obs: Dict[str, Any]) -> str:
    """Formats the current observation dict to a readable markdown format."""
    if not obs:
        return "Not initialized. Please click **Initialize Scenario**."
    
    node = obs.get("node", "Unknown")
    step = obs.get("step", 0)
    instr = obs.get("instruction", "No instructions")
    signals = obs.get("signals", {})
    scores = obs.get("scores", None)
    
    md = f"### Step {step}: {node}\n\n**Instructions:** {instr}\n\n"
    
    if signals:
        md += "#### 📡 Available Signals\n```json\n" + json.dumps(signals, indent=2) + "\n```\n"
    
    if scores:
        md += "#### 🏆 Final Score Report\n"
        for k, v in scores.items():
            if isinstance(v, float):
                md += f"- **{k}**: {v:.3f}\n"
            else:
                md += f"- **{k}**: {v}\n"
                
    return md

def generate_badges(tier: str) -> str:
    color_class = {
        "image_audit": "cov-active-image",
        "clip_audit": "cov-active-clip",
        "weight_audit": "cov-active-weight"
    }.get(tier, "cov-active-image")
    
    b1_active = tier == "image_audit"
    b2_active = tier == "clip_audit"
    b3_active = tier == "weight_audit"
    
    def render_badge(name, active, weight):
        if active:
            return f'<div class="cov-badge {color_class}" title="active in selected audit mode">{name} ✓</div>'
        else:
            return f'<div class="cov-badge cov-inactive">{name} &mdash;</div>'

    html = '<div class="coverage-row">'
    html += render_badge("Image Audit", b1_active, "n/a")
    html += render_badge("Clip Audit", b2_active, "n/a")
    html += render_badge("Weight Audit", b3_active, "n/a")
    html += '</div>'
    return html

async def handle_ingestion(ref_img, clip_vids, lora, tok, prompt, param_json, tier):
    if tier == "image_audit":
        if not ref_img:
            return "Error: reference_image is required for image_audit.", "unknown"
        if not str(prompt or "").strip():
            return "Error: prompt is required for image_audit.", "unknown"
    if tier == "clip_audit" and not clip_vids:
        return "Error: clips (1-12 .mp4) are required for clip_audit.", "unknown"
    if tier == "weight_audit" and not lora:
        return "Error: lora_weights (.safetensors) are required for weight_audit.", "unknown"
        
    try:
        url = "http://127.0.0.1:8000/ingest-artifacts"
        files_data = []
        if ref_img is not None and tier == "image_audit":
            files_data.append(("reference_image", open(ref_img.name, "rb")))
        
        if lora is not None and tier == "weight_audit":
            files_data.append(("lora_weights", open(lora.name, "rb")))
        if tok is not None and tier == "weight_audit":
            files_data.append(("tokenizer_config", open(tok.name, "rb")))
        if clip_vids is not None and tier == "clip_audit":
            for clip in clip_vids:
                files_data.append(("clips", open(clip.name, "rb")))
                
        data = {
            "prompt": str(prompt or ""),
            "param_config_json": str(param_json or "")
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, data=data, files=files_data, timeout=60.0)
            
        if resp.status_code == 200:
            return resp.json().get("ingestion_id", "Error: No ID returned"), tier
        else:
            return f"Error {resp.status_code}: {resp.text}", "unknown"
    except Exception as e:
        return f"Exception: {str(e)}\n{traceback.format_exc()}", "unknown"
        
def format_tier_label(tier: str) -> str:
    if tier == "image_audit":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Image Audit <span style="color:#22c55e;">●</span>]</div>'
    elif tier == "clip_audit":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Clip Audit <span style="color:#f59e0b;">●</span>]</div>'
    elif tier == "weight_audit":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Weight Audit <span style="color:#ef4444;">●</span>]</div>'
    else:
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #888; white-space: nowrap;">[Task: Unknown <span style="color:#666;">○</span>]</div>'

async def handle_analysis(ingestion_id, model_id, provider, api_key, max_tokens, temp, tier):
    if not ingestion_id:
        return "Error: Provide an Ingestion ID first."
    try:
        url = "http://127.0.0.1:8000/analyze-ingestion"
        data = {
            "ingestion_id": str(ingestion_id),
            "provider": str(provider),
            "max_tokens": int(max_tokens),
            "temperature": float(temp)
        }
        if model_id:
            data["model_id"] = str(model_id)
        if api_key:
            data["api_key"] = str(api_key)
        if tier in ["image_audit", "clip_audit", "weight_audit"]:
            data["task_tier"] = str(tier)
            
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, timeout=120.0)
            
        if tier not in ["image_audit", "clip_audit", "weight_audit"]:
            tier = "weight_audit" # Backend default
            
        if resp.status_code == 200:
            report = resp.json().get("report", "Error: No report returned")
            badges = generate_badges(tier)
            return badges + "\n\n" + report
        else:
            return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
         return f"Exception: {str(e)}"

TIER_INFO = {
    "image_audit": {
        "title": "Image Audit — Node 1 + Node 2",
        "body": "The agent receives image quality signals and prompt/config context. It must:\n(1) classify image regime,\n(2) identify image/prompt risks,\n(3) detect parameter anomalies with directional fixes.\nEpisode completes after Node 2 with a standalone Sub-env 1 score."
    },
    "clip_audit": {
        "title": "Clip Audit — Node 5",
        "body": "The agent receives clip evidence + dataset context and must recommend\naccept/reject/fix/defer with clear impact reasoning. Episode completes after\nclip disposition with a standalone Sub-env 2 score."
    },
    "weight_audit": {
        "title": "Weight Audit — Node 8",
        "body": "The agent receives weight-derived phoneme risk signals and must rank\nrisky phonemes, predict behavior triggers, identify risky clusters, and\nrecommend mitigations. Episode completes after Node 8 with a standalone\nSub-env 3 score."
    }
}

def build_custom_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    with gr.Blocks(theme=custom_theme, css=custom_css, title="TalkingHeadBench Console") as demo:
        gr.Markdown(f"# {title} 📡", elem_classes=["header-title"])
        
        with gr.Tabs():
            # TAB 1: INGESTION & AUTOMATED ANALYSIS
            with gr.Tab("Workspace Initialization & LLM Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Select Audit Tier & Ingest")
                        
                        task_tier = gr.Textbox(value="image_audit", visible=False)
                        
                        with gr.Row():
                            btn_image = gr.Button("🟢 IMAGE AUDIT\nReference + Prompt\nNode 1+2", elem_classes=["tier-card", "tier-selected-image"])
                            btn_clip = gr.Button("🟡 CLIP AUDIT\nDataset Clips Only\nNode 5", elem_classes=["tier-card"])
                            btn_weight = gr.Button("🔴 WEIGHT AUDIT\nLoRA Weights Only\nNode 8", elem_classes=["tier-card"])

                        with gr.Accordion(label=TIER_INFO["image_audit"]["title"], open=False) as eval_accordion:
                            eval_markdown = gr.Markdown(value=TIER_INFO["image_audit"]["body"])

                        ref_file = gr.File(label="Reference Image (.jpg / .png)")
                        clip_files = gr.File(label="Video Clips (.mp4)", file_count="multiple", visible=False)
                        lora_file = gr.File(label="LoRA Weights (.safetensors)", visible=False)
                        tok_file = gr.File(label="Tokenizer Config (.json)", visible=False)
                        prompt_text = gr.Textbox(label="Text Prompt", value="A man speaking directly to the camera.")
                        param_text = gr.Textbox(label="Config JSON (Optional)", value='{"cfg": 7.0}')
                        
                        ingest_btn = gr.Button("Ingest Reference Image", variant="primary")
                        ingestion_id_out = gr.Textbox(label="Generated Ingestion ID", interactive=False)
                        
                        # Tier Selection Logic
                        def set_image_audit():
                            return [
                                "image_audit",
                                gr.update(visible=True),
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                "Ingest Image Audit Artifacts",
                                gr.update(elem_classes=["tier-card", "tier-selected-image"]),
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(label=TIER_INFO["image_audit"]["title"], open=True),
                                TIER_INFO["image_audit"]["body"],
                            ]

                        def set_clip_audit():
                            return [
                                "clip_audit",
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                "Ingest Clip Audit Artifacts",
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(elem_classes=["tier-card", "tier-selected-clip"]),
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(label=TIER_INFO["clip_audit"]["title"], open=True),
                                TIER_INFO["clip_audit"]["body"],
                            ]

                        def set_weight_audit():
                            return [
                                "weight_audit",
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                gr.update(visible=True),
                                gr.update(visible=True),
                                "Ingest Weight Audit Artifacts",
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(elem_classes=["tier-card"]),
                                gr.update(elem_classes=["tier-card", "tier-selected-weight"]),
                                gr.update(label=TIER_INFO["weight_audit"]["title"], open=True),
                                TIER_INFO["weight_audit"]["body"],
                            ]
                            
                        tier_outputs = [
                            task_tier,
                            ref_file,
                            prompt_text,
                            clip_files,
                            lora_file,
                            tok_file,
                            ingest_btn,
                            btn_image,
                            btn_clip,
                            btn_weight,
                            eval_accordion,
                            eval_markdown,
                        ]
                        btn_image.click(set_image_audit, inputs=[], outputs=tier_outputs)
                        btn_clip.click(set_clip_audit, inputs=[], outputs=tier_outputs)
                        btn_weight.click(set_weight_audit, inputs=[], outputs=tier_outputs)
                        
                        current_task_tier = gr.Textbox(value="unknown", visible=False)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 2. Automated LLM Analysis")
                        gr.Markdown("Use an LLM Diagnostician to automatically analyze the ingested bundle.")
                        
                        with gr.Row():
                            analysis_ingestion_id = gr.Textbox(label="Ingestion ID (auto-filled from step 1)", scale=4)
                            active_tier_label = gr.HTML(value='<div style="margin-top: 28px; font-size: 0.9em; color: #888; white-space: nowrap;">[Task: Unknown <span style="color:#666;">○</span>]</div>', scale=1)
                            
                        analysis_provider = gr.Dropdown(choices=["auto", "openai", "anthropic", "huggingface", "local"], value="auto", label="LLM Provider")
                        analysis_model = gr.Textbox(label="Model ID (Optional formatting)", placeholder="e.g. meta-llama/Llama-3.1-70B-Instruct")
                        analysis_api_key = gr.Textbox(label="API Key (Required for HF/OpenAI/Anthropic)", type="password", placeholder="Enter your token here...")
                        analysis_max_tokens = gr.Slider(minimum=64, maximum=4096, value=700, step=1, label="Max Tokens")
                        analysis_temp = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.01, label="Temperature")
                        
                        analyze_btn = gr.Button("Run Diagnostic Analysis", variant="primary")
                        analysis_report_out = gr.Markdown(label="LLM Final Report")
                        
                        # Link step 1 output to step 2 input
                        ingestion_id_out.change(fn=lambda x: x, inputs=ingestion_id_out, outputs=analysis_ingestion_id)
                        
                        def handle_ingestion_id_change(user_typed_id, last_ingested_id, current_state_tier):
                            if user_typed_id != last_ingested_id and user_typed_id:
                                return "unknown"
                            return current_state_tier
                            
                        analysis_ingestion_id.change(
                            fn=handle_ingestion_id_change,
                            inputs=[analysis_ingestion_id, ingestion_id_out, current_task_tier],
                            outputs=[current_task_tier]
                        )
                        
                        current_task_tier.change(
                            fn=format_tier_label,
                            inputs=[current_task_tier],
                            outputs=[active_tier_label]
                        )
                        
                        ingest_btn.click(
                            handle_ingestion, 
                            inputs=[ref_file, clip_files, lora_file, tok_file, prompt_text, param_text, task_tier],
                            outputs=[ingestion_id_out, current_task_tier]
                        )
                        
                        analyze_btn.click(
                            handle_analysis,
                            inputs=[analysis_ingestion_id, analysis_model, analysis_provider, analysis_api_key, analysis_max_tokens, analysis_temp, current_task_tier],
                            outputs=[analysis_report_out]
                        )

            # TAB 2: MANUAL EVALUATION CONSOLE
            with gr.Tab("Manual Evaluation Console"):
                current_obs = gr.State({})

                with gr.Row():
                    # LEFT SIDEBAR - Environment Status & Actions
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Dashboard")
                            with gr.Row():
                                with gr.Column(elem_classes=["stat-box"]):
                                    step_counter = gr.Markdown("<div class='stat-value'>0/2</div><div class='stat-label'>Step</div>")
                                with gr.Column(elem_classes=["stat-box"]):
                                    node_display = gr.Markdown("<div class='stat-value'>Idle</div><div class='stat-label'>Node</div>")
                                    
                        manual_task_tier = gr.Dropdown(
                            choices=["image_audit", "clip_audit", "weight_audit"],
                            value="image_audit",
                            label="Manual Audit Mode",
                        )
                        manual_ingestion_id = gr.Textbox(label="Enter Ingestion ID to load session")
                        reset_btn = gr.Button("Initialize Scenario", variant="primary")
                        
                        # Auto-fill manual ingestion id if one was generated in Tab 1
                        ingestion_id_out.change(fn=lambda x: x, inputs=ingestion_id_out, outputs=manual_ingestion_id)
                        current_task_tier.change(
                            fn=lambda x: x if x in ["image_audit", "clip_audit", "weight_audit"] else "image_audit",
                            inputs=[current_task_tier],
                            outputs=[manual_task_tier],
                        )
                        
                        with gr.Accordion("Raw Observation Payload", open=False):
                            raw_obs_view = gr.JSON(label="Current Context")

                    # RIGHT MAIN PANEL - Interactive Data
                    with gr.Column(scale=2):
                        obs_markdown = gr.Markdown(value=format_observation({}))
                        
                        # STEP 1: ImageDiagnosticsAction
                        with gr.Group(visible=False) as grp_step1:
                            gr.Markdown("## Action 1: Image Diagnostics")
                            s1_regime = gr.Dropdown(
                                choices=["frontal_simple", "non_frontal", "complex_background", "occluded", "low_quality"],
                                label="Regime Classification", value="frontal_simple"
                            )
                            s1_risk_factors = gr.Textbox(label="Identified Risk Factors (JSON list)", value='["lateral_pose_risk"]')
                            s1_usability = gr.Slider(0.0, 1.0, value=0.5, label="Image Usability Score")
                            s1_reasoning = gr.Textbox(label="Reasoning", value="Image analysis completed.")
                            s1_submit = gr.Button("Execute Diagnostics", variant="primary")

                        # STEP 2: ParamAnomalyAction
                        with gr.Group(visible=False) as grp_step2:
                            gr.Markdown("## Action 2: Parameter Anomaly Detection")
                            s2_risk = gr.Dropdown(
                                choices=["safe", "marginal", "risky", "dangerous"],
                                label="Config Risk Level", value="marginal"
                            )
                            s2_anomalies = gr.Code(
                                label="Anomalies (JSON array of ParameterAnomaly dicts)",
                                language="json",
                                value='[]'
                            )
                            s2_summary = gr.Textbox(label="Risk Summary", value="Parameter analysis completed.")
                            s2_submit = gr.Button("Execute Parameter Check", variant="primary")

                        # CLIP STEP: ClipDispositionAction
                        with gr.Group(visible=False) as grp_clip:
                            gr.Markdown("## Action: Clip Disposition")
                            c_disposition = gr.Dropdown(
                                choices=["accept", "reject", "fix", "defer"],
                                label="Disposition",
                                value="accept",
                            )
                            c_confidence = gr.Slider(0.0, 1.0, value=0.6, label="Confidence")
                            c_rejection = gr.Code(
                                label="Rejection Reasons (JSON array)",
                                language="json",
                                value="[]",
                            )
                            c_fixes = gr.Code(
                                label="Fix Instructions (JSON array)",
                                language="json",
                                value="[]",
                            )
                            c_fix_effort = gr.Dropdown(
                                choices=["", "trivial", "moderate", "high"],
                                value="",
                                label="Estimated Fix Effort (optional)",
                            )
                            c_defer = gr.Textbox(label="Defer Reason (optional)", value="")
                            c_reasoning = gr.Textbox(
                                label="Dataset Impact Reasoning",
                                value="Disposition reasoning based on dossier quality and coverage impact.",
                            )
                            c_override = gr.Dropdown(
                                choices=["not_applicable", "declined", "applied"],
                                value="not_applicable",
                                label="Override Decision",
                            )
                            c_override_just = gr.Textbox(
                                label="Override Justification (optional)",
                                value="",
                            )
                            c_submit = gr.Button("Execute Clip Disposition", variant="primary")

                        # STEP 3: PhonemeRiskAction
                        with gr.Group(visible=False) as grp_step3:
                            gr.Markdown("## Action 3: Phoneme Behavioral Risk Assessment")
                            s3_safety = gr.Dropdown(
                                choices=["safe", "minor_concerns", "moderate_risk", "high_risk", "unsafe"],
                                label="Model Behavioral Safety", value="minor_concerns"
                            )
                            s3_risks = gr.Code(
                                label="Phoneme Risk Ranking (JSON array)",
                                language="json",
                                value='[]'
                            )
                            s3_summary = gr.Textbox(label="Overall Behavioral Summary", value="Behavioral evaluate completed.")
                            s3_submit = gr.Button("Execute Final Assessment", variant="primary")

                        # DONE VIEW
                        with gr.Group(visible=False) as grp_done:
                            gr.Markdown("## 🏁 Episode Complete")
                            score_markdown = gr.Markdown()

                # State update flow
                def update_ui_state(obs: Dict[str, Any]):
                    if not obs: 
                        return [{}, format_observation({}), "<div class='stat-value'>0/2</div><div class='stat-label'>Step</div>", "<div class='stat-value'>Error</div><div class='stat-label'>Node</div>", False, False, False, False, False, ""]
                        
                    step = obs.get("step", 0)
                    node = obs.get("node", "Unknown")
                    is_done = getattr(obs, "done", obs.get("done", False)) # Handle both object getattr and dict get
                    schema = obs.get("expected_action_schema", "")
                    scores = obs.get("scores", None)
                    
                    step_str = f"<div class='stat-value'>{step}/2</div><div class='stat-label'>Step</div>"
                    node_short = node.split(" ")[0] if " " in node else node
                    node_str = f"<div class='stat-value'>{node_short}</div><div class='stat-label'>Node</div>"
                    
                    score_md = ""
                    if is_done and scores:
                        html_scores = ""
                        for k, v in scores.items():
                            val = f"{v:.3f}" if isinstance(v, float) else str(v)
                            html_scores += f"<li><strong>{k}:</strong> <span style='color: {ACCENT_COLOR};'>{val}</span></li>"
                        score_md = f"<ul style='font-size: 1.2rem;'>{html_scores}</ul><hr/><p>Evaluation finished. Change ingestion ID and Reset to run new scenario.</p>"
                    
                    return [
                        obs,
                        format_observation(obs),
                        step_str,
                        node_str,
                        gr.update(visible=(not is_done and schema == "ImageDiagnosticsAction")),
                        gr.update(visible=(not is_done and schema == "ParamAnomalyAction")),
                        gr.update(visible=(not is_done and schema == "ClipDispositionAction")),
                        gr.update(visible=(not is_done and schema == "PhonemeRiskAction")),
                        gr.update(visible=is_done),
                        gr.update(value=score_md)
                    ]

                async def do_reset(ingestion_id, selected_tier):
                    mode_map = {
                        "image_audit": "image",
                        "clip_audit": "clips",
                        "weight_audit": "weights",
                    }
                    payload = {"mode": mode_map.get(selected_tier, "image")}
                    if str(ingestion_id or "").strip():
                        payload["ingestion_id"] = str(ingestion_id)

                    res = await web_manager.reset_environment(payload)
                    obs = res.get("observation", {})
                    return update_ui_state(obs)
                    
                async def do_step1(regime, risks_str, usability, reasoning):
                    try:
                        risks = json.loads(risks_str)
                    except:
                        risks = []
                    action_payload = {
                        "regime_classification": regime,
                        "identified_risk_factors": risks,
                        "image_usability_score": float(usability),
                        "reasoning": reasoning,
                        "prompt_issues": [],
                        "recommended_prompt_modifications": [],
                    }
                    res = await web_manager.step_environment(action_payload)
                    obs = res.get("observation", {})
                    obs["done"] = res.get("done", False) 
                    return update_ui_state(obs)

                async def do_step2(risk_level, anomalies_str, summary):
                    try:
                        anoms = json.loads(anomalies_str)
                    except:
                        anoms = []
                    action_payload = {
                        "config_risk_level": risk_level,
                        "anomalies": anoms,
                        "summary": summary,
                        "predicted_failure_modes": [],
                        "directional_fixes": [],
                    }
                    res = await web_manager.step_environment(action_payload)
                    obs = res.get("observation", {})
                    obs["done"] = res.get("done", False)
                    return update_ui_state(obs)

                async def do_step3(safety_level, risks_str, summary):
                    try:
                        risks = json.loads(risks_str)
                    except:
                        risks = []
                    action_payload = {
                        "model_behavioral_safety": safety_level,
                        "phoneme_risk_ranking": risks,
                        "summary": summary,
                        "predicted_behavior_triggers": [],
                        "risky_phoneme_clusters": [],
                        "mitigation_recommendations": [],
                    }
                    res = await web_manager.step_environment(action_payload)
                    obs = res.get("observation", {})
                    obs["done"] = res.get("done", False)
                    return update_ui_state(obs)

                async def do_clip_step(
                    disposition,
                    confidence,
                    rejection_str,
                    fixes_str,
                    fix_effort,
                    defer_reason,
                    reasoning,
                    override_decision,
                    override_justification,
                ):
                    try:
                        rejection_reasons = json.loads(rejection_str)
                        if not isinstance(rejection_reasons, list):
                            rejection_reasons = []
                    except:
                        rejection_reasons = []

                    try:
                        fix_instructions = json.loads(fixes_str)
                        if not isinstance(fix_instructions, list):
                            fix_instructions = []
                    except:
                        fix_instructions = []

                    action_payload = {
                        "disposition": disposition,
                        "confidence": float(confidence),
                        "rejection_reasons": rejection_reasons or None,
                        "fix_instructions": fix_instructions or None,
                        "estimated_fix_effort": fix_effort or None,
                        "defer_reason": (defer_reason or "").strip() or None,
                        "dataset_impact_reasoning": reasoning,
                        "override_decision": override_decision,
                        "override_justification": (override_justification or "").strip() or None,
                    }

                    res = await web_manager.step_environment(action_payload)
                    obs = res.get("observation", {})
                    obs["done"] = res.get("done", False)
                    return update_ui_state(obs)

                outputs = [
                    raw_obs_view, obs_markdown, step_counter, node_display,
                    grp_step1, grp_step2, grp_clip, grp_step3, grp_done, score_markdown
                ]
                
                reset_btn.click(do_reset, inputs=[manual_ingestion_id, manual_task_tier], outputs=outputs)
                s1_submit.click(do_step1, inputs=[s1_regime, s1_risk_factors, s1_usability, s1_reasoning], outputs=outputs)
                s2_submit.click(do_step2, inputs=[s2_risk, s2_anomalies, s2_summary], outputs=outputs)
                c_submit.click(
                    do_clip_step,
                    inputs=[
                        c_disposition,
                        c_confidence,
                        c_rejection,
                        c_fixes,
                        c_fix_effort,
                        c_defer,
                        c_reasoning,
                        c_override,
                        c_override_just,
                    ],
                    outputs=outputs,
                )
                s3_submit.click(do_step3, inputs=[s3_safety, s3_risks, s3_summary], outputs=outputs)

    return demo
