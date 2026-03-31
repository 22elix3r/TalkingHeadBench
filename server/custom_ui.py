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
.tier-selected-easy {{ border-color: #22c55e !important; background-color: rgba(34, 197, 94, 0.05) !important; }}
.tier-selected-medium {{ border-color: #f59e0b !important; background-color: rgba(245, 158, 11, 0.05) !important; }}
.tier-selected-hard {{ border-color: #ef4444 !important; background-color: rgba(239, 68, 68, 0.05) !important; }}

/* Subenv Coverage Badges */
.coverage-row {{ display: flex; gap: 10px; margin-bottom: 15px; font-family: 'Inter', sans-serif; }}
.cov-badge {{ padding: 4px 12px; border-radius: 16px; font-size: 0.85em; font-weight: bold; cursor: help; }}
.cov-active-easy {{ background-color: #22c55e; color: #fff; border: 1px solid #22c55e; }}
.cov-active-medium {{ background-color: #f59e0b; color: #fff; border: 1px solid #f59e0b; }}
.cov-active-hard {{ background-color: #ef4444; color: #fff; border: 1px solid #ef4444; }}
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
        "easy": "cov-active-easy",
        "medium": "cov-active-medium",
        "hard": "cov-active-hard"
    }.get(tier, "cov-active-easy")
    
    b1_active = True
    b2_active = tier in ["medium", "hard"]
    b3_active = tier == "hard"
    
    def render_badge(name, active, weight):
        if active:
            return f'<div class="cov-badge {color_class}" title="weight: {weight} in final score">{name} ✓</div>'
        else:
            return f'<div class="cov-badge cov-inactive">{name} &mdash;</div>'

    html = '<div class="coverage-row">'
    html += render_badge("Subenv 1", b1_active, "0.25")
    html += render_badge("Subenv 2", b2_active, "0.35")
    html += render_badge("Subenv 3", b3_active, "0.40")
    html += '</div>'
    return html

async def handle_ingestion(ref_img, clip_vids, lora, tok, prompt, param_json, tier):
    if not ref_img:
        return "Error: reference_image is required for all tiers.", "unknown"
    if tier in ["medium", "hard"] and not clip_vids:
        return f"Error: clips (1-12 .mp4) are required for {tier.upper()} tier.", "unknown"
    if tier == "hard" and not lora:
        return "Error: lora_weights (.safetensors) are required for HARD tier.", "unknown"
        
    try:
        url = "http://127.0.0.1:8000/ingest-artifacts"
        files_data = []
        files_data.append(("reference_image", open(ref_img.name, "rb")))
        
        if lora is not None and tier == "hard":
            files_data.append(("lora_weights", open(lora.name, "rb")))
        if tok is not None and tier == "hard":
            files_data.append(("tokenizer_config", open(tok.name, "rb")))
        if clip_vids is not None and tier in ["medium", "hard"]:
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
    if tier == "easy":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Easy <span style="color:#22c55e;">●</span>]</div>'
    elif tier == "medium":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Medium <span style="color:#f59e0b;">●</span>]</div>'
    elif tier == "hard":
        return '<div style="margin-top: 28px; font-size: 0.9em; color: #E0E0E0; white-space: nowrap;">[Task: Hard <span style="color:#ef4444;">●</span>]</div>'
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
        if tier in ["easy", "medium", "hard"]:
            data["task_tier"] = str(tier)
            
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, timeout=120.0)
            
        if tier not in ["easy", "medium", "hard"]:
            tier = "hard" # Backend default
            
        if resp.status_code == 200:
            report = resp.json().get("report", "Error: No report returned")
            badges = generate_badges(tier)
            return badges + "\n\n" + report
        else:
            return f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
         return f"Exception: {str(e)}"

TIER_INFO = {
    "easy": {
        "title": "Node 1 — Image Diagnostician + Node 2 — Parameter Anomaly Detector",
        "body": "The agent will receive image quality signals extracted from your reference photo\nand your generation parameter config. It must:\n(1) classify the image regime (frontal_simple, non_frontal, complex_background,\n    occluded, low_quality),\n(2) identify risk factors in the image and prompt,\n(3) detect parameter anomalies and propose directional fixes.\nSub-env 1 score = 0.25 × final reward. Sub-envs 2 and 3 are not evaluated."
    },
    "medium": {
        "title": "Subenv 1 + Subenv 2 — Dataset Clip Audit",
        "body": "Adds clip-level signal extraction on top of the Easy task. The agent must also:\n(4) assess identity drift severity per clip (none / minor / moderate / severe),\n(5) evaluate lip sync quality and temporal stability,\n(6) recommend clip disposition (accept / reject / conditional),\n(7) reason about dataset health impact.\nSub-env 2 score = 0.35 × final reward. Sub-env 3 is not evaluated."
    },
    "hard": {
        "title": "Full Audit — Subenv 1 + 2 + 3 (LoRA Weight Behavioral Audit)",
        "body": "Adds LoRA weight inspection on top of Medium. The agent must also:\n(8) rank phonemes by behavioral risk score,\n(9) predict phoneme → behavior trigger associations,\n(10) identify risky phoneme clusters,\n(11) recommend mitigations (retrain / remove / add counter-examples /\n     reduce rank / apply regularization / flag for review).\nSub-env 3 score = 0.40 × final reward. Full formula active:\nfinal = 0.25 × s1 + 0.35 × s2 + 0.40 × s3."
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
                        
                        task_tier = gr.Textbox(value="easy", visible=False)
                        
                        with gr.Row():
                            btn_easy = gr.Button("🟢 EASY\nImage & Prompt Audit\nNode 1+2", elem_classes=["tier-card", "tier-selected-easy"])
                            btn_medium = gr.Button("🟡 MEDIUM\nImage + Clip Dataset\nNode 1+2+5/6", elem_classes=["tier-card"])
                            btn_hard = gr.Button("🔴 HARD\nFull Pipeline Audit\nAll Nodes", elem_classes=["tier-card"])

                        with gr.Accordion(label=TIER_INFO["easy"]["title"], open=False) as eval_accordion:
                            eval_markdown = gr.Markdown(value=TIER_INFO["easy"]["body"])

                        ref_file = gr.File(label="Reference Image (.jpg / .png)")
                        clip_files = gr.File(label="Video Clips (.mp4)", file_count="multiple", visible=False)
                        lora_file = gr.File(label="LoRA Weights (.safetensors)", visible=False)
                        tok_file = gr.File(label="Tokenizer Config (.json)", visible=False)
                        prompt_text = gr.Textbox(label="Text Prompt (Optional)", value="A man speaking directly to the camera.")
                        param_text = gr.Textbox(label="Config JSON (Optional)", value='{"cfg": 7.0}')
                        
                        ingest_btn = gr.Button("Ingest Reference Image", variant="primary")
                        ingestion_id_out = gr.Textbox(label="Generated Ingestion ID", interactive=False)
                        
                        # Tier Selection Logic
                        def set_easy():
                            return ["easy", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "Ingest Reference Image", gr.update(elem_classes=["tier-card", "tier-selected-easy"]), gr.update(elem_classes=["tier-card"]), gr.update(elem_classes=["tier-card"]), gr.update(label=TIER_INFO["easy"]["title"], open=True), TIER_INFO["easy"]["body"]]
                        def set_medium():
                            return ["medium", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "Ingest Image + Clips", gr.update(elem_classes=["tier-card"]), gr.update(elem_classes=["tier-card", "tier-selected-medium"]), gr.update(elem_classes=["tier-card"]), gr.update(label=TIER_INFO["medium"]["title"], open=True), TIER_INFO["medium"]["body"]]
                        def set_hard():
                            return ["hard", gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), "Ingest All Artifacts", gr.update(elem_classes=["tier-card"]), gr.update(elem_classes=["tier-card"]), gr.update(elem_classes=["tier-card", "tier-selected-hard"]), gr.update(label=TIER_INFO["hard"]["title"], open=True), TIER_INFO["hard"]["body"]]
                            
                        tier_outputs = [task_tier, clip_files, lora_file, tok_file, ingest_btn, btn_easy, btn_medium, btn_hard, eval_accordion, eval_markdown]
                        btn_easy.click(set_easy, inputs=[], outputs=tier_outputs)
                        btn_medium.click(set_medium, inputs=[], outputs=tier_outputs)
                        btn_hard.click(set_hard, inputs=[], outputs=tier_outputs)
                        
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
                                    step_counter = gr.Markdown("<div class='stat-value'>0/3</div><div class='stat-label'>Step</div>")
                                with gr.Column(elem_classes=["stat-box"]):
                                    node_display = gr.Markdown("<div class='stat-value'>Idle</div><div class='stat-label'>Node</div>")
                                    
                        manual_ingestion_id = gr.Textbox(label="Enter Ingestion ID to load session")
                        reset_btn = gr.Button("Initialize Scenario", variant="primary")
                        
                        # Auto-fill manual ingestion id if one was generated in Tab 1
                        ingestion_id_out.change(fn=lambda x: x, inputs=ingestion_id_out, outputs=manual_ingestion_id)
                        
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
                        return [{}, format_observation({}), "<div class='stat-value'>0/3</div><div class='stat-label'>Step</div>", "<div class='stat-value'>Error</div><div class='stat-label'>Node</div>", False, False, False, False, ""]
                        
                    step = obs.get("step", 0)
                    node = obs.get("node", "Unknown")
                    is_done = getattr(obs, "done", obs.get("done", False)) # Handle both object getattr and dict get
                    schema = obs.get("expected_action_schema", "")
                    scores = obs.get("scores", None)
                    
                    step_str = f"<div class='stat-value'>{step}/3</div><div class='stat-label'>Step</div>"
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
                        gr.update(visible=(not is_done and schema == "PhonemeRiskAction")),
                        gr.update(visible=is_done),
                        gr.update(value=score_md)
                    ]

                async def do_reset(ingestion_id):
                    res = await web_manager.reset_environment({"ingestion_id": str(ingestion_id)})
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

                outputs = [
                    raw_obs_view, obs_markdown, step_counter, node_display,
                    grp_step1, grp_step2, grp_step3, grp_done, score_markdown
                ]
                
                reset_btn.click(do_reset, inputs=[manual_ingestion_id], outputs=outputs)
                s1_submit.click(do_step1, inputs=[s1_regime, s1_risk_factors, s1_usability, s1_reasoning], outputs=outputs)
                s2_submit.click(do_step2, inputs=[s2_risk, s2_anomalies, s2_summary], outputs=outputs)
                s3_submit.click(do_step3, inputs=[s3_safety, s3_risks, s3_summary], outputs=outputs)

    return demo
