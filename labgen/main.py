import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

import argparse
import os
import datetime
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm
from pipeline.assemble import load_config, render_latex, compile_pdf
from pipeline.llm import generate_report_sections, generate_circuit_design
from pipeline.research import save_research_context
from pipeline.rag import build_rag_index
from pipeline.cad import design_cad_agent
from pipeline.verify import run_all_checks, write_report, extract_features

def validate_schemdraw_ast(code):
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.Name) and node.id in ['eval', 'exec', 'open', '__import__', 'globals', 'locals']:
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return False
    return True

def sanitize_netlist_comp(comp):
    import re
    if re.search(r'(;|\||&|`|\$|shell\s)', comp, re.IGNORECASE) or '.control' in comp.lower():
        return "* [SANITIZED]"
    return comp

def draw_triac_circuit(output_path):
    with schemdraw.Drawing(file=output_path, show=False) as d:
        d += elm.SourceV().up().label('Vac\n(Sweep)')
        d += elm.Resistor().right().label(r'1k$\Omega$')
        d += elm.Triac().down().label('TRIAC')
        d += elm.Line().left()
        d += elm.Ground()

def create_triac_netlist(run_dir):
    cir_path = os.path.join(run_dir, "triac_iv.cir")
    txt_out = os.path.join(run_dir, "iv_data.txt")

    model_path = os.path.abspath("models/triac.sub").replace('\\', '/')
    netlist = f"""TRIAC V-I Characteristics
.include "{model_path}"

* Circuit
V1 1 0 DC 0
R1 1 2 1k
XT1 2 0 3 TRIAC

* Gate drive (constant current or voltage)
Ig 0 3 DC 5m

* Analysis
.dc V1 -15 15 0.1

.control
    run
    * Plot V(2) vs current (which is I(V1))
    let V_triac = V(2)
    let I_triac = -I(V1)
    wrdata {txt_out} V_triac I_triac
.endc
.end
"""
    with open(cir_path, 'w') as f:
        f.write(netlist)
    return cir_path, txt_out

def execute_dynamic_circuit(run_dir, exp_name, circuit_prompt):
    logger.info("Generating dynamic circuit design via LLM...")
    circuit_json = generate_circuit_design(exp_name, circuit_prompt)

    cir_path = os.path.join(run_dir, "dynamic.cir")
    txt_out = os.path.join(run_dir, "iv_data.txt")

    netlist_content = f"Dynamic Circuit: {exp_name}\n"
    triac_path = os.path.abspath("models/triac.sub").replace('\\', '/')
    diac_path = os.path.abspath("models/diac.sub").replace('\\', '/')
    netlist_str = str(circuit_json.get("netlist_components", [])).upper()
    if "TRIAC" in netlist_str or "triac" in circuit_prompt.lower():
        netlist_content += f'.include "{triac_path}"\n'
    if "DIAC" in netlist_str or "diac" in circuit_prompt.lower():
        netlist_content += f'.include "{diac_path}"\n'
    netlist_content += "\n" 

    netlist_content += "* Circuit\n"
    for comp in circuit_json.get("netlist_components", []):
        netlist_content += f"{sanitize_netlist_comp(comp)}\n"

    netlist_content += f"""
* Analysis
.dc V1 -15 15 0.1

.control
    run
    let V_target = V(2)
    let I_target = -I(V1)
    wrdata {txt_out} V_target I_target
.endc
.end
"""
    with open(cir_path, 'w') as f:
        f.write(netlist_content)

    schem_path = os.path.join(run_dir, "figs", "schematic.png")

    try:
        schemdraw_code = circuit_json.get("schemdraw_code", "")
        if schemdraw_code:
            import schemdraw
            import schemdraw.elements as elm
            safe_builtins = {
                'print': print, 'range': range, 'int': int, 'float': float,
                'str': str, 'list': list, 'dict': dict, 'Exception': Exception,
                'zip': zip, 'enumerate': enumerate, 'len': len
            }
            safe_globals = {
                "__builtins__": safe_builtins,
                "schemdraw": schemdraw,
                "elm": elm
            }
            local_vars = {}
            if validate_schemdraw_ast(schemdraw_code):
                exec(schemdraw_code, safe_globals, local_vars)
                if 'draw_circuit' in local_vars:
                    local_vars['draw_circuit'](schem_path)
                else:
                    draw_triac_circuit(schem_path)
            else:
                draw_triac_circuit(schem_path)
        else:
            draw_triac_circuit(schem_path)
    except Exception as e:
        logger.error(f"Error executing schemdraw_code: {e}")
        draw_triac_circuit(schem_path)

    return cir_path, txt_out, schem_path, circuit_json

def _build_data_table_from_simulation(txt_out: str) -> str:
    import os
    if not os.path.exists(txt_out) and os.path.exists(txt_out.replace('.txt', '_tran.txt')):
        txt_out = txt_out.replace('.txt', '_tran.txt')
    try:
        df = pd.read_csv(txt_out, sep=r'\s+', header=None)
        if df.shape[1] >= 4:
            v = df[1]
            i = df[3]
        elif df.shape[1] >= 2:
            v = df[0]
            i = df[1]
        else:
            return ""

        v_sorted = list(v)
        i_sorted = list(i)

        n_points = min(10, len(v_sorted))
        indices = [int(j * (len(v_sorted) - 1) / (n_points - 1)) for j in range(n_points)]

        rows = []
        for idx in indices:
            v_val = v_sorted[idx]
            i_val = i_sorted[idx] * 1000
            rows.append(f"        {v_val:.1f} & {i_val:.1f} \\\\")

        table = f"""\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{|c|c|}}
        \\hline
        \\textbf{{Voltage (V)}} & \\textbf{{Current (mA)}} \\\\
        \\hline
{chr(10).join(rows)}
        \\hline
    \\end{{tabular}}
    \\caption{{Simulated Data (Sample Points)}}
\\end{{table}}"""
        return table
    except Exception as e:
        logger.error(f"Error building data table: {e}")
        return ""

def run_generation(args, settings):
    slug = args.name.lower().replace(" ", "_")
    run_dir = os.path.join("runs", slug)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "sim"), exist_ok=True)

    logger.info(f"--- Running LabGen for: {args.name} ---")

    logger.info("Initializing RAG index...")
    build_rag_index()

    logger.info("Running LangGraph pipeline...")
    from pipeline.graph import run_pipeline

    circuit_prompt = args.circuit_prompt if args.circuit_prompt else ""

    graph_result = run_pipeline(args.name, circuit_prompt)
    circuit_json = graph_result.get("circuit_json", {})
    llm_sections = graph_result.get("report_sections", {})

    logger.info("Executing dynamic circuit...")
    cir_path = os.path.join(run_dir, "dynamic.cir")
    txt_out = os.path.join(run_dir, "iv_data.txt")
    schem_path = os.path.join(run_dir, "figs", "schematic.png")

    netlist_content = f"Dynamic Circuit: {args.name}\n"
    triac_path = os.path.abspath("models/triac.sub").replace('\\', '/')
    diac_path = os.path.abspath("models/diac.sub").replace('\\', '/')
    netlist_str = str(circuit_json.get("netlist_components", [])).upper()
    if "TRIAC" in netlist_str or "triac" in args.name.lower():
        netlist_content += f'.include "{triac_path}"\n'
    if "DIAC" in netlist_str or "diac" in args.name.lower():
        netlist_content += f'.include "{diac_path}"\n'
    netlist_content += "\n" 

    netlist_content += "* Circuit\n"
    for comp in circuit_json.get("netlist_components", []):
        netlist_content += f"{sanitize_netlist_comp(comp)}\n"


    # Extract components and find a resistor to vary
    comps = [sanitize_netlist_comp(c) for c in circuit_json.get("netlist_components", [])]
    var_res_idx = -1
    for i, c in enumerate(comps):
        if c.strip().upper().startswith("R"):
            var_res_idx = i
            break
            
    r_vals = ["1k", "5k", "10k"] if var_res_idx != -1 else ["1k"]
    colors = ['b', 'r', 'g']
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.title(f"{args.name} Characteristics", fontsize=14)
    plt.xlabel("Voltage (V)", fontsize=12)
    plt.ylabel("Current (mA)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    success_sim = False
    tran_data = []
    
    for r_idx, r_val in enumerate(r_vals):
        loop_txt_out = txt_out.replace('.txt', f'_{r_idx}.txt')
        
        loop_netlist = f"Dynamic Circuit: {args.name}\n"
        if "TRIAC" in netlist_str or "triac" in args.name.lower():
            loop_netlist += f'.include "{triac_path}"\n'
        if "DIAC" in netlist_str or "diac" in args.name.lower():
            loop_netlist += f'.include "{diac_path}"\n'
        loop_netlist += "\n* Circuit\n"
        
        for i, comp in enumerate(comps):
            if i == var_res_idx:
                parts = comp.split()
                if len(parts) >= 4:
                    parts[3] = r_val
                    loop_netlist += " ".join(parts) + "\n"
                else:
                    loop_netlist += f"{comp}\n"
            else:
                loop_netlist += f"{comp}\n"
                
        is_transient = any(kw in args.name.lower() for kw in ["buck", "boost", "converter", "rectifier", "inverter", "oscillator", "filter", "chopper", "switching"])
        
        loop_netlist += f"""
* Analysis
"""
        if is_transient:
            loop_netlist += f"""
.tran 10u 10m

.control
    run
    setplot tran1
    wrdata {loop_txt_out.replace('.txt', '_tran.txt')} time V(2) -I(V1)
.endc
.end
"""
        else:
            loop_netlist += f"""
.dc V1 -15 15 0.1

.control
    run
    let V_target = V(2)
    let I_target = -I(V1)
    wrdata {loop_txt_out} V_target I_target
.endc
.end
"""
        with open(cir_path, 'w') as f:
            f.write(loop_netlist)
            
        res = subprocess.run(["ngspice", "-b", cir_path], capture_output=True)
        if res.returncode == 0 and (os.path.exists(loop_txt_out) or os.path.exists(loop_txt_out.replace('.txt', '_tran.txt'))):
            success_sim = True
            if os.path.exists(loop_txt_out):
                try:
                    df = pd.read_csv(loop_txt_out, sep=r'\s+', header=None)
                    if df.shape[1] >= 4:
                        plt.plot(df[1], df[3] * 1000, linewidth=2, color=colors[r_idx % len(colors)], label=f"R={r_val}")
                    elif df.shape[1] >= 2:
                        plt.plot(df[0], df[1] * 1000, linewidth=2, color=colors[r_idx % len(colors)], label=f"R={r_val}")
                except Exception as e:
                    pass
            
            if os.path.exists(loop_txt_out.replace('.txt', '_tran.txt')):
                try:
                    df_tran = pd.read_csv(loop_txt_out.replace('.txt', '_tran.txt'), sep=r'\s+', header=None)
                    tran_data.append((r_val, colors[r_idx % len(colors)], df_tran))
                except Exception as e:
                    pass

    if not success_sim:
        logger.warning("Dynamic circuit failed, falling back to template circuit...")
        from pipeline.circuit_templates import get_fallback_circuit
        fallback_json = get_fallback_circuit(args.name, circuit_prompt)
        circuit_json.update(fallback_json)
        
        fb_netlist = f"Fallback Circuit: {args.name}\n"
        fb_netlist += "\n".join(circuit_json["netlist_components"])
        
        is_transient = any(kw in args.name.lower() for kw in ["buck", "boost", "converter", "rectifier", "inverter", "oscillator", "filter", "chopper", "switching"])
        
        fb_netlist += "\n* Analysis\n"
        if is_transient:
            fb_netlist += f"""
.tran 10u 10m
.control
    run
    setplot tran1
    wrdata {txt_out.replace('.txt', '_tran.txt')} time V(2) -I(V1)
.endc
.end
"""
        else:
            fb_netlist += f"""
.dc V1 -15 15 0.1
.control
    run
    let V_target = V(2)
    let I_target = -I(V1)
    wrdata {txt_out} V_target I_target
.endc
.end
"""
        with open(cir_path, 'w') as f:
            f.write(fb_netlist)
            
        subprocess.run(["ngspice", "-b", cir_path], capture_output=True)
        if is_transient:
            try:
                df_tran = pd.read_csv(txt_out.replace('.txt', '_tran.txt'), sep=r'\s+', header=None)
                tran_data.append(("Fallback", 'b', df_tran))
            except Exception:
                pass
        else:
            try:
                df = pd.read_csv(txt_out, sep=r'\s+', header=None)
                if df.shape[1] >= 4:
                    plt.plot(df[1], df[3] * 1000, linewidth=2, color='b', label="Fallback")
                elif df.shape[1] >= 2:
                    plt.plot(df[0], df[1] * 1000, linewidth=2, color='b', label="Fallback")
            except Exception:
                pass
    else:
        import shutil
        try:
            if os.path.exists(txt_out.replace('.txt', '_0.txt')):
                shutil.copy(txt_out.replace('.txt', '_0.txt'), txt_out)
            if os.path.exists(txt_out.replace('.txt', '_0_tran.txt')):
                shutil.copy(txt_out.replace('.txt', '_0_tran.txt'), txt_out.replace('.txt', '_tran.txt'))
        except:
            pass
            
    plot_path = os.path.join(run_dir, "figs", f"{slug}_plot.png")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    
    tran_plot_path = None
    if tran_data:
        plt.figure(figsize=(8, 6))
        plt.title(f"{args.name} Transient Response (Voltage)", fontsize=14)
        plt.xlabel("Time (ms)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        for r_val, color, df_tran in tran_data:
            if df_tran.shape[1] >= 4:
                plt.plot(df_tran[1] * 1000, df_tran[3], linewidth=2, color=color, label=f"R={r_val}")
            elif df_tran.shape[1] >= 2:
                plt.plot(df_tran[0] * 1000, df_tran[1], linewidth=2, color=color, label=f"R={r_val}")
        plt.legend()
        plt.tight_layout()
        tran_plot_path = os.path.join(run_dir, "figs", f"{slug}_tran_plot.png")
        plt.savefig(tran_plot_path, dpi=300)
    plt.close()
        
    def _execute_schemdraw(code, path, skip_ast=False):
        import schemdraw
        import schemdraw.elements as elm
        safe_builtins = {
            'print': print, 'range': range, 'int': int, 'float': float,
            'str': str, 'list': list, 'dict': dict, 'Exception': Exception,
            'zip': zip, 'enumerate': enumerate, 'len': len
        }
        safe_globals = {
            "__builtins__": safe_builtins,
            "schemdraw": schemdraw,
            "elm": elm
        }
        local_vars = {}
        if skip_ast or validate_schemdraw_ast(code):
            try:
                exec(code, safe_globals, local_vars)
                if 'draw_circuit' in local_vars:
                    local_vars['draw_circuit'](path)
                    return True
            except Exception as e:
                logger.error(f"Execution error: {e}")
        return False

    schemdraw_code = circuit_json.get("schemdraw_code", "")
    if not (schemdraw_code and _execute_schemdraw(schemdraw_code, schem_path, skip_ast=False)):
        logger.warning("Dynamic schemdraw failed, falling back to template...")
        from pipeline.circuit_templates import get_fallback_circuit
        fallback_json = get_fallback_circuit(args.name, circuit_prompt)
        fb_code = fallback_json.get("schemdraw_code", "")
        if fb_code:
            _execute_schemdraw(fb_code, schem_path, skip_ast=True)

    logger.info("Scraping theory reference images (if enabled)...")
    theory_img_path = ""
    if settings.get("scraper", {}).get("enabled"):
        try:
            from pipeline.scraper_integration import get_theory_image
            theory_img_path = get_theory_image(args.name + " electronic device", run_dir)
        except Exception as e:
            logger.error(f"Scraper integration failed: {e}")

    logger.info("Assembling LaTeX report...")
    config = load_config()

    research_context = graph_result.get("research_context", "")
    save_research_context(run_dir, args.name, research_context)

    data_table_latex = _build_data_table_from_simulation(txt_out)

    def esc(t):
        if isinstance(t, str): return t.replace('_', '\\_')
        if isinstance(t, list): return [esc(x) for x in t]
        return t

    sections = {
        "objectives": esc(llm_sections.get("objectives", [])),
        "theory": esc(llm_sections.get("theory", "")),
        "discussion": esc(llm_sections.get("discussion", "")),
        "conclusion": esc(llm_sections.get("conclusion", "")),
        "procedure": esc(llm_sections.get("procedure", [
            "Connect the setup as per the diagram.",
            "Set up the simulation and initialize parameters.",
            "Record the output and plot the results."
        ])),
        "data_table_latex": data_table_latex,
        "references": ["Generated by LabGen Knowledge Hub API", "Ngspice Simulation Data."]
    }

    if circuit_json:
        sections["circuit_design"] = circuit_json.get("circuit_design_text", "")

    sections["apparatus"] = circuit_json.get("apparatus", []) if circuit_json else []

    context = {
        "config": config,
        "experiment_no": f"{args.exp:02d}",
        "experiment_name": args.name,
        "date_performance": datetime.date.today().strftime("%B %d, %Y"),
        "date_submission": (datetime.date.today() + datetime.timedelta(days=7)).strftime("%B %d, %Y"),
        "sections": sections,
        "circuit_img": os.path.abspath(schem_path).replace('\\\\', '/') if schem_path else "",
        "theory_img": os.path.abspath(theory_img_path).replace('\\\\', '/') if theory_img_path else "",
        "plots": [
            {"path": os.path.abspath(plot_path).replace('\\\\', '/') if plot_path else "", "caption": f"Simulated {args.name} DC Characteristics"}
        ]
    }
    
    if tran_plot_path and os.path.exists(tran_plot_path):
        context["plots"].append({"path": os.path.abspath(tran_plot_path).replace('\\\\', '/'), "caption": f"Simulated {args.name} Transient Response"})

    if args.cad_prompt:
        from pipeline.cad import design_cad_agent
        cad_step_path = os.path.join(run_dir, "cad_model.step")
        success = design_cad_agent(args.cad_prompt, cad_step_path)
        if success:
            cad_svg = os.path.abspath(cad_step_path.replace(".step", ".svg")).replace("\\", "/")
            if os.path.exists(cad_svg):
                context["cad_img"] = cad_svg

    if hasattr(args, 'fluidsim_prompt') and args.fluidsim_prompt:
        from pipeline.fluidsim import generate_fluidsim_circuit
        fs_path = os.path.join(run_dir, "pneumatic_circuit.ct")
        success = generate_fluidsim_circuit(args.fluidsim_prompt, fs_path)
        if success:
            # LaTeX \includegraphics cannot render JSON. Store it in a separate context key.
            context["fluidsim_data"] = os.path.abspath(fs_path.replace(".ct", ".json")).replace("\\", "/")

    pdf_filename = f"Exp_{args.exp:02d}_{slug}.tex"
    tex_out = os.path.join(run_dir, pdf_filename)
    render_latex(os.path.join("templates", "report.tex.j2"), tex_out, context)

    compile_pdf(tex_out, run_dir)

    if settings.get("verification", {}).get("enabled", True):
        logger.info("Running verification...")
        report_bundle = {
            "experiment_name": args.name,
            "sections": sections,
            "circuit_json": circuit_json,
            "iv_data_path": txt_out,
            "research_context": research_context,
            "settings": settings,
            "plots": context["plots"]
        }
        results = run_all_checks(report_bundle)
        write_report(results, os.path.join(run_dir, "verification_report.json"))

    logger.info("Done!")

def run_verification(args, settings):
    if args.input.endswith(".pdf"):
        from pipeline.ingest import extract_pdf
        logger.info(f"Extracting PDF: {args.input}")
        extracted = extract_pdf(args.input)
        if "error" in extracted:
            logger.error(f"Error: {extracted['error']}")
            return

        sections = extracted.get("sections", {})
        latex_table = extracted.get("latex_table", "")
        if latex_table:
            sections["data_table_latex"] = latex_table

        report_bundle = {
            "experiment_name": args.experiment or "Unknown",
            "sections": sections,
            "circuit_json": {},
            "iv_data_path": args.data if args.data and os.path.exists(args.data) else "",
            "research_context": "",
            "settings": settings,
            "plots": []
        }
    else:
        run_dir = args.input
        tex_file = None
        for f in os.listdir(run_dir):
            if f.endswith(".tex"):
                tex_file = os.path.join(run_dir, f)
                break
        if not tex_file:
            logger.warning("No .tex file found in run directory")
            return

        with open(tex_file, "r") as f:
            tex_content = f.read()

        import re
        sections = {}
        section_matches = re.findall(r"\\section\{([^}]+)\}(.*?)(?=\\section|\\end\{document\})", tex_content, re.DOTALL)
        for name, content in section_matches:
            sections[name.lower().replace(" ", "_")] = content.strip()

        table_match = re.search(r"\\begin\{table\}.*?\\end\{table\}", tex_content, re.DOTALL)
        if table_match:
            sections["data_table_latex"] = table_match.group(0)

        iv_path = os.path.join(run_dir, "iv_data.txt")
        if not os.path.exists(iv_path) and args.data:
            iv_path = args.data

        research_path = os.path.join(run_dir, "research_context.json")
        research_context = ""
        if os.path.exists(research_path):
            with open(research_path, "r") as f:
                research_context = json.load(f).get("context", "")

        circuit_json = {}
        circuit_path = os.path.join(run_dir, "dynamic.cir")
        if os.path.exists(circuit_path):
            pass

        report_bundle = {
            "experiment_name": args.experiment or "Unknown",
            "sections": sections,
            "circuit_json": circuit_json,
            "iv_data_path": iv_path,
            "research_context": research_context,
            "settings": settings,
            "plots": []
        }

    logger.info("Running verification...")
    results = run_all_checks(report_bundle)

    print(json.dumps(results, indent=2))

    if args.output:
        write_report(results, args.output)
    else:
        out_path = os.path.join(os.path.dirname(args.input), "verification_report.json") if not args.input.endswith(".pdf") else args.input.replace(".pdf", "_verification.json")
        write_report(results, out_path)

def run_index(args, settings):
    from pipeline.rag import build_rag_index, get_rag_context
    from pipeline.cad import design_cad_agent
    if args.rebuild:
        logger.info("Rebuilding RAG index...")
        build_rag_index(force_rebuild=True)
    else:
        logger.info("Loading/building RAG index...")
        build_rag_index()
    
    if args.query:
        logger.info(f"\nQuery: {args.query}")
        print(get_rag_context(args.query, top_k=args.top_k))

def main():
    parser = argparse.ArgumentParser(description="LabGen - EEE Lab Report Generator & Verifier")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate lab report")
    gen_parser.add_argument("name", help="Name of the experiment")
    gen_parser.add_argument("circuit_prompt", nargs="?", default="", help="Prompt describing circuit connections")
    gen_parser.add_argument("--cad-prompt", help="Prompt for generating 3D CAD mechanical models via CadQuery")
    gen_parser.add_argument("--fluidsim-prompt", help="Prompt for generating FluidSim pneumatic/hydraulic circuits")
    gen_parser.add_argument("--exp", type=int, default=2, help="Experiment number")

    verify_parser = subparsers.add_parser("verify", help="Verify lab report")
    verify_parser.add_argument("input", help="Path to PDF, .tex file, or run directory")
    verify_parser.add_argument("--experiment", help="Experiment name (for PDF verification)")
    verify_parser.add_argument("--data", help="Path to iv_data.txt for data cross-check")
    verify_parser.add_argument("--output", help="Output path for verification report")

    index_parser = subparsers.add_parser("index", help="Manage RAG index")
    index_parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    index_parser.add_argument("--query", help="Test query against index")
    index_parser.add_argument("--top-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    from pipeline.config import load_settings, get_api_key
    settings = load_settings()

    # Commands that don't need API key
    if args.command in ("index", "verify"):
        if args.command == "index":
            run_index(args, settings)
        else:
            run_verification(args, settings)
        return

    api_key = get_api_key()
    if not api_key or api_key == "YOUR_API_KEY":
        logger.error("Error: Please configure your API key in settings.json or export GEMINI_API_KEY.")
        return
    os.environ["GEMINI_API_KEY"] = api_key

    if args.command == "generate":
        run_generation(args, settings)

if __name__ == "__main__":
    main()