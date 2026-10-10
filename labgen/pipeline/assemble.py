import os
import jinja2
import subprocess
import shutil

import json

from pipeline.config import load_settings

def load_config(config_path="settings.json"):
    return load_settings().get("report", {})

def render_latex(template_path, output_tex_path, context):
    # Ensure template_path is absolute or searched correctly
    if not os.path.isabs(template_path):
        base_candidates = [
            os.getcwd(),
            os.path.join(os.getcwd(), "labgen"),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ]
        for base in base_candidates:
            candidate = os.path.join(base, template_path)
            if os.path.exists(candidate):
                template_path = candidate
                break

    template_dir = os.path.dirname(os.path.abspath(template_path))
    template_name = os.path.basename(template_path)
    
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        block_start_string='{%',
        block_end_string='%}',
        variable_start_string='{{',
        variable_end_string='}}',
        comment_start_string='{#',
        comment_end_string='#}',
        autoescape=False
    )
    
    template = env.get_template(template_name)
    rendered = template.render(**context)
    
    with open(output_tex_path, 'w') as f:
        f.write(rendered)

def compile_pdf(tex_file, output_dir):
    """Compiles the tex file to PDF using pdflatex."""
    cmd = ["pdflatex", "-output-directory", output_dir, "-interaction=nonstopmode", tex_file]
    
    print("Compiling LaTeX to PDF using pdflatex...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("pdflatex failed. Output:")
        print(result.stdout)
        print(result.stderr)
        
    pdf_file = tex_file.replace(".tex", ".pdf")
    if os.path.exists(pdf_file):
        print(f"PDF successfully generated: {pdf_file}")
    else:
        print(f"Failed to generate PDF.")
