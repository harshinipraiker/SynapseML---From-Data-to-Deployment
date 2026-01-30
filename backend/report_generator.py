import os
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import uuid

# Ensure directory exists
os.makedirs('reports', exist_ok=True)

def generate_report(results: dict):
    """
    Generates a PDF report from the AutoML results.
    """
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report.html')

    # Render the HTML template with the results
    html_out = template.render(results)

    # Generate a unique filename for the report
    report_filename = f"report_{uuid.uuid4()}.pdf"
    report_path = os.path.join('reports', report_filename)
    
    # Generate PDF
    HTML(string=html_out).write_pdf(report_path)
    
    print(f"Report generated and saved to {report_path}")
    return report_path