# backend/pipeline_viz.py
import graphviz
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import traceback

def _get_step_name(obj):
    """Helper to get a user-friendly name for a scikit-learn object."""
    return obj.__class__.__name__

def generate_lifecycle_flow(pipeline: Pipeline, model_name: str, file_path: str):
    """
    Generates a high-level ML lifecycle flowchart for a scikit-learn pipeline.
    """
    try:
        dot = graphviz.Digraph(
            'ML_Lifecycle_Flow',
            graph_attr={'rankdir': 'LR', 'splines': 'ortho', 'nodesep': '0.5'}
        )
        node_styles = {
            'data': {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#a7c7e7'},
            'process': {'shape': 'record', 'style': 'rounded,filled', 'fillcolor': '#fdfd96'},
            'model': {'shape': 'box3d', 'style': 'filled', 'fillcolor': '#ffb347'},
            'eval': {'shape': 'Mdiamond', 'style': 'filled', 'fillcolor': '#77dd77'},
            'deploy': {'shape': 'cds', 'style': 'filled', 'fillcolor': '#c3b1e1'}
        }
        dot.node('A', 'Input\nDataset', **node_styles['data'])
        preprocessor_details = []
        preprocessor = pipeline.named_steps.get('preprocessor')
        if isinstance(preprocessor, ColumnTransformer):
            for name, transformer_pipeline, columns in preprocessor.transformers_:
                if transformer_pipeline != 'drop' and len(columns) > 0:
                    if isinstance(transformer_pipeline, Pipeline):
                        steps = ' -> '.join([_get_step_name(s) for _, s in transformer_pipeline.steps])
                        preprocessor_details.append(f'• {name.capitalize()} Pipeline: {steps}')
                    else:
                        preprocessor_details.append(f'• {_get_step_name(transformer_pipeline)} on {name}')
        prep_label = 'Preprocessing'
        if preprocessor_details:
            prep_label += r'\n\n' + r'\l'.join(preprocessor_details) + r'\l'
        dot.node('B', f'{{ {prep_label} }}', **node_styles['process'])
        dot.node('C', f'Model Training\n({model_name})', **node_styles['model'])
        dot.node('D', 'Evaluation\n(Metrics & SHAP)', **node_styles['eval'])
        dot.node('E', 'Deployment\n(Save to MLflow)', **node_styles['deploy'])
        dot.edge('A', 'B'); dot.edge('B', 'C'); dot.edge('C', 'D'); dot.edge('D', 'E')
        output_filename = os.path.splitext(file_path)[0]
        dot.render(output_filename, format='png', cleanup=True)
        print(f"--- [VIZ] Lifecycle flowchart saved to {file_path} ---")
        return file_path
    except Exception:
        print("!!!!!!!!!! [VIZ CRITICAL ERROR] Graphviz failed to render. Check installation. !!!!!!!!!!")
        traceback.print_exc()
        return None