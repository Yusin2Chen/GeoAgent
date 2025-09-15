import json
import html
import os
import argparse
import ast  # Keep for robustness

# --- Core Functions ---

# Color map for node labels
COLOR_MAP = {
    'default': '#add8e6',  # lightblue
    'expand': '#90ee90',  # lightgreen
    'fix': '#ffa500',  # orange
    'select': '#ffd700',  # gold
    'dead': '#ff9999',  # Light red
    'simulate': '#dda0dd',  # plum
    'fix_simulate': 'yellow',
    'default_simulate': 'cyan',
    'expand_simulate': 'lime',
    'initial_step': '#1e90ff',  # deepskyblue
    'fix-attempt': '#ffa500',
    'simulation_success': '#3cb371',  # mediumseagreen
    'simulation_failed': '#ff6347',  # tomato
    're-decompose': '#b22222'  # firebrick
}
DEFAULT_COLOR = '#d3d3d3'  # lightgrey


def load_log_data(log_file_path: str) -> list:
    """Loads and sorts rollout data from the JSON Lines log file."""
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at '{log_file_path}'")
        return []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    data.sort(key=lambda item: int(item['rollout_key'].split('_')[-1]))
    return data


def create_mermaid_definition(rollout_data: dict) -> str:
    """Creates a Mermaid.js graph definition string from a tree snapshot."""
    snapshot = rollout_data.get('tree_snapshot', {})
    nodes = snapshot.get('nodes', {})
    edges = snapshot.get('edges', [])
    selection_path = rollout_data.get('selected_nodes_path', [])

    mermaid_lines = ["graph TD;"]

    for node_id, node_data in nodes.items():
        node_dict = node_data
        if isinstance(node_dict, str):
            try:
                node_dict = ast.literal_eval(node_dict)
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse node data string for node ID {node_id}: {node_data}")
                continue

        label_key = node_dict.get('label', 'default')
        base_label = label_key.split('_')[0]
        color = COLOR_MAP.get(label_key, COLOR_MAP.get(base_label, DEFAULT_COLOR))
        penwidth = '4px' if int(node_id) in selection_path else '2px'

        # --- CHANGE 1: Modified node text to be a single, compact line ---
        node_text = (
            f"ID: {node_id} | V: {node_dict.get('value', 0):.2f} | N: {node_dict.get('visits', 0)}"
        )
        safe_node_text = html.escape(node_text).replace('"', '&quot;')

        mermaid_lines.append(f'    {node_id}["{safe_node_text}"];')
        mermaid_lines.append(f'    style {node_id} fill:{color},stroke:#333,stroke-width:{penwidth};')

    for u, v in edges:
        mermaid_lines.append(f'    {u} --> {v};')

    return "\n".join(mermaid_lines)


def generate_cumulative_node_details(rollout_data: dict, rollout_index: int) -> str:
    """
    --- CHANGE 2: New function to generate the cumulative node details pane. ---
    Generates an HTML block showing details for all nodes up to the current rollout.
    """
    snapshot = rollout_data.get('tree_snapshot', {})
    nodes = snapshot.get('nodes', {})
    safe_html = lambda s: html.escape(str(s))

    output_html = [f"<h2>Node Details up to Rollout {rollout_index + 1}</h2>"]

    # Sort node IDs in descending order to show the newest nodes first
    sorted_node_ids = sorted(nodes.keys(), key=int, reverse=True)

    for node_id in sorted_node_ids:
        node_data = nodes[node_id]
        node_dict = node_data
        if isinstance(node_dict, str):
            try:
                node_dict = ast.literal_eval(node_dict)
            except (ValueError, SyntaxError):
                continue  # Skip malformed nodes

        # Start a block for this node
        node_html = [f"<div class='node-details-block'>"]
        node_html.append(f"<h4>Node {node_id}</h4>")

        # Extract and display Task and Code from the 'current_task' dictionary
        current_task = node_dict.get('current_task', {})
        if not current_task:
            node_html.append("<p><em>Node has no task data (e.g., initial root).</em></p>")
        else:
            for task_step, task_data in current_task.items():
                node_html.append(
                    f"<p><strong>Task (Step {task_step}):</strong> {safe_html(task_data.get('task', 'N/A'))}</p>")
                node_html.append(
                    f"<strong>Code:</strong><pre><code>{safe_html(task_data.get('code', '# No code'))}</code></pre>")

        # Extract and display Error
        error_msg = node_dict.get('error', '').strip()
        if error_msg:
            node_html.append(f"<p class='error-msg'><strong>Error:</strong> {safe_html(error_msg)}</p>")
        else:
            node_html.append("<p><strong>Error:</strong> None</p>")

        # Extract and display Score (using the node's 'value')
        score = node_dict.get('value', 0.0)
        node_html.append(f"<p><strong>Score (Node Value):</strong> {score:.4f}</p>")

        node_html.append("</div>")
        output_html.extend(node_html)

    return "".join(output_html)


def generate_html_visualization(log_data: list, output_path: str):
    """Generates the final interactive HTML file using Mermaid.js."""
    mermaid_definitions = []
    details_html = []

    print(f"Processing {len(log_data)} rollout snapshots for Mermaid.js...")
    for i, data in enumerate(log_data):
        mermaid_definitions.append(create_mermaid_definition(data))
        # Use the new function to generate the details pane content
        details_html.append(generate_cumulative_node_details(data, i))
    print("All snapshots processed.")

    # --- CHANGE 3: Updated HTML template with English text and new CSS ---
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MCTS Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; display: flex; flex-direction: column; height: 100vh; margin: 0; }}
            .controls {{ padding: 10px; border-bottom: 2px solid #ccc; background: #f4f4f4; text-align: center; }}
            #rollout-slider {{ width: 80%; max-width: 800px; vertical-align: middle; }}
            #rollout-label {{ font-size: 1.2em; font-weight: bold; margin-left: 15px; }}
            .container {{ display: flex; flex: 1; overflow: hidden; }}
            .pane {{ height: 100%; overflow: auto; padding: 15px; box-sizing: border-box; }}
            .graph-pane {{ flex: 3; border-right: 2px solid #ccc; text-align: center; }}
            .details-pane {{ flex: 2; background: #fafafa; }}
            .node-details-block {{ border: 1px solid #ccc; border-radius: 5px; margin-bottom: 15px; padding: 10px; background: #fff; }}
            .error-msg {{ color: #d8000c; }}
            pre {{ background: #eee; border: 1px solid #ddd; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }}
            code {{ font-family: 'Courier New', Courier, monospace; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <input type="range" id="rollout-slider" min="0" max="{len(log_data) - 1}" value="0" step="1">
            <span id="rollout-label">Rollout: 1/{len(log_data)}</span>
        </div>
        <div class="container">
            <div id="graph-container" class="pane graph-pane"></div>
            <div id="details-container" class="pane details-pane"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: false, theme: 'base' }});

            const mermaidData = {json.dumps(mermaid_definitions)};
            const detailsData = {json.dumps(details_html, ensure_ascii=False)};

            const slider = document.getElementById('rollout-slider');
            const graphContainer = document.getElementById('graph-container');
            const detailsContainer = document.getElementById('details-container');
            const label = document.getElementById('rollout-label');

            const renderGraph = async (index) => {{
                if (index < 0 || index >= mermaidData.length) return;
                const mermaidCode = mermaidData[index];
                const graphId = 'mcts-graph-' + index;
                try {{
                    const {{ svg }} = await mermaid.render(graphId, mermaidCode);
                    graphContainer.innerHTML = svg;
                }} catch (e) {{
                    const safeError = new Option(e.message).innerHTML;
                    graphContainer.innerHTML = `<strong>Error rendering graph:</strong><pre>${{safeError}}</pre>`;
                }}
            }};

            function updateView(index) {{
                const idx = parseInt(index, 10);
                renderGraph(idx);
                detailsContainer.innerHTML = detailsData[idx];
                label.textContent = `Rollout: ${{idx + 1}}/{len(log_data)}`;
            }}

            slider.addEventListener('input', (event) => {{
                updateView(event.target.value);
            }});

            updateView(0);
        </script>
    </body>
    </html>
    """

    print(f"Saving visualization to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print("Done.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an interactive MCTS visualization from a log file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-l", "--log-file",
        type=str,
        default='./graph_GEE_0004_rollout_log.jsonl',
        help="Path to the input log file (.jsonl)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default='./my_mcts_run_visualization.html',
        help="Path for the output HTML file"
    )

    args = parser.parse_args()

    log_file_to_process = args.log_file
    output_html_file = args.output

    print(f"Reading log file: {log_file_to_process}")
    print(f"Generating HTML file: {output_html_file}")

    log_data = load_log_data(log_file_to_process)

    if log_data:
        generate_html_visualization(log_data, output_html_file)
    else:
        print("Could not load data. Exiting.")
        