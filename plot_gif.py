import json
import os
import argparse
import ast
import io
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 1920, 1080
BG_COLOR = "white"
FONT_COLOR = "black"
GRAPH_PANE_WIDTH = int(IMG_WIDTH * 0.6)
DETAILS_PANE_WIDTH = IMG_WIDTH - GRAPH_PANE_WIDTH
MARGIN = 40
MIN_HORZ_GAP = 1.5
VERT_GAP = 1.5

try:
    FONT_REGULAR = ImageFont.truetype("arial.ttf", 24)
    FONT_BOLD = ImageFont.truetype("arialbd.ttf", 32)
    FONT_CODE = ImageFont.truetype("consola.ttf", 22)
except IOError:
    print("Warning: Arial/Consolas fonts not found. Using default font.")
    FONT_REGULAR = ImageFont.load_default()
    FONT_BOLD = ImageFont.load_default()
    FONT_CODE = ImageFont.load_default()

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

# --- Layout and Drawing Functions ---
def get_tree_layout(G, root, horz_gap=1.5, vert_gap=1.5):
    try:
        # Use pydot/graphviz if available
        return nx.nx_agraph.graphviz_layout(G, prog='dot')
    except (ImportError, OSError):
        pass

    pos = {}

    # 返回每个子树的边界(leftmost, rightmost)
    def first_walk(node, depth=0):
        children = list(G.successors(node))
        if not children:
            pos[node] = (0, -depth)
            return 0, 0  # leftmost, rightmost

        child_boundaries = []
        for child in children:
            left, right = first_walk(child, depth + 1)
            child_boundaries.append([child, left, right])

        # 调整每个子树的位置，防止重叠
        for i in range(1, len(child_boundaries)):
            prev_child, prev_left, prev_right = child_boundaries[i - 1]
            curr_child, curr_left, curr_right = child_boundaries[i]
            shift = prev_right - curr_left + horz_gap
            if shift > 0:
                # 将整个子树向右移动
                def move_subtree(n, delta):
                    x, y = pos[n]
                    pos[n] = (x + delta, y)
                    for c in G.successors(n):
                        move_subtree(c, delta)
                move_subtree(curr_child, shift)
                # 更新当前子树边界
                child_boundaries[i][1] += shift
                child_boundaries[i][2] += shift

        # 父节点居中
        leftmost = child_boundaries[0][1]
        rightmost = child_boundaries[-1][2]
        pos[node] = ((leftmost + rightmost) / 2, -depth)
        return leftmost, rightmost

    def second_walk(node, mod_sum=0):
        x, y = pos[node]
        pos[node] = (x + mod_sum, y * vert_gap)
        for child in G.successors(node):
            second_walk(child, mod_sum)

    first_walk(root)
    second_walk(root)

    # 中心化
    x_coords = [p[0] for p in pos.values()]
    x_offset = -(min(x_coords) + (max(x_coords) - min(x_coords)) / 2)
    return {n: (p[0] + x_offset, p[1]) for n, p in pos.items()}

# --- Load and Text Functions ---
def load_log_data(log_file_path: str):
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at '{log_file_path}'")
        return []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    data.sort(key=lambda item: int(item['rollout_key'].split('_')[-1]))
    return data

def text_wrap(text, font, max_width):
    lines = []
    words = text.split(' ')
    current_line = ""
    for word in words:
        if font.getlength(current_line + " " + word) <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())
    return lines

# --- Frame Creation ---
def create_frame_image(safe_nodes, all_edges, nodes_to_draw, full_layout, current_focus_node_id, title):
    img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    G_anim = nx.DiGraph()
    G_anim.add_nodes_from(nodes_to_draw)
    anim_edges = [(u, v) for u, v in all_edges if u in nodes_to_draw and v in nodes_to_draw]
    G_anim.add_edges_from(anim_edges)

    node_labels = {nid: f"ID:{nid}\nV:{dat.get('value', 0):.1f}\nN:{dat.get('visits', 0)}" for nid, dat in safe_nodes.items() if nid in nodes_to_draw}
    #node_colors = [COLOR_MAP.get(safe_nodes[nid].get('label', 'default').split('_')[0], DEFAULT_COLOR) for nid in G_anim.nodes()]
    node_colors = []
    for nid in G_anim.nodes():
        label = safe_nodes[nid].get('label', 'default')
        color = COLOR_MAP.get(label, DEFAULT_COLOR)
        node_colors.append(color)
    edge_colors = ['red' if n == current_focus_node_id else 'black' for n in G_anim.nodes()]

    fig_width, fig_height = GRAPH_PANE_WIDTH / 100, IMG_HEIGHT / 100
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')

    if G_anim.nodes:
        nx.draw_networkx(G_anim, full_layout, ax=ax, nodelist=list(G_anim.nodes()), labels=node_labels, node_size=2500,
                         node_color=node_colors, font_size=10, width=1.5, edgecolors=edge_colors, node_shape='o',
                         arrows=True, arrowstyle='->', arrowsize=20)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    graph_img = Image.open(buf)
    graph_img = graph_img.resize((GRAPH_PANE_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img.paste(graph_img, (0, 0))

    x_cursor = GRAPH_PANE_WIDTH + MARGIN
    y_cursor = MARGIN
    details_max_width = DETAILS_PANE_WIDTH - (MARGIN * 2)

    draw.text((x_cursor, y_cursor), title, font=FONT_BOLD, fill=FONT_COLOR)
    y_cursor += 60

    node_data = safe_nodes.get(current_focus_node_id)
    if node_data:
        draw.line([(x_cursor, y_cursor), (IMG_WIDTH - MARGIN, y_cursor)], fill="#cccccc", width=2)
        y_cursor += 20

        current_task = node_data.get('current_task', {})
        task_text, code_text = "Task: N/A", "# No code"
        for _, task_data in current_task.items():
            task_text = f"Task: {task_data.get('task', 'N/A')}"
            code_text = task_data.get('code', '# No code')

        for line in text_wrap(task_text, FONT_REGULAR, details_max_width):
            draw.text((x_cursor, y_cursor), line, font=FONT_REGULAR, fill=FONT_COLOR)
            y_cursor += 35
        y_cursor += 15

        draw.text((x_cursor, y_cursor), "Code:", font=FONT_BOLD, fill=FONT_COLOR)
        y_cursor += 40
        code_box_height = FONT_CODE.getbbox(code_text)[3] * (code_text.count('\n') + 2) + 20
        draw.rectangle([x_cursor, y_cursor, IMG_WIDTH - MARGIN, y_cursor + code_box_height], fill="#f0f0f0",
                       outline="#cccccc")
        draw.text((x_cursor + 10, y_cursor + 10), code_text, font=FONT_CODE, fill="#333333")
        y_cursor += code_box_height + 30

        error_msg = f"Error: {node_data.get('error', 'None').strip() or 'None'}"
        error_color = "#d8000c" if "None" not in error_msg else FONT_COLOR
        draw.text((x_cursor, y_cursor), error_msg, font=FONT_REGULAR, fill=error_color)
        y_cursor += 40

        score_msg = f"Score (Node Value): {node_data.get('value', 0.0):.4f}"
        draw.text((x_cursor, y_cursor), score_msg, font=FONT_REGULAR, fill=FONT_COLOR)

    return img

# --- GIF Creation ---
def create_gif(log_data: list, output_gif_path: str, interval_seconds: float):
    all_frames = []
    known_node_ids = set()
    expanded_nodes = set()
    total_rollouts = len(log_data)

    print(f"Starting to generate frames with node-by-node animation...")

    for i, rollout_data in enumerate(log_data):
        print(f"  - Processing Rollout {i + 1}/{total_rollouts}")

        action = rollout_data.get('action', 'update')
        selection_path = rollout_data.get('selected_nodes_path', [])
        snapshot = rollout_data.get('tree_snapshot', {})
        current_nodes_raw = snapshot.get('nodes', {})
        all_edges_raw = snapshot.get('edges', [])

        safe_nodes = {str(nid): (ast.literal_eval(ndata) if isinstance(ndata, str) else ndata)
                      for nid, ndata in current_nodes_raw.items()}
        current_node_ids = set(safe_nodes.keys())
        focus_node_id = str(selection_path[-1]) if selection_path else sorted(list(current_node_ids), key=int)[-1]

        new_node_ids = sorted(list(current_node_ids - known_node_ids), key=int)
        G_full = nx.DiGraph()
        G_full.add_nodes_from(current_node_ids)
        safe_edges = [(str(u), str(v)) for u, v in all_edges_raw if
                      str(u) in current_node_ids and str(v) in current_node_ids]
        G_full.add_edges_from(safe_edges)

        if not G_full.nodes() or not nx.is_arborescence(G_full):
            print(f"    Warning: Graph in rollout {i + 1} is not a valid tree. Skipping frame.")
            continue

        root_node = [n for n, d in G_full.in_degree() if d == 0][0]
        full_layout = get_tree_layout(G_full, root_node)

        if not new_node_ids:
            title = f"Rollout{i + 1}({action}): Node {focus_node_id}"
            frame = create_frame_image(safe_nodes, safe_edges, current_node_ids, full_layout, focus_node_id, title)
            all_frames.append(frame)
        else:
            nodes_to_draw = set(expanded_nodes)
            for new_id in new_node_ids:
                nodes_to_draw.add(new_id)
                expanded_nodes.add(new_id)
                title = f"Rollout{i + 1}({action}): Node {new_id} appears"
                frame = create_frame_image(safe_nodes, safe_edges, nodes_to_draw, full_layout, new_id, title)
                all_frames.append(frame)

        known_node_ids = current_node_ids

    if not all_frames:
        print("No frames were generated. Exiting.")
        return

    print(f"\nAssembling {len(all_frames)} frames into a GIF...")
    all_frames[0].save(
        output_gif_path, format='GIF', save_all=True,
        append_images=all_frames[1:], duration=int(interval_seconds * 1000), loop=0
    )
    print(f"GIF saved successfully to '{output_gif_path}'")

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an animated GIF directly from an MCTS log file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-l", "--log-file", type=str, default='./graph_GEE_0004_rollout_log.jsonl',
        help="Path to the input log file (.jsonl)."
    )
    parser.add_argument(
        "-o", "--output-gif", type=str, default='./mcts_animation_final.gif',
        help="Path for the output GIF file."
    )
    parser.add_argument(
        "-s", "--interval", type=float, default=1.0,
        help="Interval in seconds between each frame of the GIF."
    )
    args = parser.parse_args()

    log_data = load_log_data(args.log_file)
    if log_data:
        create_gif(log_data, args.output_gif, args.interval)
    else:
        print("Could not load data. Exiting.")
