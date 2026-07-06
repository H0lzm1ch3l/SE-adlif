import json
from torch.utils.tensorboard import SummaryWriter

# Load the JSON data
with open('neuron_viz_data.json', 'r') as f:
    data = json.load(f)

# Load the HTML template
with open('neuron_viz.html', 'r') as f:
    html_template = f.read()

# Embed the data into the HTML
html_content = html_template.replace(
    """fetch('neuron_viz_data.json')
            .then(response => response.json())
            .then(json => {
                data = json;
                initViz();
            });""",
    f"""Promise.resolve({json.dumps(data)})
            .then(json => {{
                data = json;
                initViz();
            }});"""
)

# Log to TensorBoard
writer = SummaryWriter("tb_logs")
writer.add_text("Neuron Visualization", html_content, 0)
writer.close()

print("Logged to TensorBoard")