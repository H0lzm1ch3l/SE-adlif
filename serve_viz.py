import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs

# Load the data
with open('neuron_viz_data.json', 'r') as f:
    data = json.load(f)

# Load the HTML template
with open('neuron_viz.html', 'r') as f:
    html_template = f.read()

# Embed the data into the HTML
html_content = html_template.replace(
    "fetch('neuron_viz_data.json')",
    f"Promise.resolve({json.dumps(data)})"
).replace(
    ".then(response => response.json())",
    ".then(data => data)"
)

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html_content.encode())
        else:
            self.send_error(404)

# Start server
PORT = 0  # Auto-assign port
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    port = httpd.server_address[1]
    print(f"Serving neuron visualization at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()