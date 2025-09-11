#!/usr/bin/env python3
'''Simple HTTP server for RunPod results'''
import http.server
import socketserver
import os
import socket

os.chdir('large_files/plots')

# Find available port
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

PORT = find_free_port()

Handler = http.server.SimpleHTTPRequestHandler

print("=" * 60)
print(f"Serving results at: http://localhost:{PORT}")
print("=" * 60)
print(f"Files available:")
for f in os.listdir('.'):
    if f.endswith('.html'):
        print(f"  http://localhost:{PORT}/{f}")
print()
print("Press Ctrl+C to stop")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
