#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs
import threading

PORT = 8002
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRECTORY = os.path.dirname(BASE_DIR)
USER_DATA_PATH = os.path.join(DIRECTORY, "personaldata_storage", "user_data.json")

os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == '/api/update_user_data':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            with open(USER_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps({'success': True, 'message': '数据已成功更新'})
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(302)
            self.send_header('Location', '/111/auth/app_login_register.html')
            self.end_headers()
            return

        if self.path == '/api/get_user_data':
            try:
                with open(USER_DATA_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            except FileNotFoundError:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'users': []}).encode('utf-8'))
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

os.chdir(DIRECTORY)
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    print(f"服务器运行在 http://localhost:{PORT}")
    httpd.serve_forever()