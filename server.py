from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSMiddleware(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add Cross-Origin Isolation headers
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

# Run the server
if __name__ == "__main__":
    PORT = 8000
    httpd = HTTPServer(('localhost', PORT), CORSMiddleware)
    print(f"Serving on http://localhost:{PORT}")
    httpd.serve_forever()
