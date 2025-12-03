const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const { createProxyServer } = require('http-proxy');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ 
  dev,
});
const handle = app.getRequestHandler();

// Backend addresses
const HTTP_BACKEND = process.env.HTTP_BACKEND || 'http://nexent-config:5010';
const WS_BACKEND = process.env.WS_BACKEND || 'ws://nexent-runtime:5014';
const RUNTIME_HTTP_BACKEND = process.env.RUNTIME_HTTP_BACKEND || 'http://nexent-runtime:5014';
const MINIO_BACKEND = process.env.MINIO_ENDPOINT || 'http://nexent-minio:9000';
const MARKET_BACKEND = process.env.MARKET_BACKEND || 'http://localhost:8010';
const PORT = 3000;

const proxy = createProxyServer({
  proxyTimeout: 15000,
  timeout: 15000,
  secure: false,
  changeOrigin: true
});

// Add error handling for proxy
proxy.on('error', (err, req, res) => {
  console.error('[Proxy] Proxy Error:', err.message);
  if (!res.headersSent) {
    res.writeHead(500, {
      'Content-Type': 'application/json',
    });
    res.end(JSON.stringify({
      error: 'Proxy error',
      message: err.message,
      code: 'PROXY_ERROR'
    }));
  }
});

// Global exception handler
process.on('uncaughtException', (err) => {
  console.error('[Global] Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[Global] Unhandled Rejection at:', promise, 'reason:', reason);
});

app.prepare().then(() => {
  const server = createServer((req, res) => {
    const parsedUrl = parse(req.url, true);
    const { pathname } = parsedUrl;
    
    console.log(`[Proxy] Incoming request: ${req.method} ${req.url}`);

    // Proxy HTTP requests
    if (pathname.includes('/attachments/') && !pathname.startsWith('/api/')) {
      console.log(`[Proxy] Routing to MinIO: ${MINIO_BACKEND}`);
      proxy.web(req, res, { target: MINIO_BACKEND }, (err) => {
        console.error('[Proxy] MinIO Proxy Error:', err.message);
        if (!res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ 
            error: 'Bad Gateway', 
            message: 'Unable to reach MinIO service',
            code: 'MINIO_SERVICE_UNAVAILABLE'
          }));
        }
      });
    } else if (pathname.startsWith('/runtime-api/')) {
      // Handle runtime API requests
      console.log(`[Proxy] Routing runtime-api request to: ${RUNTIME_HTTP_BACKEND}`);
      req.url = req.url.replace('/runtime-api', '/api');
      proxy.web(req, res, { target: RUNTIME_HTTP_BACKEND, changeOrigin: true }, (err) => {
        console.error('[Proxy] Runtime API Proxy Error:', err.message);
        if (!res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ 
            error: 'Bad Gateway', 
            message: 'Unable to reach runtime service',
            code: 'RUNTIME_SERVICE_UNAVAILABLE'
          }));
        }
      });
    } else if (pathname.startsWith('/api/market/')) {
      // Route market endpoints to market backend
      console.log(`[Proxy] Routing market request to: ${MARKET_BACKEND}`);
      req.url = req.url.replace('/api/market', '');
      proxy.web(req, res, { target: MARKET_BACKEND, changeOrigin: true }, (err) => {
        console.error('[Proxy] Market Proxy Error:', err.message);
        if (!res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ 
            error: 'Bad Gateway', 
            message: 'Unable to reach market service',
            code: 'MARKET_SERVICE_UNAVAILABLE'
          }));
        }
      });
    } else if (pathname.startsWith('/api/')) {
      // Explicitly route specific endpoints to appropriate services
      console.log(`[Proxy] Processing API request: ${req.method} ${req.url}`);
      console.log(`[Proxy] Pathname: ${pathname}`);
      console.log(`[Proxy] Request method: ${req.method}`);
      
      const isAgentRun = pathname.startsWith('/api/agent/run');
      const isAgentStop = pathname.startsWith('/api/agent/stop');
      const isConversation = pathname.startsWith('/api/conversation/');
      const isMemory = pathname.startsWith('/api/memory/');
      const isFilePreprocess = pathname.startsWith('/api/file/preprocess');
      const isFileStoragePost = (pathname.startsWith('/api/file/storage') && req.method === 'POST');
      
      console.log(`[Proxy] Route checks:`);
      console.log(`[Proxy]   isAgentRun: ${isAgentRun}`);
      console.log(`[Proxy]   isAgentStop: ${isAgentStop}`);
      console.log(`[Proxy]   isConversation: ${isConversation}`);
      console.log(`[Proxy]   isMemory: ${isMemory}`);
      console.log(`[Proxy]   isFilePreprocess: ${isFilePreprocess}`);
      console.log(`[Proxy]   isFileStoragePost: ${isFileStoragePost} (pathname.startsWith('/api/file/storage'): ${pathname.startsWith('/api/file/storage')})`);
      
      const routeToRuntime = (
        isAgentRun ||
        isAgentStop ||
        isConversation ||
        isMemory ||
        isFilePreprocess ||
        isFileStoragePost
      );
      
      console.log(`[Proxy] Should route to runtime: ${routeToRuntime}`);
      
      if (routeToRuntime) {
        // Route to runtime backend
        console.log(`[Proxy] Routing API request to Runtime Service: ${RUNTIME_HTTP_BACKEND}`);
        proxy.web(req, res, { target: RUNTIME_HTTP_BACKEND, changeOrigin: true }, (err) => {
          console.error('[Proxy] Runtime Service Proxy Error:', err.message);
          if (!res.headersSent) {
            res.writeHead(502, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
              error: 'Bad Gateway', 
              message: 'Unable to reach runtime service (' + err.message + ')',
              code: 'RUNTIME_SERVICE_UNAVAILABLE'
            }));
          }
        });
      } else {
        // Route to config backend
        console.log(`[Proxy] Routing API request to Config Service: ${HTTP_BACKEND}`);
        proxy.web(req, res, { target: HTTP_BACKEND, changeOrigin: true }, (err) => {
          console.error('[Proxy] Config Service Proxy Error:', err.message);
          if (!res.headersSent) {
            res.writeHead(502, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
              error: 'Bad Gateway', 
              message: 'Unable to reach config service (' + err.message + ')',
              code: 'CONFIG_SERVICE_UNAVAILABLE'
            }));
          }
        });
      }
    } else {
      // Let Next.js handle all other requests
      console.log(`[Proxy] Routing to Next.js`);
      handle(req, res, parsedUrl);
    }
  });

  // Proxy WebSocket upgrade requests
  server.on('upgrade', (req, socket, head) => {
    const { pathname } = parse(req.url);
    if (pathname.startsWith('/api/voice/')) {
      proxy.ws(req, socket, head, { target: WS_BACKEND, changeOrigin: true }, (err) => {
        console.error('[Proxy] WebSocket Proxy Error:', err.message);
        socket.destroy();
      });
    } else {
      console.log(`[Proxy] Ignoring non-voice WebSocket upgrade for: ${pathname}`);
    }
  });

  server.listen(PORT, '0.0.0.0', (err) => {
    if (err) throw err;
    console.log(`> Ready on http://0.0.0.0:${PORT}`);
    console.log('> --- Backend URL Configuration ---');
    console.log(`> HTTP Backend Target: ${HTTP_BACKEND}`);
    console.log(`> WebSocket Backend Target: ${WS_BACKEND}`);
    console.log(`> Runtime HTTP Backend Target: ${RUNTIME_HTTP_BACKEND}`);
    console.log(`> MinIO Backend Target: ${MINIO_BACKEND}`);
    console.log(`> Market Backend Target: ${MARKET_BACKEND}`);
    console.log('> ---------------------------------');
  });
});