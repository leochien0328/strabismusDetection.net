declare module 'http-proxy-middleware' {
    import { RequestHandler } from 'express';
  
    export interface ProxyOptions {
      target: string;
      changeOrigin?: boolean;
      ws?: boolean;
      pathRewrite?: { [key: string]: string };
      router?: { [key: string]: string };
      logLevel?: 'debug' | 'info' | 'warn' | 'error' | 'silent';
      onProxyReq?: (proxyReq: any, req: any, res: any) => void;
      onProxyRes?: (proxyRes: any, req: any, res: any) => void;
      onError?: (err: any, req: any, res: any) => void;
    }
  
    export function createProxyMiddleware(options: ProxyOptions): RequestHandler;
  }
  