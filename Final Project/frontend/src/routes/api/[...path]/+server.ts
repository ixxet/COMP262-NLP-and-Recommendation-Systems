import { env } from '$env/dynamic/private';
import type { RequestHandler } from './$types';

const API_BASE_URL = env.API_BASE_URL ?? 'http://localhost:8000';

async function proxy({ params, request }: Parameters<RequestHandler>[0]) {
  const target = `${API_BASE_URL}/${params.path}`;
  const headers = new Headers(request.headers);
  headers.delete('host');

  const response = await fetch(target, {
    method: request.method,
    headers,
    body: request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer()
  });

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers
  });
}

export const GET: RequestHandler = proxy;
export const POST: RequestHandler = proxy;
