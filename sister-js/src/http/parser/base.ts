import { ParseError } from '../exception'
import { Request } from '../request'
import { type HTTPHeaders, HTTPMethod, type QueryParam } from '../types'

export function parseHttpRequest (request: Buffer): Request {
  const [baseHeaders, ...bodyRequest] = request.toString('binary').split('\r\n\r\n')

  const [head, ...headers] = baseHeaders.split('\r\n')

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [httpMethod, uri, ...rest] = head.split(' ')
  const { base, query } = parseQuery(uri)

  if (!Object.values(HTTPMethod).includes(httpMethod as HTTPMethod)) {
    throw new ParseError('Invalid http request method')
  }

  const httpHeader = parseHttpHeader(headers)

  const body = Buffer.from(bodyRequest.join('\r\n\r\n'), 'binary')

  return new Request(httpMethod as HTTPMethod, base, query, httpHeader, body)
}

function parseHttpHeader (raw: string[]): HTTPHeaders {
  const headers: HTTPHeaders = new Map<string, string>()

  raw.forEach(line => {
    const [key, ...rest] = line.split(':')
    headers.set(key.toLowerCase(), rest.join(':').trim())
  })

  return headers
}

function parseQuery (raw: string): {
  base: string
  query: QueryParam
} {
  const query: QueryParam = {}

  const splitted = raw.split('?')

  if (splitted.length !== 1) {
    const params = splitted[1].split('&')

    for (const param of params) {
      const [key, value] = param.split('=')
      query[key] = value
    }
  }

  return {
    base: splitted[0],
    query
  }
}

export function parseUrlEncode (raw: string): any {
  const result: any = {}

  const params = raw.split('&')

  for (const param of params) {
    const [key, value] = param.split('=')
    result[key] = value
  }

  return result
}
