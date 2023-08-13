import { ParseError } from './exception'
import { Request } from './request'
import { type HTTPHeaders, HTTPMethod, type QueryParam } from './types'

export function parseHttpRequest (request: Buffer): Request {
  const stringRep = request.toString('utf-8').split('\r\n')

  if (stringRep.length !== 3) {
    throw new ParseError('Invalid http request format')
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [httpMethod, uri, ...rest] = stringRep[0].split(' ')

  const { base, query } = parseQuery(uri)

  if (!Object.values(HTTPMethod).includes(httpMethod as HTTPMethod)) {
    throw new ParseError('Invalid http request method')
  }

  const httpHeader = parseHttpHeader(stringRep[1])

  const body = Buffer.from(stringRep[2], 'utf-8')

  return new Request(httpMethod as HTTPMethod, base, query, httpHeader, body)
}

function parseHttpHeader (raw: string): HTTPHeaders {
  const headers: HTTPHeaders = new Map<string, string>()

  raw.split('\n').forEach(line => {
    const offset = line.indexOf(':')

    if (offset === -1) {
      throw new ParseError('Invalid http header format')
    }

    const value = line.slice(offset + 1)
    const key = line.slice(0, offset)
    headers.set(key, value)
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
