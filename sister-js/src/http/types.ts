import { type Request } from './request'
import { type Response } from './response'

export enum HTTPMethod {
  GET = 'GET',
  POST = 'POST',
  PUT = 'PUT',
  DELETE = 'DELETE'
}

export type HTTPHeaders = Map<string, string>

export type QueryParam = Record<string, string>

export interface StatusCode {
  status: string
  code: number
}

export const HTTPStatus = {
  OK: {
    status: 'OK',
    code: 200
  },
  BAD_REQUEST: {
    status: 'Bad Request',
    code: 400
  },
  UNAUTHORIZED: {
    status: 'Unauthorized',
    code: 401
  },
  NOT_FOUND: {
    status: 'Not Found',
    code: 404
  },
  INTERNAL_SERVER_ERROR: {
    status: 'Internal Server Error',
    code: 500
  }
}

export type Next = () => void

export type RequestHandler = (request: Request, response: Response) => Promise<void> | void

export type MiddlewareHandler = (request: Request, response: Response, next: Next) => Promise<void> | void

export type ErrorHandler = (request: Request, response: Response, next: Next, error: Error) => void

export interface Route {
  method: HTTPMethod
  endpoint: string
  handler: RequestHandler
}

export enum ContentType {
  text = 'text/plain',
  json = 'application/json'
}

export const ContentTypeHeader = 'Content-Type'
