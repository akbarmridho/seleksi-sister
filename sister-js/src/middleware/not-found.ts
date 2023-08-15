import { type ErrorHandler, HTTPStatus, type Next } from '../http/types'
import { type Request } from '../http/request'
import { type Response } from '../http/response'
import { HandlerNotFound } from '../http/exception'

export const notFoundHandler: ErrorHandler = (request: Request, response: Response, next: Next, error: Error) => {
  if (error instanceof HandlerNotFound) {
    response.status(HTTPStatus.NOT_FOUND).sendEmpty()
  } else {
    next()
  }
}
