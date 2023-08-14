import { HandlerNotFound } from '../exception'
import { type Request } from '../request'
import { type Response } from '../response'
import { HTTPStatus, type ErrorHandler, type Next } from '../types'

export const notFoundHandler: ErrorHandler = (request: Request, response: Response, next: Next, error: Error) => {
  if (error instanceof HandlerNotFound) {
    response.status(HTTPStatus.NOT_FOUND).sendEmpty()
  } else {
    next()
  }
}
