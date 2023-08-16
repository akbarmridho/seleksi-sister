import { type ErrorHandler, HTTPStatus, type Next } from '../http/types'
import { type Request } from '../http/request'
import { type Response } from '../http/response'

export const errorHandler: ErrorHandler = (request: Request, response: Response, next: Next, error: Error) => {
  response.status(HTTPStatus.INTERNAL_SERVER_ERROR)

  if (process.env.NODE_ENV === 'production') {
    response.sendEmpty()
  } else {
    response.sendText(error.stack ?? error.message)
  }
}
