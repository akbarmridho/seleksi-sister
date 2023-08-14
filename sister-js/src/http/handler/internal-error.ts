import { type Request } from '../request'
import { type Response } from '../response'
import { HTTPStatus, type ErrorHandler, type Next } from '../types'

export const errorHandler: ErrorHandler = (request: Request, response: Response, next: Next, error: Error) => {
  response.status(HTTPStatus.NOT_FOUND)

  if (process.env.NODE_ENV === 'production') {
    response.sendEmpty()
  } else {
    response.sendText(error.stack ?? error.message)
  }
}
