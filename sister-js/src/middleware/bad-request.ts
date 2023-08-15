import { type ErrorHandler } from '../http/types'
import { BadRequestException } from '../http/exception'

export const badRequestHandler: ErrorHandler = (request, response, next, error) => {
  if (error instanceof BadRequestException) {
    response.status(error.code).sendJson({
      message: error.message
    })
  } else {
    next()
  }
}
