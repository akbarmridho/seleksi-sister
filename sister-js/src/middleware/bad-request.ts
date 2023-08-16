import { type ErrorHandler } from '../http/types'
import { HTTPRequestException } from '../http/exception'

export const badRequestHandler: ErrorHandler = (request, response, next, error) => {
  if (error instanceof HTTPRequestException) {
    console.log(error.message)
    response.status(error.code).sendJson({
      message: error.message
    })
  } else {
    next()
  }
}
