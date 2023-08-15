import { HTTPStatus, type StatusCode } from './types'

export class ParseError extends Error {}

export class RequestError extends Error {}

export class ResponseAlreadySent extends Error {
  public constructor () {
    super('Response already sent')
  }
}

export class HandlerNotFound extends Error {}

export class HTTPRequestException extends Error {
  public constructor (public readonly code: StatusCode, message: string) {
    super(message)
  }
}

export class BadRequestException extends HTTPRequestException {
  public constructor (message: string) {
    super(HTTPStatus.BAD_REQUEST, message)
  }
}
