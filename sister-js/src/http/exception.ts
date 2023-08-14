export class ParseError extends Error {};

export class ResponseAlreadySent extends Error {
  public constructor () {
    super('Response already sent')
  }
}

export class HandlerNotFound extends Error {};
