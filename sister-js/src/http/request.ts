import { type QueryParam, type HTTPHeaders, type HTTPMethod } from './types'

export class Request {
  public constructor (public readonly method: HTTPMethod, public readonly uri: string, public readonly query: QueryParam, public readonly headers: HTTPHeaders, readonly body: Buffer) {

  }
}
