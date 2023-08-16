import { type QueryParam, type HTTPHeaders, type HTTPMethod, ContentType } from './types'
import { RequestError } from './exception'
import { parseJson } from '../json/parser'
import { parseUrlEncode } from './parser/base'
import { type FormValue, parseMultipart } from './parser/multipart'

export class Request {
  public constructor (public readonly method: HTTPMethod, public readonly uri: string, public readonly query: QueryParam, public readonly headers: HTTPHeaders, readonly body: Buffer) {

  }

  public getContentType () {
    const header = this.headers.get('content-type')

    if (header === undefined) {
      throw new RequestError('Cannot find content-type header')
    }

    return header
  }

  public text (): string {
    const contentType = this.getContentType()

    if (contentType === ContentType.text) {
      return this.body.toString()
    }

    throw new RequestError('Content type is not text')
  }

  public json (): any {
    const contentType = this.getContentType()

    if (contentType === ContentType.json) {
      const raw = this.body.toString()
      return parseJson(raw)
    }

    throw new RequestError('Content type is not json')
  }

  public urlEncoded (): any {
    const contentType = this.getContentType()

    if (contentType === ContentType.urlEncoded) {
      return parseUrlEncode(this.body.toString())
    }

    throw new RequestError('Content type is not url encoded')
  }

  public formMultipart (): Map<string, FormValue> {
    const contentType = this.getContentType()

    if (contentType.startsWith(ContentType.formData)) {
      const boundaryRaw = contentType.split(';')[1].split('=')[1]

      let boundary: string
      if (boundaryRaw.startsWith('"') && boundaryRaw.endsWith('"')) {
        boundary = boundaryRaw.slice(1, -1)
      } else {
        boundary = boundaryRaw
      }

      const result: Map<string, FormValue> = new Map<string, FormValue>()
      parseMultipart(boundary, this.body).forEach(each => {
        result.set(each.name, each)
      })

      return result
    }

    throw new RequestError('Content type is not multipart form data')
  }
}
