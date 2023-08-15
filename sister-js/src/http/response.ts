import { type Socket } from 'net'
import { type HTTPHeaders, HTTPStatus, type StatusCode, ContentTypeHeader, ContentType } from './types'
import { ResponseAlreadySent } from './exception'
import { jsonToString } from '../json/stringifier'

export class Response {
  private sent: boolean
  protected readonly httpVersion: string
  protected headers: HTTPHeaders

  public constructor (public code: StatusCode = HTTPStatus.OK, public readonly socket: Socket) {
    this.sent = false
    this.headers = new Map<string, string>()
    this.httpVersion = 'HTTP/1.1'
  }

  public status (status: StatusCode) {
    this.code = status
    return this
  }

  private sendHeaders () {
    this.socket.write(`${this.httpVersion} ${this.code.code} ${this.code.status}\r\n`)

    const headers: string[] = []

    this.headers.forEach((value, key) => {
      headers.push(`${key}:${value}`)
    })

    this.socket.write(`${headers.join('\r\n')}\r\n\r\n`)
  }

  public send (buffer: Uint8Array, encoding: BufferEncoding) {
    if (this.sent) {
      throw new ResponseAlreadySent()
    }

    this.sendHeaders()

    this.socket.write(buffer, encoding)
    this.sent = true
    this.socket.end()
  }

  public sendEmpty () {
    if (this.sent) {
      throw new ResponseAlreadySent()
    }

    this.sendHeaders()
    this.sent = true
    this.socket.end()
  }

  public sendText (body: string) {
    this.headers.set(ContentTypeHeader, ContentType.text)
    this.send(Buffer.from(body, 'utf-8'), 'utf-8')
  }

  public sendJson (body: any) {
    this.headers.set(ContentTypeHeader, ContentType.json)
    this.send(Buffer.from(jsonToString(body), 'utf-8'), 'utf-8')
  }
}
