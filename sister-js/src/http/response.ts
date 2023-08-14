import { type Socket } from 'net'
import { type HTTPHeaders, HTTPStatus, type StatusCode } from './types'
import { ResponseAlreadySent } from './exception'

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

    this.socket.write(`${headers.join('\n')}\r\n`)
  }

  public send (buffer: Uint8Array, encoding: BufferEncoding) {
    if (this.sent) {
      throw new ResponseAlreadySent()
    }

    this.sendHeaders()

    this.socket.write(buffer, encoding)
    this.sent = true
  }
}
