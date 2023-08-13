import { Socket } from 'net'

const HOST = '127.0.0.1'

export class Http {
  private readonly socket: Socket

  public constructor () {
    this.socket = new Socket()
  }

  public serve (port: number) {
    this.socket.connect(port, HOST)

    this.socket.on('connect', () => {
      console.log(`Connected to ${HOST}:${port}`)
    })

    // this.socket.on('data', (data) => {
    //   data.
    // })
  }
}
